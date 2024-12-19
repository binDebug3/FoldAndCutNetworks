import os
import os.path as osp
import json
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset, AttributedGraphDataset, QM9
from torch_geometric.utils import degree


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


# One training epoch for GNN model.
def train(train_loader, model, ds_name, optimizer, device):
    model.train() # training mode
    
    for i, data in enumerate(train_loader):
        # print(i / len(train_loader))
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)

        # Set the correct loss for each dataset we use
        if ds_name in {"ENZYMES", "Cora", "MNIST", "CIFAR10"}:
            loss = F.cross_entropy(output, data.y)
        elif ds_name in {"QM9"}:
            loss = F.mse_loss(output, data.y)
        else:
            raise Exception("Dataset does not have loss specified in train()")
        loss.backward()
        optimizer.step()


# Get acc. of GNN model.
def test(loader, model, device):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


# 5-CV for GNN training and hyperparameter selection.
def gnn_evaluation(gnn, ds_name, layers=[1], hidden=[32], fold=False, max_num_epochs=200, batch_size=128, start_lr=0.01, 
                   min_lr = 0.000001, factor=0.5, patience=5, all_std=True):
    """
    Run an evaluation of our GNN.
    
    Parameters:
        gnn (class): uninitialized model class
        ds_name (str): name of dataset to use
        layers (list): list of numbers of layers in the model (each is tested)
        hidden (list): list of sizes for the hidden layers (each is tested)
        fold (bool): whether to 
    """
    # Load dataset and shuffle.
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'datasets', ds_name)
    # Get parent data housing from dataset name
    data_parents = {
        "ENZYMES": "TUDataset",
        "QM9": "QM9",
        "MNIST": "GNNBenchmarkDataset",
        "CIFAR10": "GNNBenchmarkDataset",
        "Cora": "AttributedGraphDataset"
    }
    data_parent = data_parents[ds_name]
    # Load dataset
    if data_parent == "TUDataset":
        dataset = TUDataset(path, name=ds_name).shuffle()
    elif data_parent == "QM9":
        dataset = QM9(path)
    elif data_parent == "GNNBenchmarkDataset":
        dataset = GNNBenchmarkDataset(path, name=ds_name).shuffle()
    elif data_parent == "AttributedGraphDataset":
        dataset = AttributedGraphDataset(path, name=ds_name).shuffle()
    else:
        raise Exception(f"Unable to load datastet {ds_name}")
    if ds_name in {"MNIST", "CIFAR10", "QM9", "ENZYMES"}:
        graph_level_task = True
    else: 
        graph_level_task = False

    # One-hot degree if node labels are not available.
    # The following if clause is taken from  https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/datasets.py.
    if dataset.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

    # Set device.
    # device = torch.device('cpu') 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Set up cross validation
    if ds_name == "Cora":
        num_folds = 1 # Cora only has one graph
    else:
        num_folds = 5

    kf = KFold(n_splits=num_folds, shuffle=True)
    dataset.shuffle()

    test_accuracies_all = []
    test_accuracies_complete = []

    # Collect val. and test acc. over all hyperparameter combinations.
    for l in layers:
        for h in hidden:

            experiment_report = {"Layers": l, "Hidden Size": h}
            test_accuracies = []

            for train_val_index, test_index in kf.split(list(range(len(dataset)))):
                # Sample 10% split from training split for validation.
                train_index, val_index = train_test_split(train_val_index, test_size=0.1)
                best_val_acc = 0.0
                best_test = 0.0
                
                # Split data.
                train_dataset = dataset[train_index.tolist()]
                val_dataset = dataset[val_index.tolist()]
                test_dataset = dataset[test_index.tolist()]

                # Prepare batching.
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

                # Setup model.
                model = gnn(dataset.num_features, hidden_channels=h, num_layers=l, num_classes=dataset.num_classes, 
                            graph_level_task=graph_level_task, fold=fold).to(device)
                num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                try:
                    model.reset_parameters()
                except:
                    pass

                optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                       factor=factor, patience=patience,
                                                                       min_lr=0.0000001)
                for epoch in range(1, max_num_epochs + 1):
                    lr = scheduler.optimizer.param_groups[0]['lr']
                    train(train_loader, model, ds_name, optimizer, device)
                    val_acc = test(val_loader, model, device)
                    print(f"Epoch {epoch}; Validation accuracy: {val_acc}")
                    scheduler.step(val_acc)

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        # best_test = best_val_acc
                        best_test = test(test_loader, model, device) * 100.0

                    # Break if learning rate is smaller 10**-6.
                    if lr < min_lr:
                        break

                test_accuracies.append(best_test)
                if all_std:
                    test_accuracies_complete.append(best_test)

            test_accuracies_all.append(float(np.array(test_accuracies).mean()))
            experiment_report["Average Best Accuracy"] = float(np.array(test_accuracies).mean())
            experiment_report["Parameters"] = num_params

            # Save to a JSON file
            file_index = 0
            if fold:
                fold_style = "Fold" 
            else:
                fold_style = "NoFold"
            while True: # index to avoid overwriting files
                file_name = f"BenchmarkTests/GNN/experiments/{ds_name}_{fold_style}_{file_index}.json"
                if not os.path.exists(file_name):
                    with open(file_name, "w") as json_file:
                        json.dump(experiment_report, json_file, indent=4)
                    break
                file_index += 1

    if all_std:
        return np.array(test_accuracies_all)
    else:
        return np.array(test_accuracies_all)
    