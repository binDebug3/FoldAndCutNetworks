# cnn imports
import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split, TensorDataset
import torch.optim as optim
from torchvision.transforms import ToTensor
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

#benchmarking imports
import sys
import os
import time
import datetime as dt
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_model(model_name:str, params:dict=None):
    """
    Returns a new instance of the CNN model

    Parameters:
        model_name (str): The name of the model to train.
        params (dict): The hyperparameters to use for the acme model.
    
    Returns:
        model: The model to train
    """
    input_channels = params.get('input_channels', 1)
    return CNNModel(input_channels)



#define a CNN 
class CNNModel(nn.Module):
    def __init__(self, input_channels, input_size=28):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        
        # Dynamically calculate the size after convolution and pooling
        conv_output_size = (input_size - 4) // 2
        if conv_output_size < 2:
            conv_output_size = (input_size - 4)

        self.pool = nn.MaxPool2d(2, 2)
        
        self.linear1 = nn.Linear(64 * conv_output_size * conv_output_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x))) if x.shape[2] > 2 and x.shape[3] > 2 else torch.relu(self.conv2(x))
        
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


def load_and_prepare_data(dataset_name, batch_size=64, val_split=0.1):
    """
    Load and prepare the dataset, returning data loaders for training, validation, and testing.

    Parameters:
        dataset_name (str): The name of the dataset ('mnist', 'fashion-mnist', 'cifar10')
        batch_size (int): The batch size for the data loaders
        val_split (float): The proportion of the training data to use for validation

    Returns:
        train_loader (DataLoader): DataLoader for the training data
        val_loader (DataLoader): DataLoader for the validation data
        test_loader (DataLoader): DataLoader for the test data
        input_channels (int): The number of input channels (1 for grayscale, 3 for RGB)
    """

    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        #digits dataset
        digits = load_digits()
        X, y = digits['data'], digits['target']

        # Convert the data to float32 and the labels to int64
        X = X.astype('float32')
        y = y.astype('int64')

        # Normalize the pixel values to the range [0, 1]
        X /= 16.0  # Since the pixel values in load_digits range from 0 to 16

        # Reshape the data to match PyTorch's expectations (N, C, H, W)
        X = X.reshape(-1, 1, 8, 8)
        input_channels = 1

        # First, split into train and test sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Then, split the training set further into train and validation sets (90% train, 10% validation)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_split, random_state=42)

        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train)
        y_train = torch.tensor(y_train)
        X_val = torch.tensor(X_val)
        y_val = torch.tensor(y_val)
        X_test = torch.tensor(X_test)
        y_test = torch.tensor(y_test)

        # Create TensorDataset and DataLoader for MNIST
        train_subset = TensorDataset(X_train, y_train)
        val_subset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)

    elif dataset_name == 'fashion-mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.FashionMNIST('./', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST('./', train=False, transform=transform)
        input_channels = 1

        # Calculate the number of samples for validation and training
        val_size = int(val_split * len(train_dataset))
        train_size = len(train_dataset) - val_size

        # Split the training dataset into training and validation sets
        train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        train_dataset = datasets.CIFAR10('./', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('./', train=False, transform=transform)
        input_channels = 3

        # Calculate the number of samples for validation and training
        val_size = int(val_split * len(train_dataset))
        train_size = len(train_dataset) - val_size

        # Split the training dataset into training and validation sets
        train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    else:
        raise ValueError("Dataset not supported: choose 'mnist', 'fashion-mnist', or 'cifar10'.")

    

    # Create DataLoaders for each set
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_subset, val_loader, test_loader, input_channels


import torch.optim as optim
import numpy as np
import time

def run_cnn(model, full_train_data, val_loader, test_loader, device, data_sizes, epochs=10, repeat=5):
    """
    Train the CNN model and track training and validation accuracy and training time over different data sizes.

    Parameters:
        model_class: The class of the model (e.g., CNNModel)
        full_train_data: Full training dataset
        val_loader: Validation data loader
        test_loader: Test data loader
        device: Device to run the model on (CPU or GPU)
        data_sizes: List of different data sizes to train on
        epochs: Number of epochs to train the model
        repeat: Number of times to repeat the experiment for std calculation
    
    Returns:
        dict: A dictionary containing mean and std of training accuracy, validation accuracy, and training time for each data size
    """
    results_dict = {"cnn" : {}}

    train_accs = []
    val_accs = []
    times = []

    progressB = tqdm(total=repeat*len(data_sizes), desc=f"Benchmarking CNN")
    for i in range(repeat):
        train_acc_list = []
        val_acc_list = []
        time_list = []

        for size in data_sizes:
            # Create a subset of the training data
            subset_indices = list(range(size))
            train_subset = torch.utils.data.Subset(full_train_data, subset_indices)
            train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)

            # Reinitialize the model for each repetition and data size
            model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
            model.to(device)
            optimizer = optim.Adam(model.parameters())
            criterion = nn.CrossEntropyLoss()

            start_time = time.perf_counter()

            for epoch in range(epochs):
                model.train()
                correct_train = 0

                for batch, (data, label) in enumerate(train_loader):
                    data, label = data.to(device), label.to(device)
                    optimizer.zero_grad()
                    output = model(data)

                    loss = criterion(output, label)
                    loss.backward()
                    optimizer.step()

                    pred = output.argmax(dim=1, keepdim=True)
                    correct_train += pred.eq(label.view_as(pred)).sum().item()

                train_accuracy = correct_train / len(train_loader.dataset)

                # Evaluate on validation set
                model.eval()
                correct_val = 0
                with torch.no_grad():
                    for batch, (data, label) in enumerate(val_loader):
                        data, label = data.to(device), label.to(device)
                        output = model(data)
                        pred = output.argmax(dim=1, keepdim=True)
                        correct_val += pred.eq(label.view_as(pred)).sum().item()

                val_accuracy = correct_val / len(val_loader.dataset)

            end_time = time.perf_counter()
            training_time = end_time - start_time

            # Store results for this size
            time_list.append(training_time)
            train_acc_list.append(train_accuracy)
            val_acc_list.append(val_accuracy)
            progressB.update(1)
            
        # Store the results of this repetition
        train_accs.append(np.array(train_acc_list))
        val_accs.append(np.array(val_acc_list))
        times.append(np.array(time_list))
        


    progressB.close()
    # Calculate mean and std across all repetitions
    train_acc_mean = np.mean(train_accs, axis=0)
    val_acc_mean = np.mean(val_accs, axis=0)
    time_mean = np.mean(times, axis=0)

    train_acc_std = np.std(train_accs, axis=0)
    val_acc_std = np.std(val_accs, axis=0)
    time_std = np.std(times, axis=0)

    results_dict['cnn']["mean"] = {"train_acc": train_acc_mean, "val_acc": val_acc_mean, "time": time_mean}
    results_dict['cnn']["std"] = {"train_acc": train_acc_std, "val_acc": val_acc_std, "time": time_std}

    return results_dict


def save_cnn_data(data: np.ndarray, dataset_name: str, model_name: str, info_type: str, iteration: int, datetime: str, val: bool=False):
    """
    Save the CNN results to a numpy file.

    Parameters:
        data (np.ndarray): Data to save.
        dataset_name (str): Name of the dataset.
        model_name (str): Name of the model (e.g., "cnn").
        info_type (str): Type of information (e.g., "acc", "time").
        iteration (int): The number of times this model configuration has been repeated.
        datetime (str): Current date and time.
        val (bool): Whether the data is validation data.
    """
    dir = f"results/{dataset_name}/{model_name}/npy_files"
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    train_val = "train" if not val else "val"
    filename = f"{train_val}_{info_type}_i{iteration}_d{datetime}.npy"
    path = os.path.join(dir, filename)
    
    np.save(path, data)
    print(f"Data saved to {path}")

def plot_results(results):
    """
    Plot the benchmarking results, including training accuracy, validation accuracy, and training time over different data sizes.

    Parameters:
        results (dict): A dictionary containing the training accuracy, validation accuracy, and training time for each data size.
    """

    data_sizes = results['data_sizes']
    train_acc = results['train_acc']
    val_acc = results['val_acc']
    train_time = results['train_time']

    plt.figure(figsize=(18, 6))

    # Training Accuracy Plot
    plt.subplot(1, 3, 1)
    plt.plot(data_sizes, train_acc, label='Train Accuracy', marker='o', color='tab:green')
    plt.xlabel('Data Size')
    plt.ylabel('Accuracy (%)')
    plt.title('train_acc')
    plt.ylim(0.7, 1.02)
    plt.axhline(y=1, color='k', linestyle='--')
    plt.legend()

    # Validation Accuracy Plot
    plt.subplot(1, 3, 2)
    plt.plot(data_sizes, val_acc, label='Val Accuracy', marker='o', color='tab:orange')
    plt.xlabel('Data Size')
    plt.ylabel('Accuracy (%)')
    plt.title('val_acc')
    plt.ylim(0.7, 1.02)
    plt.axhline(y=1, color='k', linestyle='--')
    plt.legend()

    # Training Time Plot
    plt.subplot(1, 3, 3)
    plt.plot(data_sizes, train_time, label='Training Time', marker='o', color='tab:red')
    plt.xlabel('Data Size')
    plt.ylabel('Training Time (s)')
    plt.title('time')
    plt.yscale('log')  # Logarithmic scale for training time
    plt.legend()

    plt.suptitle('Model Benchmarking')
    plt.tight_layout()
    plt.show()

