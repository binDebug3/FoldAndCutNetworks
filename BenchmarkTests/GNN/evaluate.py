from BenchmarkTests.GNN.utils import gnn_evaluation
from BenchmarkTests.GNN.custom_gnn_models import GINNetwork, FoldGINNetwork
from torch_geometric.nn.models import GIN


dataset = "MNIST" # "ENZYMES", "Cora", "MNIST", "CIFAR10", "QM9"

hyperparam_accuracy = gnn_evaluation(GINNetwork, dataset, [2, 3, 4, 5], 
                                     [32, 64, 128], max_num_epochs=200, 
                                     batch_size=64, start_lr=0.01, 
                                     all_std=True)

print(hyperparam_accuracy.min())
print(hyperparam_accuracy.mean())
print(hyperparam_accuracy.max())