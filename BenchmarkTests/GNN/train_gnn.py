import os
import json

import fire
import numpy as np

from custom_gnn_models import GCNNetwork
from utils import gnn_evaluation




def main(dataset:str, fold:bool):
    """
    Test graph neural networks that use standard MLPs against ones that use fold layers.
    
    Params:
        dataset (str): dataset to be used
        fold (bool): whether our GCN is a standard one or uses custom folding
    """
    # Save outputs
    log_path = 'BenchmarkTests/GNN/logs/' + dataset + '/'

    data_channels = {
        "ENZYMES": [3, 6],
        "QM9": [11, 19],
        "MNIST": [3, 10],
        "CIFAR10": [5, 10],
        "Cora": [1433, 7]
    }

    # Make model
    model = GCNNetwork(in_channels=data_channels[dataset][0], num_classes=data_channels[dataset][1])
    # Train model
    test_accs = gnn_evaluation(model, dataset)
    
    # Save output
    test_accs_path = log_path + "test_accuracies.txt"
    with open(test_accs_path, "w") as f:
        f.write((test_accs))



if __name__ == "__main__":
    # Fire allows adding command-line arguments to any function
    fire.Fire(main)