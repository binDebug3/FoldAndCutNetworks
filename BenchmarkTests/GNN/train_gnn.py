import os
import json

import fire
import numpy as np

import sys
sys.path.append("/home/harrisab/FoldCut/FoldAndCutNetworks/")
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from custom_gnn_models import GCNNetwork
from utils import gnn_evaluation



def main(dataset:str, fold=False, layers=[1], hidden=[32]):
    """
    Test graph neural networks that use standard MLPs against ones that use fold layers.
    
    Params:
        dataset (str): dataset to be used
        fold (bool): whether our GCN is a standard one or uses custom folding
        layers (list:int): number of layers to experiment with using
        hidden (list:int): size of the hidden layers
    """
    # Save outputs
    log_path = 'BenchmarkTests/GNN/logs/' + dataset + '/'

    # Train model
    test_accs = gnn_evaluation(GCNNetwork, dataset, fold=fold, )
    
    # Save output
    try:
        test_accs_path = log_path + "test_accuracies.txt"
        with open(test_accs_path, "w") as f:
            f.write((test_accs))
    except FileNotFoundError:
        test_accs_path = "test_accuracies.txt"
        with open(test_accs_path, "w") as f:
            f.write((test_accs))



if __name__ == "__main__":
    # Fire allows adding command-line arguments to any function
    fire.Fire(main)