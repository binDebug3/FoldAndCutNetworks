# fundamentals
import sys
import warnings
import datetime as dt
import numpy as np
import pandas as pd

# datasets
from sklearn.datasets import load_breast_cancer, load_digits
from torchvision import datasets

# models
import torch
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# our files
from experimenter import *
import cnn_bench




# SETTINGS
all_benchmark_models = ["randomforest", "knn", "dl_fold", "dl_softfold", "dl_cnn", "dl_resnet", "metric"]
benchmark_models = all_benchmark_models[2:4]
all_benchmark_datasets = ["digits", "fashionMNIST", "cancer", "cifar10", "imagenet"]
benchmark_datasets = all_benchmark_datasets[:1]
default_ratio_list = [0.1, 0.2]


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_cifar10(astorch:bool=False, shuffle:bool=True, random_state:int=None, test_size:float=0.2, verbose:bool=0) -> tuple:
    """
    This function loads the cifar10 dataset from sklearn.
    Parameters:
        astorch (bool): default=False. If True, load the data as torch tensors.
        shufle (bool): default=True. If True, shuffle the data.
        random_state (int): default=None. The random state to use when splitting the data.
        test_size (float): default=0.2. The proportion of the data to use as the test set.
        verbose (int): default=0. If 1, print the shape of the data and the target.
    Returns:
        X_train (np.ndarray): The training data.
        X_test (np.ndarray): The testing data.
        y_train (np.ndarray): The training target.
        y_test (np.ndarray): The testing target.
    """
    batch1 = unpickle('data/cifar-10-batches-py/data_batch_1')
    X = batch1[b'data']
    y = np.array(batch1[b'labels'])

    # split data
    first_size = 50
    count = 0
    if shuffle and verbose > 1:
        print("\tShuffling:", end=" ")
    while True:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                            random_state=random_state, shuffle=shuffle)
        # make sure there are at least lmnn_default_params of each class in the training set
        if not shuffle or all([sum(y_train[:first_size] == uclass) >= lmnn_default_neighbors for uclass in set(y_train)]):
            break
        count += 1
        if verbose > 1:
            print(count, end=", ")
    if verbose > 1:
        print(f"\n\tTrain set has {len(set(y_train))} classes and test set has {len(set(y_test))} classes")
    if verbose > 0:
        print("\tX shape: ", X_train.shape)
        print("\ty shape: ", y_train.shape)
    
    if astorch:
        X_train = torch.tensor(X_train)
        X_test = torch.tensor(X_test)
        y_train = torch.tensor(y_train)
        y_test = torch.tensor(y_test)
    return X_train, X_test, y_train, y_test



def load_cancer(astorch:bool=False, shuffle:bool=True, random_state:int=None, test_size:float=0.2, verbose:int=0) -> tuple:
    """
    This function loads the breast cancer dataset from sklearn.
    Parameters:
        astorch (bool): default=False. If True, load the data as torch tensors.
        shuffle (bool): default=True. If True, shuffle the data.
        random_state (int): default=None. The random state to use when splitting the data.
        test_size (float): default=0.2. The proportion of the data to use as the test set.
        verbose (int): default=0. If 1, print the shape of the data and the target.
    Returns:
        X_train (np.ndarray): The training data.
        X_test (np.ndarray): The testing data.
        y_train (np.ndarray): The training target.
        y_test (np.ndarray): The testing target.
    """
    cancer = load_breast_cancer()
    X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y = cancer.target

    # split data
    n_classes = len(set(y))
    first_size = 20
    count = 0
    if shuffle and verbose > 1:
        print("\tShuffling:", end=" ")
    while True:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                            random_state=random_state, shuffle=shuffle)
        if not shuffle or len(set(y_train[:first_size])) >= n_classes-1 and len(set(y_test[:first_size])) >= n_classes-1:
            break
        count += 1
        if verbose > 1:
            print(count, end=", ")
    if verbose > 1:
        print(f"\n\tTrain set has {len(set(y_train))} classes and test set has {len(set(y_test))} classes")

    # scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if verbose > 0:
        print("\tX shape: ", X_train.shape)
        print("\ty shape: ", y_train.shape)
    
    if astorch:
        X_train = torch.tensor(X_train)
        X_test = torch.tensor(X_test)
        y_train = torch.tensor(y_train)
        y_test = torch.tensor(y_test)
    return X_train, X_test, y_train, y_test



def load_digits_data(astorch:bool=False, shuffle:bool=True, random_state:int=None, test_size:float=0.2, verbose:int=0) -> tuple:
    """
    This function loads the digits dataset from sklearn.
    Parameters:
        astorch (bool): default=False. If True, load the data as torch tensors.
        shuffle (bool): default=True. If True, shuffle the data.
        random_state (int): default=None. The random state to use when splitting the data.
        test_size (float): default=0.2. The proportion of the data to use as the test set.
        verbose (int): default=0. If 1, print the shape of the data and the target.
    Returns:
        X_train (np.ndarray): The training data.
        X_test (np.ndarray): The testing data.
        y_train (np.ndarray): The training target.
        y_test (np.ndarray): The testing target.
    """
    # digits_X, digits_y = load_digits(return_X_y=True)
    digits = load_digits()
    digits_X = digits.data
    digits_y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(digits_X, digits_y, test_size=test_size, 
                                                        random_state=random_state, shuffle=shuffle)

    if verbose > 0:
        print("\tX shape: ", X_train.shape)
        print("\ty shape: ", y_train.shape)
    
    if astorch:
        X_train = torch.tensor(X_train)
        X_test = torch.tensor(X_test)
        y_train = torch.tensor(y_train)
        y_test = torch.tensor(y_test)
    return X_train, X_test, y_train, y_test
    

def load_fashion(astorch:bool=False, shuffle:bool=True, random_state:int=None, test_size:float=0.2, verbose:int=0) -> tuple:
    """
    This function loads the fashion mnist dataset from torchvision.
    Parameters:
        astorch (bool): default=False. If True, load the data as torch tensors.
        shuffle (bool): default=True. If True, shuffle the data.
        random_state (int): default=None. The random state to use when splitting the data.
        test_size (float): default=0.2. The proportion of the data to use as the test set.
        verbose (int): default=0. If 1, print the shape of the data and the target.
    Returns:
        X_train (np.ndarray): The training data.
        X_test (np.ndarray): The testing data.
        y_train (np.ndarray): The training target.
        y_test (np.ndarray): The testing target.
    """
    if random_state is not None:
        warnings.warn("random_state is not implemented for this dataset, ignoring the provided value.")
    if test_size != 0.2:
        warnings.warn("test_size is not implemented for this dataset, ignoring the provided value.")
    
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    
    # Extract data and labels from the training set
    X_train = training_data.data.numpy()
    y_train = training_data.targets.numpy()

    # Extract data and labels from the test set
    X_test = test_data.data.numpy()
    y_test = test_data.targets.numpy()
    
    # shuffle data
    if shuffle:
        random_idx = np.random.permutation(len(X_train))
        X_train = X_train[random_idx]
        y_train = y_train[random_idx]
        random_idx = np.random.permutation(len(X_test))
        X_test = X_test[random_idx]
        y_test = y_test[random_idx]

    # Reshape X_train and X_test to be 2D arrays (flatten the 28x28 images)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    if verbose > 0:
        print("\tX shape: ", X_train.shape)
        print("\ty shape: ", y_train.shape)
    
    if astorch:
        X_train = torch.tensor(X_train)
        X_test = torch.tensor(X_test)
        y_train = torch.tensor(y_train)
        y_test = torch.tensor(y_test)
    return X_train, X_test, y_train, y_test
    

def load_imagenet(astorch:bool=False, random_state:int=None, test_size:float=0.2, verbose:int=0) -> tuple:
    """
    This function loads the imagenet dataset from torchvision.
    Parameters:
        astorch (bool): default=False. If True, load the data as torch tensors.
        random_state (int): default=None. The random state to use when splitting the data.
        test_size (float): default=0.2. The proportion of the data to use as the test set.
        verbose (int): default=0. If 1, print the shape of the data and the target.
    Returns:
        X_train (np.ndarray): The training data.
        X_test (np.ndarray): The testing data.
        y_train (np.ndarray): The training target.
        y_test (np.ndarray): The testing target.
    """
    raise NotImplementedError("This function is not implemented yet.")







config = {"verbose": 1,
          "random_state": 42,
          "test_size": 0.2,
          "datasets": 
                {"cifar10":
                    {"func": load_cifar10,
                    "data_ratios": default_ratio_list},
                "fashionMNIST":
                    {"func": load_fashion,
                    "data_ratios": default_ratio_list},
                "digits":
                    {"func": load_digits_data,
                    "data_ratios": default_ratio_list},
                "cancer":
                    {"func": load_cancer,
                    "data_ratios": default_ratio_list},
                "imagenet":
                    {"func": load_imagenet,
                    "data_ratios": default_ratio_list}
            }
        }



def test_model(model_name, date_time:str, dataset_name:str=None, astorch:bool=False, return_sizes:bool=False, repeat:int=5, verbose:int=0) -> dict:
    """
    This function tests a model on a dataset.
    Parameters:
        model_name (str): The name of the model to test.
        date_time (dt.datetime): The date and time of the test.
        dataset_name (str): default=None. The name of the dataset to test on. If none, test on all datasets.
        return_sizes (bool): default=False. If True, return the sample sizes used.
        repeat (int): default=5. The number of times to repeat the test.
        verbose (int): default=0. If 1, print the shape of the data and the target.
    Returns:
        model_results (dict): The results of the benchmark.
    """
    # validate inputs
    assert model_name in all_benchmark_models, f"model_name '{model_name}' must be one of {all_benchmark_models}"
    if dataset_name is not None:
        assert type(dataset_name) == str, "dataset must be a string"
        assert dataset_name in all_benchmark_datasets, f"dataset '{dataset_name}' must be one of {all_benchmark_datasets}"
        datasets = [dataset_name]
    else:
        datasets = benchmark_datasets
    assert type(date_time) == str, "date_time must be a string object"
    assert type(return_sizes) == bool, "return_sizes must be a boolean"
    assert type(verbose) == int, "verbose must be an integer"
        
    # iterate over datasets (recommended one)
    results = {}
    sample_size_list = []
    for dataset in datasets:
        if verbose > 0:
            print(f"Testing {model_name} on {dataset}")
        
        # load data
        data_loader = config["datasets"][dataset]["func"]
        X_train, X_test, y_train, y_test = data_loader(random_state=config["random_state"], 
                                                       test_size=config["test_size"], 
                                                       astorch=astorch, 
                                                       verbose=verbose)
        train_size = len(X_train)
        sample_sizes = [int(ratio*train_size) for ratio in config["datasets"][dataset]["data_ratios"]]
        sample_size_list.append(sample_sizes)
        experiment_info = (dataset_name, sample_sizes, X_train, y_train, X_test, y_test)
        
        # run benchmark
        model_results = benchmark_ml(model_name, experiment_info, date_time, repeat=repeat)
        results[dataset] = model_results
    
    
    # return results
    if len(datasets) == 1:
        if return_sizes:
            return results[dataset], sample_size_list[0]
        return results[dataset]
    if return_sizes:
        return results, sample_size_list
    return results