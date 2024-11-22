# fundamentals
import sys
import ssl
import os
import warnings
import json
import pickle
import gzip
import datetime as dt
import numpy as np      # type: ignore
import pandas as pd     # type: ignore

# datasets
from sklearn.datasets import load_breast_cancer, load_digits# type: ignore
from torchvision import datasets, transforms                # type: ignore
from pmlb import fetch_data                                 # type: ignore
from ucimlrepo import fetch_ucirepo                         # type: ignore
from urllib.request import urlretrieve                      # type: ignore
import tarfile
from sklearn.utils import shuffle as sk_shuffle

# models
import torch    # type: ignore
from torchvision.transforms import ToTensor                 # type: ignore
from sklearn.model_selection import train_test_split        # type: ignore
from sklearn.preprocessing import StandardScaler            # type: ignore

# our files
try:
    from experimenter import *
    import cnn_bench
except ImportError:
    from BenchmarkTests.experimenter import *
    from BenchmarkTests import cnn_bench

print("\nWorking Directory:", os.getcwd(), "\n")
onsup = 'SLURM_JOB_ID' in os.environ
config_path = "../BenchmarkTests/config.json" if onsup else "config.json"
architecture_path = "../BenchmarkTests/architectures.json" if onsup else "architectures.json"
data_path = "../data" if onsup else "../data"



def unpickle(file:str) -> dict:
    """
    This function loads a pickle file.
    Parameters:
        file (str): The file path.
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_pmlb_dataset(dataset_name:str, astorch:bool=False, shuffle:bool=True, random_state:int=None, test_size:float=0.2, verbose:int=0) -> tuple:
    """
    This function loads a dataset from the PMLB library.
        Examples Datasets are:
            "titanic", "sleep", "connect_4", "covtype", "mfeat_pixel", "dna"
            For more information on the datasets, see: https://epistasislab.github.io/pmlb/index.html
            This will show you the available datasets, their names, and if they're for classification or regression tasks
    Parameters:
            dataset_name (str): The name of the dataset to load.
            astorch (bool): default=False. If True, load the data as torch tensors.
            shuffle (bool): default=True. If True, shuffle the data.
            random_state (int): default=None. The random state to use when splitting the data.
            test_size (float): default=0.2. The proportion of the data to use as the test set.
            verbose (int): default=0. If 1, print the shape of the data and the target.
    Returns:
            X_train (np.ndarray) or (torch.tensor): The training data.
            X_test (np.ndarray) or (torch.tensor): The testing data.
            y_train (np.ndarray) or (torch.tensor): The training target.
            y_test (np.ndarray) or (torch.tensor): The testing target."""
    
    X,y = fetch_data(dataset_name, return_X_y=True)

    if shuffle and verbose > 1:
        print("\tShuffling:", end=" ")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                        random_state=random_state, shuffle=shuffle)
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
    cifar10_dir = os.path.join(data_path, "cifar-10-batches-py")

    # Check if CIFAR-10 data exists locally
    ### currently downloading cifar10 doesn't work ###
    if not os.path.exists(cifar10_dir):
        # Data not found, download it
        if verbose > 0:
            print("CIFAR-10 data not found locally. Downloading...")
        # Define the transformation to convert images to tensors
        transform = transforms.Compose([transforms.ToTensor()])

        # Download CIFAR-10 dataset
        ssl._create_default_https_context = ssl._create_unverified_context
        datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
        datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)

    if verbose > 1:
        print("CIFAR-10 dataset downloaded and extracted.")

    batch1 = unpickle(data_path + '/cifar-10-batches-py/data_batch_1')
    X = batch1[b'data']
    y = np.array(batch1[b'labels'])

    # split data
    if shuffle and verbose > 1:
        print("\tShuffling:", end=" ")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                            random_state=random_state, shuffle=shuffle)
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



def load_higgs(astorch=False, shuffle=True, random_state=None, test_size=0.2, verbose=0):
    """
    Loads the HIGGS dataset from 'HIGGS.csv' file.
    """
    import os

    # Path to HIGGS data
    higgs_path = os.path.join(data_path, "HIGGS.csv")

    # Check if the data file exists
    if not os.path.exists(higgs_path):
        # Data not found, download it
        if verbose > 0:
            print("HIGGS data not found locally. Downloading...")
        higgs_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz'
        gzip_path = os.path.join(data_path, 'HIGGS.csv.gz')

        ssl._create_default_https_context = ssl._create_unverified_context
        urlretrieve(higgs_url, gzip_path)
        # Unzip the file
        
        with gzip.open(gzip_path, 'rb') as f_in:
            with open(higgs_path, 'wb') as f_out:
                f_out.write(f_in.read())
        os.remove(gzip_path)  # Remove the gz file after extraction

    # Load the data
    data = pd.read_csv(higgs_path, header=None)
    X = data.iloc[:, 1:].values  # Features start from second column
    y = data.iloc[:, 0].values   # Labels are in the first column

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state, shuffle=shuffle)
    if verbose > 0:
        print("\tX shape:", X_train.shape)
        print("\ty shape:", y_train.shape)

    if astorch:
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train)
        y_test = torch.tensor(y_test)

    return X_train, X_test, y_train, y_test


def load_covertype(astorch=False, shuffle=True, random_state=None, test_size=0.2, verbose=0):
    """
    Loads the Covertype dataset from 'covtype.data' file.
    """

    # Path to Covertype data
    covtype_path = os.path.join(data_path, "covtype.data")

    # Check if the data file exists
    if not os.path.exists(covtype_path):
        # Data not found, download it
        if verbose > 0:
            print("Covertype data not found locally. Downloading...")
        ssl._create_default_https_context = ssl._create_unverified_context
        covertype = fetch_ucirepo(id=31) 
        X = covertype.data.features 
        y = covertype.data.targets 

    else:
        data = pd.read_csv(covtype_path, header=None)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state, shuffle=shuffle)
    if verbose > 0:
        print("\tX shape:", X_train.shape)
        print("\ty shape:", y_train.shape)
    
    if astorch:
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
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

    train_path = os.path.join(data_path, "mnist_train.csv")
    test_path = os.path.join(data_path, "mnist_test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        # Data not found, download it
        if verbose > 0:
            print("MNIST data not found locally. Downloading...")
        digits = load_digits()
        digits_X = digits.data
        digits_y = digits.target
        X_train, X_test, y_train, y_test = train_test_split(digits_X, digits_y, test_size=test_size, 
                                                            random_state=random_state, shuffle=shuffle)
    else:
        train_data = pd.read_csv(train_path, header=0, index_col = 0)
        test_data = pd.read_csv(test_path, header=0, index_col = 0)

        X_train = train_data.values
        y_train = train_data.index.values

        X_test = test_data.values
        y_test = test_data.index.values

        if shuffle:
            random_idx = np.random.permutation(len(X_train))
            X_train = X_train[random_idx]
            y_train = y_train[random_idx]
            random_idx = np.random.permutation(len(X_test))
            X_test = X_test[random_idx]
            y_test = y_test[random_idx]

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

    # Paths to Fashion MNIST data
    train_path = os.path.join(data_path, "fashion-mnist_train.csv")
    test_path = os.path.join(data_path, "fashion-mnist_test.csv")
    print("\n*** Working directory", os.getcwd())
    print("\n*** Test path:", test_path)

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        # Data not found, download it
        if verbose > 0:
            print("Fashion MNIST data not found locally. Downloading...")
    
        training_data = datasets.FashionMNIST(
            root=data_path,
            train=True,
            download=True,
            transform=ToTensor()
        )

        test_data = datasets.FashionMNIST(
            root=data_path,
            train=False,
            download=True,
            transform=ToTensor()
        )


        # Paths to the downloaded files
        train_images_path = data_path + "/FashionMNIST/raw/train-images-idx3-ubyte.gz"
        train_labels_path = data_path + "/FashionMNIST/raw/train-labels-idx1-ubyte.gz"
        test_images_path = data_path + "/FashionMNIST/raw/t10k-images-idx3-ubyte.gz"
        test_labels_path = data_path + "/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz"

        # Read the data
        train_images = read_idx(train_images_path)
        train_labels = read_idx(train_labels_path)
        test_images = read_idx(test_images_path)
        test_labels = read_idx(test_labels_path)

        train_data = pd.DataFrame(train_images)
        train_data.insert(0, 'label', train_labels)  # Add labels as the first column

        test_data = pd.DataFrame(test_images)
        test_data.insert(0, 'label', test_labels)  # Add labels as the first column


        train_csv_path = os.path.join(data_path, "fashion-mnist_train.csv")
        test_csv_path = os.path.join(data_path, "fashion-mnist_test.csv")

        train_data.to_csv(train_csv_path, header=0, index_col=0)
        test_data.to_csv(test_csv_path, header=0, index_col=0)

    else:
       train_data = pd.read_csv(train_path, header=0, index_col=0)
       test_data = pd.read_csv(test_path, header=0, index_col=0)

    X_train = train_data.values
    y_train = train_data.index.values

    X_test = test_data.values
    y_test = test_data.index.values

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
    

def load_imagenet(astorch:bool=False, random_state:int=None, test_size:float=0.2, errors="ignore", verbose:int=0) -> tuple:
    """
    This function loads the imagenet dataset from torchvision.
    Parameters:
        astorch (bool): default=False. If True, load the data as torch tensors.
        random_state (int): default=None. The random state to use when splitting the data.
        test_size (float): default=0.2. The proportion of the data to use as the test set.
        errors (str): default="ignore". How to handle errors when loading the data.
        verbose (int): default=0. If 1, print the shape of the data and the target.
    Returns:
        X_train (np.ndarray): The training data.
        X_test (np.ndarray): The testing data.
        y_train (np.ndarray): The training target.
        y_test (np.ndarray): The testing target.
    """
    if errors == "raise":
        raise NotImplementedError("This function is not implemented yet.")
    if errors == "flag":
        print("Load_imagenet is not implemented yet.")
    return None, None, None, None



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
    with open(config_path, "r") as f:
        settings = json.load(f)
    with open(architecture_path, "r") as f:
        architectures = json.load(f)
    all_benchmark_models = settings["all_benchmark_models"]
    benchmark_datasets = settings["benchmark_datasets"]
    ratio_list = settings["default_ratio_list"]
    rs = settings["random_state"]
    rs = None if rs == 0 else rs
    test_size = settings["test_size"]
    config = {"cifar10": load_cifar10,
            "fashionMNIST": load_fashion,
            "digits": load_digits_data,
            "cancer": load_cancer,
            "HIGGS": load_higgs,
            "covtype": load_covertype,
            "imagenet": load_imagenet}
    
    assert model_name in all_benchmark_models or model_name in list(architectures.keys()), \
        f"model_name '{model_name}' must be one of {all_benchmark_models} or in '{architecture_path}'"
    if dataset_name is not None:
        assert type(dataset_name) == str, "dataset must be a string"
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
            print(f"\n\n|||||||\nTesting {model_name} on {dataset}")
        
        # load data
        data_loader = config.get(dataset, load_pmlb_dataset)
        if data_loader == load_pmlb_dataset:
            X_train, X_test, y_train, y_test = data_loader(dataset, random_state=rs, test_size=test_size, 
                                                        astorch=astorch, verbose=verbose)
        else:
            X_train, X_test, y_train, y_test = data_loader(random_state=rs, test_size=test_size, 
                                                        astorch=astorch, verbose=verbose)
        train_size = len(X_train)
        sample_sizes = [int(ratio*train_size) for ratio in ratio_list]
        sample_size_list.append(sample_sizes)
        experiment_info = (dataset_name, sample_sizes, X_train, y_train, X_test, y_test)
        
        # run benchmark
        model_results = benchmark_ml(model_name, experiment_info, date_time, repeat=repeat, verbose=verbose)
        results[dataset] = model_results
    
    
    # return results
    if len(datasets) == 1:
        if return_sizes:
            return results[dataset], sample_size_list[0]
        return results[dataset]
    if return_sizes:
        return results, sample_size_list
    return results


if __name__ == "__main__":
    load_digits_data(verbose=1)
