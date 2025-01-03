import sys
import os
import time
import json
import datetime as dt
import math
import gzip
import numpy as np                                  # type: ignore
from tqdm import tqdm                               # type: ignore
import pdb
try:
    from jeffutils.utils import stack_trace         # type: ignore
except ImportError:
    pass

# plotting imports
import matplotlib.pyplot as plt                     # type: ignore
import plotly.graph_objects as go                   # type: ignore
from plotly.subplots import make_subplots           # type: ignore

# ml imports
from sklearn.metrics import accuracy_score          # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.neighbors import KNeighborsClassifier  # type: ignore
#from metric_learn import LMNN                       # type: ignore
from sklearn.utils import shuffle                   # type: ignore

#deep learning imports
try:
    from cnn_bench import CNNModel
except ImportError:
    from BenchmarkTests.cnn_bench import CNNModel

# our model imports
sys.path.append('../')
from models.model_bank import *
from models.training import *

onsup = 'SLURM_JOB_ID' in os.environ
config_path = "../BenchmarkTests/config.json" if onsup else "config.json"
architecture_path = "../BenchmarkTests/architectures.json" if onsup else "architectures.json"
data_path = "../data" if onsup else "../data"



### HELPER FUNCTIONS ###

class InvalidModelError(ValueError):
    pass

def get_last_file(dir:str, partial_name:str, insert:str="_d", file_type:str="npy"):
    """
    Get the last file in a directory matching the partial name

    Parameters:
    partial_name (str): partial name of the file
    dir (str): directory to search in
    insert (str): string to insert before the datetime
    file_type (str): type of file to search for

    Returns:
    str: path to the last file
    """
    # get all the datetimes
    date_format = json.load(open(config_path)).get("date_format")
    shift = 0 if insert == "_" else 1
    datetimes = [name.split("_")[-1].split(".")[0][shift:] for name in os.listdir(dir) if partial_name in name]
    
    # get the latest datetime by converting to datetime object
    vals = [dt.datetime.strptime(date, date_format) for date in datetimes]
    if len(vals) == 0:
        return None
    
    # get the latest datetime and return the path
    latest_datetime = max(vals).strftime(date_format)
    return os.path.join(dir, f"{partial_name}{insert}{latest_datetime}.{file_type}")



def build_dir(dataset_name:str, model_name:str):
    """
    Build the directory to save the numpy files
    Parameters:
        dataset_name (str): name of the dataset
        model_name (str): name of the model
    Returns:
        str: directory
    """
    path = f"../data/results/{dataset_name}/{model_name}/npy_files"
    if onsup:
        path = f"../{path}"
    return path



def build_name(val:bool, info_type:str, iteration):
    """
    Build the name of the numpy file
    Parameters:
        val (bool): whether the data is validation data
        info_type (str): type of information
        iteration (int): iteration number
    Returns:
        str: name of the numpy file
    """
    train = "train" if not val else "val"
    return f"{train}_{info_type}_i{iteration}"



def load_result_data(dataset_name:str, model_name:str, info_type:str, iteration:int, 
              val:bool=False, verbose:int=0):
    """
    Load the data from the most recent numpy file matching the partial name
    Parameters:
        dataset_name (str): name of the dataset
        model_name (str): name of the model
        info_type (str): type of information (loss, acc, time, params)
        iteration (int): iteration number
        val (bool): whether the data is validation data
        verbose (int): whether to print the error message
    Returns:
        np.ndarray: data from the numpy file or None if the file does not exist
    """
    dir = build_dir(dataset_name, model_name)
    partial_name = build_name(val, info_type, iteration)
    
    # check to see if we want to get the std
    if info_type in ["acc", "loss", "time"]:
        do_std = True
    else:
        do_std = False
    
    try:
        return np.load(get_last_file(dir, partial_name)), do_std
    except Exception as e:
        if verbose > 1:
            print("Error: file not found")
            try:
                print(stack_trace(e))
            except NameError:
                print(e)
        return None, False
    


def save_data(data:np.ndarray, save_constants:tuple, info_type:str, iteration:int, 
              val:bool=False, refresh:bool=False, repeat:int=5) -> None:
    """
    Save the data to a numpy file
    Parameters:
        data (np.ndarray): data to save
        save_constants (tuple): contains
            dataset_name (str): name of the dataset
            model_name (str): name of the model
            datetime (str): current date and time
        info_type (str): type of information
        iteration (int): the number of times this model config has been repeated
        val (bool): whether the data is validation data
        refresh (bool): whether to refresh the last file
        repeat (int): number of times to repeat the experiment
    """
    #test comment so I can push to github
    dataset_name, model_name, datetime = save_constants
    dir = build_dir(dataset_name, model_name)
    partial_name = build_name(val, info_type, iteration)
    if not os.path.exists(dir):
        os.makedirs(dir)
        print("Made a new directory", dir)
    
    # check if the directory has files
    info_list = json.load(open(config_path)).get("info_list")
    if len(os.listdir(dir)) < len(info_list) * repeat:
        refresh = False

    # remove the last file with the same partial name
    if refresh:
        last_file = get_last_file(dir, partial_name)
        if last_file is not None:
            os.remove(last_file)

    path = f"results/{dataset_name}/{model_name}/npy_files/{partial_name}_d{datetime}.npy"
    if onsup:
        path = "../" + path
    np.save(path, data)
    return None


def delete_data(dataset_name:list=None, model_name:list=None, 
                after:dt.datetime=None, before:dt.datetime=None,
                safety_sleep:int=10, verbose:int=2) -> None:
    """
    Delete locally saved data from the numpy files
    Parameters:
        dataset_name (list): list of dataset names to clear
        model_name (list): list of model names to clear
        after (dt.datetime): clear data after this date
        before (dt.datetime): clear data before this date
        verbose (int): verbosity level
    """
    # input validation
    assert dataset_name is None or \
            type(dataset_name) == str or \
            (type(dataset_name) == list and all([type(name) == str for name in dataset_name])), \
            f"'dataset_name' must be a string or a list of strings not {type(dataset_name)}"
    assert model_name is None or \
            type(model_name) == str or \
            (type(model_name) == list and all([type(name) == str for name in model_name])), \
            f"'model_name' must be a string or a list of strings not {type(model_name)}"
    assert before is None or \
            type(before) == dt.datetime, \
            f"'before' must be a datetime object not {type(before)}"
    assert after is None or \
            type(after) == dt.datetime, \
            f"'after' must be a datetime object not {type(after)}"
    if after is not None and before is not None:
        assert after < before, f"'after' ({after}) must be before 'before' ({before})"
    assert type(safety_sleep) == int, f"'safety_sleep' must be an integer not {safety_sleep}"
    assert safety_sleep > 2, f"'safety_sleep' must be an integer over 2 not {safety_sleep}"
    
    if dataset_name is None:
        print("Warning! Deleting all data for all datasets")
        time.sleep(1)
    if model_name is None:
        print("Warning! Deleting all data for all models")
        time.sleep(1)
    if after is None and before is None:
        print("Warning! Deleting data for all dates")
        time.sleep(1)
    
    date_format = json.load(open(config_path)).get("date_format")
    dataset_names = [dataset_name] if type(dataset_name) == str else dataset_name
    model_names = [model_name] if type(model_name) == str else model_name
    dataset_names = dataset_names if dataset_names is not None else os.listdir("results")
    possible_models = json.load(open(config_path)).get("possible_models")
    model_names = model_names if model_names is not None else possible_models
    
    print("Deleting data...")
    for i in range(safety_sleep):
        print(f"{safety_sleep-i-1}", end="\r")
        time.sleep(1)
    if verbose > 0:
        print(f"From datasets: \t{dataset_names}")
        print(f"From models: \t{model_names}")
        after_print = after if after is not None else "beginning of time"
        before_print = before if before is not None else "end of time"
        print(f"Created between: {after_print} and {before_print}")
        print()
    for i in range(safety_sleep):
        print(f"{safety_sleep-i-1}", end="\r")
        time.sleep(1)
    
    count = 0
    retained = 0
    for dname in dataset_names:
        for mname in model_names:
            dir = build_dir(dname, mname)
            if not os.path.exists(dir):
                if verbose > 2:
                    print(f"Skipping '{dir}' because it does not exist.")
                continue
            for file in os.listdir(dir):
                if ".npy" in file:
                    date = dt.datetime.strptime(file.split("_")[-1].split(".")[0][1:], date_format)
                    if (after is None or date >= after) and (before is None or date <= before):
                        os.remove(os.path.join(dir, file))
                        count += 1
                        if verbose > 0:
                            print("\tDeleted\t", file, end=" ")
                            if verbose > 1:
                                print("\tfrom", dir)
                            else:
                                print()
                    else:
                        retained += 1
    print("Data deletion complete")
    if verbose > 0:
        print(f"Deleted {count} files from {len(dataset_names)} datasets and {len(model_names)} models")
        print(f"Retained {retained} files")
        
            
def read_idx(filepath):
    """
    Reads IDX format files (for MNIST and Fashion-MNIST datasets).
    """
    with gzip.open(filepath, 'rb') if filepath.endswith(".gz") else open(filepath, 'rb') as f:
        # Read the magic number and dimensions
        magic_number = int.from_bytes(f.read(4), byteorder='big')
        num_items = int.from_bytes(f.read(4), byteorder='big')
        if magic_number == 2049:  # Labels
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data
        elif magic_number == 2051:  # Images
            rows = int.from_bytes(f.read(4), byteorder='big')
            cols = int.from_bytes(f.read(4), byteorder='big')
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_items, rows * cols)
            return data
        else:
            raise ValueError(f"Unsupported magic number: {magic_number}")        




### MODEL FUNCTIONS ###
def update_architecture(data:list, input_size:int) -> dict:
    """
    Recursively update the values of keys named 'width', 'in_features', or 'out_features'
    by multiplying them by input_size.
    Parameters:
        data (list): the model architecture as a list of layers
        input_size (int): the number of input features in the data
    Returns:
        data (dict): the updated model architecture as a dictionary
    """
    # handle control case
    if len(data) == 1 and isinstance(data[0], int):
        return [input_size]

    # recursively update widths and feature dimensions
    for layer in data:
        for param, value in layer["params"].items():
            if param in {'width', 'in_features', 'out_features'}:
                if isinstance(value, (int, float)):
                    layer["params"][param] = int(np.round(value * input_size))
        if layer["type"] == "Fold":
            layer["params"]["leak"] = 0.
    return data


def get_model(model_name:str, input_size:int=0, output_size:int=0, lmnn_default_neighbors:int=3, 
              architecture_path_local:str=None, no_cut:bool=False, no_relu:bool=False) -> object:
    """
    Returns a new instance of the model based on the model name.
    Can be "randomforest", "knn", or "metric".
    Parameters:
        model_name (str): The name of the model to train.
        input_size (int): The dimension of the input data.
        output_size (int): The dimension of the output data.
        lmnn_default_neighbors (int): The number of neighbors to use for the metric learning model
        architecture_path_local (str): an alternative path to an architecture json to use, default: BenchmarkTests/architectures.json
        no_cut (bool): if true, then the final linear layer is not included and only the layers specified in the architecture dictionary
    Returns:
        model: The model to train
    """
    if architecture_path_local is None : 
        with open(architecture_path) as f:
            architectures = json.load(f)
    else : 
        with open(architecture_path_local) as f: 
            architectures = json.load(f)
    if model_name in architectures.keys():
        meta_data = architectures.get(model_name, {})
        arch = update_architecture(meta_data.get("structure", [1]), input_size=input_size)
        lr = meta_data.get("learning_rate", 1e-3)
        mdl = DynamicOrigami(architecture=arch, num_classes=output_size, no_cut=no_cut, no_relu=no_relu)
        return mdl, lr
    else:
        mdl = RandomForestClassifier(n_jobs=-1) if model_name == "randomforest" else \
            KNeighborsClassifier(n_jobs=-1) if model_name == "knn" else \
            LMNN(n_neighbors=lmnn_default_neighbors) if model_name == "metric" else \
            OrigamiFold4(input_size) if model_name == "dl_fold" else \
            OrigamiSoft4(input_size) if model_name == "dl_softfold" else \
            CNNModel(input_channels= input_size) if model_name == "dl_cnn" else None
            
    if mdl is None:
        raise InvalidModelError(f"Invalid model name. Must be 'randomforest', 'knn', 'dl_cnn', 'dl_fold', 'dl_softfold' or 'metric' not '{model_name}'.")
    return mdl




def run_lmnn(x_train:np.ndarray, y_train:np.ndarray, x_test:np.ndarray, lmnn_default_neighbors:int=3):
    """
    Train the LMNN model and predict on the test set
    Parameters:
        x_train (np.ndarray): training data
        y_train (np.ndarray): training labels
        x_test (np.ndarray): testing data
        lmnn_default_neighbors (int): number of neighbors to use for the metric learning model
    Returns:
        np.ndarray: predictions on the test set
        np.ndarray: predictions on the training set
        float: average training time
    """
    # train and fit metric learning model
    lmnn = LMNN(n_neighbors=lmnn_default_neighbors)
    start_time = time.perf_counter()
    lmnn.fit(x_train, y_train)
    X_train_lmnn = lmnn.transform(x_train)
    X_test_lmnn = lmnn.transform(x_test)

    # train and fit knn predictor model
    knn = KNeighborsClassifier(n_neighbors=lmnn_default_neighbors, n_jobs=-1)
    knn.fit(X_train_lmnn, y_train)
    end_time = time.perf_counter()

    y_pred = knn.predict(X_test_lmnn)
    y_pred_train = knn.predict(X_train_lmnn)

    return y_pred, y_pred_train, end_time - start_time



def run_standard(model, x_train:np.ndarray, y_train:np.ndarray, x_test:np.ndarray):
    """
    Train a standard model and predict on the test set

    Parameters:
        model: model to train
        x_train (np.ndarray): training data
        y_train (np.ndarray): training labels
        x_test (np.ndarray): testing data
    
    Returns:
        np.ndarray: predictions on the test set
        np.ndarray: predictions on the training set
        float: average training time
    """
    start_time = time.perf_counter()
    model.fit(x_train, y_train)
    end_time = time.perf_counter()

    y_pred = model.predict(x_test)
    y_pred_train = model.predict(x_train)

    return y_pred, y_pred_train, end_time - start_time


def run_deep_learning(model, x_train:np.ndarray, y_train:np.ndarray, 
                      x_test:np.ndarray, y_test:np.ndarray, 
                      learning_rate:float=1e-3, n_epochs:int=200,
                      return_training:bool=True, verbose:int=0):
    """
    Train a deep learning model and predict on the test set
    Parameters:
        model: model to train
        x_train (np.ndarray): training data
        y_train (np.ndarray): training labels
        x_test (np.ndarray): testing data
        y_test (np.ndarray): testing labels
        n_epochs (int): number of epochs to train the model
        learning_rate (float): the learning rate for the model
        return_training (bool): whether to return the meta data from train time
        verbose (int): The verbosity level
    Returns:
        train_losses (list): training losses for each epoch
        val_losses (list): validation losses for each validation epoch
        train_accuracies (list): training accuracies for each epoch
        val_accuracies (list): validation accuracies for each validation epoch
        train_time (float): average training time
        inference_speed (float): average inference speed
        num_parameters (int): number of parameters in the model
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = load_data(x_train, y_train)
    val_loader = load_data(x_test, y_test)
    start_time = time.perf_counter()
    train_losses, val_losses, train_accuracies, val_accuracies, *learning_rates = train(model, optimizer, train_loader, val_loader, 
                                                                       validate_rate=0.05, epochs=n_epochs, verbose=verbose)
    end_time = time.perf_counter()
    
    # calculate inference speed
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_inference = time.perf_counter()
    with torch.no_grad():
        y_preds = [model(x.to(DEVICE)).argmax(dim=1) for x, y in val_loader]
    end_inference = time.perf_counter()
    inference_speed = (end_inference - start_inference) / len(y_test)
    
    # get parameter count
    num_parameters = sum(p.numel() for p in model.parameters())
    
    if not return_training:
        model.eval()
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            y_pred_trains = [model(x.to(DEVICE)).argmax(dim=1) for x, y in train_loader]
            y_preds = [model(x.to(DEVICE)).argmax(dim=1) for x, y in val_loader]
        y_pred_trains = torch.cat(y_pred_trains).cpu().numpy()
        y_preds = torch.cat(y_preds).cpu().numpy()  
        return y_preds, y_pred_trains, end_time - start_time, inference_speed, num_parameters
    
    return train_losses, val_losses, train_accuracies, val_accuracies, \
            end_time - start_time, inference_speed, num_parameters


def benchmark_ml(model_name:str, experiment_info, datetime, repeat:int=5, 
                 save_all:bool=True, save_any:bool=True, refresh:bool=True, verbose:int=0):
    """
    Trains a model on the cancer dataset with different data sizes and saves the accuracy and time data.
    Parameters:
        model_name (str): The name of the model to train. Can be "randomforest", "knn", or "metric".
        experiment_info (tuple): Contains
            dataset_name (str): The name of the dataset to train on.
            data_sizes (list(int)): A list of the sizes of the dataset to train on.
            X_train (np.ndarray): The training data.
            y_train (np.ndarray): The training labels.
            X_test (np.ndarray): The testing data.
            y_test (np.ndarray): The testing labels.
        datetime (str): The current date and time.
        repeat (int): The number of times to repeat the experiment.
        save_all (bool): Whether to save all the data or just the means and stds
        save_any (bool): Whether to save any data at all
        refresh (bool): Whether to refresh the last file
        verbose (int): The verbosity level
    Returns:
        results_dict (dict): A dictionary containing the accuracy and time data for each model and iteration
    """
    val_length = 20
    lmnndn = 3
    if not save_any:
        save_all = False
    # unpack experiment info
    dataset_name, data_sizes, X_train, y_train, X_test, y_test = experiment_info
    results_dict = {model_name: {}}
    save_constants = (dataset_name, model_name, datetime)
    n_epochs = json.load(open(config_path)).get("num_epochs")
    info_list = json.load(open(config_path)).get("info_list")
    info_titles = json.load(open(config_path)).get("info_titles")

    
    for i in range(repeat):
        X_train, y_train = shuffle(X_train, y_train, random_state=i)
        # if len(np.unique(y_train)) < 2:
        #     raise ValueError("Not enough classes in the training data")

        # set up data sample
        size = data_sizes[-1] if len(data_sizes) > 0 else None
        if size is None or size > len(X_train):
            data_sizes[data_sizes.index(size)] = len(X_train)
            size = len(X_train)
        tmp_X_train, tmp_y_train = X_train[:size], y_train[:size]
        test_size = min(size, len(X_test)//2)
        tmp_X_test, tmp_y_test = X_test[:test_size], y_test[:test_size]
        outs = len(np.unique(y_train))
        
        # train the model and get performance results
        if "Fold" in model_name or "Control" == model_name:
            model, lr = get_model(model_name, input_size=X_train.shape[1], output_size=outs)
            train_losses, val_losses, train_accuracies, val_accuracies, train_time, inference_speed, num_parameters = \
                run_deep_learning(model, tmp_X_train, tmp_y_train, tmp_X_test, tmp_y_test, 
                                  n_epochs=n_epochs, learning_rate=lr, verbose=verbose)
        else:
            model = get_model(model_name, lmnn_default_neighbors=lmnndn)
            func = run_standard if model_name != "metric" else run_lmnn
            y_pred, y_pred_train, train_time = func(model, tmp_X_train, tmp_y_train, tmp_X_test, verbose=verbose)
            # evaluating accuracy
            acc_train = accuracy_score(tmp_y_train, y_pred_train)
            acc_test = accuracy_score(tmp_y_test, y_pred)
            train_accuracies = [acc_train]*val_length
            val_accuracies = [acc_test]*val_length
            train_losses = [0]*val_length
            val_losses = [0]*val_length
            inference_speed = 0
            num_parameters = 0


        # Done evaluating, now saving data
        metric_list = [np.array(train_losses), 
                       np.array(val_losses), 
                       np.array(train_accuracies), 
                       np.array(val_accuracies), 
                       np.array(train_time), 
                       np.array(inference_speed), 
                       np.array(num_parameters)]
        if save_all:
            # check if the lengths match
            if len(metric_list) != len(info_list) or len(metric_list) != len(info_titles):
                print("Error: metric list length does not match info list length")
                print(f"metric_list: {len(metric_list)}. info_list: {len(info_list)}. info_titles: {len(info_titles)}")
                if os.isatty(0):
                    pdb.set_trace() 
            # save each metric for each iteration
            for j, data, info_type in zip(range(len(metric_list)), metric_list, info_titles):
                save_data(data, save_constants, info_type, i, val=j==1, refresh=refresh, repeat=repeat)
        results_dict[model_name][i] = {name: value for name, value in zip(info_list, metric_list)}


    # Done benchmarking, calculate means and stds and saving them
    results_mean = {}
    results_std = {}
    for info in info_list:
        # compute means and stds of the data for each metric
        values = np.array([results_dict[model_name][i][info] for i in range(repeat)])
        results_mean[info] = np.mean(values, axis=0)
        results_std[info] = np.std(values, axis=0)

    # save means and stds
    if not save_any:
        data_list, val_list, type_list, name_list = [], [], [], []
        for info in info_list:
            # datalist contains the mean and std for each metric
            data_list.extend([results_mean[info], results_std[info]])
            is_validation = "val" in info
            # val_list contains whether the data is validation data
            val_list.extend([is_validation, is_validation])
            metric_type = "acc" if "acc" in info else "val" if "val" in info else "time"
            # type_list contains the type of metric (accuracy, validation, or time)
            type_list.extend([metric_type] * 2)
            # name_list contains the name of the metric (mean or std)
            name_list.extend(["mean", "std"])

        # save the data
        for data, info_type, name, val in zip(data_list, type_list, name_list, val_list):
            save_data(data, save_constants, info_type, name, val=val, refresh=refresh, repeat=repeat)

    # update the results dictionary with the means and stds
    results_dict[model_name]["mean"] = {info: results_mean[info] for info in info_list}
    results_dict[model_name]["std"] = {info: results_std[info] for info in info_list}
    return results_dict





### MODEL EVALUATION ###

def rebuild_results(train_benchmarking:dict, val_benchmarking:dict, dataset_name:str, 
                    train_metrics:list=["loss", "acc", "time", "params"], 
                    val_metrics:list=["loss"],
                    repeat:int=5, verbose:int=0) -> dict:
    """
    Rebuild the benchmarking results from the numpy files
    Parameters:
        benchmarking (dict): dictionary to store the results
        data_set (str): name of the dataset
        all_data (bool): whether to load all the data or just the means and std
        repeat (int): number of times to repeat the experiment
        sup_date (bool): Whether the data is downloaded from the supercomputer
        verbose (int): the verbosity level
    Returns:
        benchmarking (dict): dictionary containing the benchmarking results
    """
    possible_architectures = list(json.load(open(architecture_path)).keys())
    for model_name in possible_architectures:
        # skip checking
        if model_name in train_benchmarking.keys():
            if verbose > 1:
                print("Skipping", model_name, "because it is already loaded")
            continue
        folder = build_dir(dataset_name, model_name)
        exists = os.path.exists(folder)
        if not exists or (exists and len(os.listdir(folder)) == 0):
            if verbose > 1:
                print(f"Skipping '{model_name}' because the folder is empty")
            continue
        

        if verbose > 0:
            print(f"Loading '{model_name}' data from '{folder}'")

        train_benchmarking[model_name] = {}
        val_benchmarking[model_name] = {}

        # Loop over all metrics we want to load
        mean_metric_dict = {}
        std_metric_dict = {}
        for metric in train_metrics:
            metric_list = []
            for i in range(repeat):
                train_data, do_std = load_result_data(dataset_name, model_name, metric, i, val=False, verbose=verbose)
                metric_list.append(train_data)
            
            # Get the composite array
            composite_array = np.array(metric_list)
            mean = np.mean(composite_array, axis=0)
            if do_std:
                std = np.std(composite_array, axis=0)
            else:
                std = np.zeros_like(mean)
            
            # store them in the temporary dictionary
            mean_metric_dict[metric] = mean
            std_metric_dict[metric] = std
            
        # store the temporary dictionary in the benchmarking dictionary
        train_benchmarking[model_name]['mean'] = mean_metric_dict
        train_benchmarking[model_name]['std'] = std_metric_dict
        
        # Do the validation metrics
        mean_metric_dict = {}
        std_metric_dict = {}
        for metric in val_metrics:
            metric_list = []
            for i in range(repeat):
                val_data, do_std = load_result_data(dataset_name, model_name, metric, i, val=True, verbose=verbose)
                metric_list.append(val_data)
            
            # Get the composite array
            composite_array = np.array(metric_list)
            mean = np.mean(composite_array, axis=0)
            if do_std:
                std = np.std(composite_array, axis=0)
            else:
                std = np.zeros_like(mean)
            
            # store them in the temporary dictionary
            mean_metric_dict[metric] = mean
            std_metric_dict[metric] = std
            
        # add additional key and values to the benchmarking dictionary
        val_benchmarking[model_name]['mean'] = mean_metric_dict
        val_benchmarking[model_name]['std'] = std_metric_dict
    
    # print(train_benchmarking)
    return train_benchmarking,val_benchmarking


def plotly_results(benchmarking:dict, constants:tuple, scale:int=5, repeat:int=5,
                 save_fig:bool=True, replace_fig:bool=False, from_data:bool=True, 
                 errors:str='raise', rows:int=1, verbose:int=0) -> None:
    """
    Plot the benchmarking results, Duplicate of plot_results but with plotly
    Parameters:
        benchmarking (dict): dictionary containing the benchmarking results
        constants (tuple): contains
            data_sizes (list(int)): list of data sizes
            datetime (str): current date and time
            dataset_name (str): name of the dataset
        scale (int): scale of the figure
        repeat (int): number of times to repeat the experiment
        save_fig (bool): whether to save the figure
        replace_fig (bool): whether to replace the old figure
        from_data (bool): whether to load the data from the numpy files
        errors (str): how to handle errors
        rows (str): how to arrange the subplots
        verbose (int): the verbosity level
    """
    assert errors in ["raise", "ignore", "flag"], f"'errors' must be 'raise', 'ignore', or 'flag' not '{errors}'"
    assert len(constants) == 3, f"constants must have 3 elements not {len(constants)}"
    assert type(scale) == int, f"'scale' must be an integer not {type(scale)}"
    assert scale > 0 and scale < 20, f"'scale' must be between 0 and 20 not {scale}"
    assert type(repeat) == int, f"'repeat' must be an integer not {type(repeat)}"
    assert repeat > 0, f"'repeat' must be greater than 0 not {repeat}"
    assert type(save_fig) == bool, f"'save_fig' must be a boolean not {type(save_fig)}"
    assert type(replace_fig) == bool, f"'replace_fig' must be a boolean not {type(replace_fig)}"
    assert type(from_data) == bool, f"'from_data' must be a boolean not {type(from_data)}"
    assert type(rows) == int, f"'cols' must be an integer not {type(rows)}"
    assert rows > 0, f"'cols' must be greater than 0 not {rows}"
    
    data_sizes, datetime, dataset_name = constants
    info_ylabels = ["Loss", "Loss", "Accuracy (%)", "Accuracy (%)", "Training Time (s)"] #, "Prediction Time (s)"
    colors = [
        (27,158,119,0.2),  # '#1b9e77'
        (217,95,2,0.2),    # '#d95f02'
        (117,112,179,0.2), # '#7570b3'
        (231,41,138,0.2),  # '#e7298a'
        (102,166,30,0.2)   # '#66a61e'
    ]
    n_epochs = json.load(open(config_path)).get("num_epochs")
    info_list = json.load(open(config_path)).get("info_list")
    info_length = len(info_list)
    pnt_density = 20
    cols = math.ceil(info_length / rows)
    first=True

    if from_data:
        benchmarking = rebuild_results(benchmarking, dataset_name, repeat=repeat, verbose=verbose)
    
    loop = tqdm(total=len(benchmarking.items())*info_length, position=0, leave=True, disable=verbose<0)
    subs = [f"{' '.join([word.capitalize() for word in info_type.split('_')])}" for info_type in info_list]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subs)
    for j, (model_name, model_results) in enumerate(benchmarking.items()):
        for i, (info_type, means), (_, stds) in zip(range(info_length), model_results["mean"].items(), model_results["std"].items()):
            loop.update()
            try:
                start = 0
                if type(means) == np.float64:
                    means = np.array([means])
                    stds = np.array([model_results["std"][info_type]])
                    start = j
                if len(means) > pnt_density:
                    # downsample the 1d means data where pnt_density is the number of points desired
                    sep = max(int(len(means) / pnt_density), 1)
                    means = means[::sep]
                    stds = stds[::sep]
                domain = np.linspace(start, n_epochs, len(means))
            except TypeError as e:
                if errors == "raise":
                    raise e
                if errors == "ignore":
                    continue
                if errors == "flag":
                    print("Error with", model_name, info_type, "-", e)
                    continue
            row = (i // cols) + 1  # Calculate row number
            col = (i % cols) + 1   # Calculate column number
            fig.add_trace(
                go.Scatter(x=domain, y=means, mode='lines+markers', name=model_name,
                        line=dict(color=f"rgb{colors[j][:-1]}"), showlegend=first),
                row=row, col=col
            )
            fig.add_trace(
                go.Scatter(x=domain, y=means + stds, mode='lines', line=dict(width=0),
                        fill='tonexty', fillcolor=f"rgba{colors[j]}", showlegend=False),
                row=row, col=col
            )
            fig.add_trace(
                go.Scatter(x=domain, y=means - stds, mode='lines', line=dict(width=0),
                        fill='tonexty', fillcolor=f"rgba{colors[j]}", showlegend=False),
                row=row, col=col
            )
            fig.update_xaxes(title_text="Epoch", row=row, col=col)
            fig.update_yaxes(title_text=info_ylabels[i], row=row, col=col)
            if "acc" in info_type:
                fig.add_hline(y=1, line=dict(color='black', dash='dash'), row=row, col=col)
            if "time" in info_type:
                fig.update_yaxes(type="log", row=row, col=col)
            first=False

    # Update layout for the overall title
    fig.update_layout(
        title=f"Model Benchmarking on '{dataset_name}'",
        showlegend=True,
        height=300*rows,  # You can adjust the height
        width=1200,  # You can adjust the width
    )
    loop.close()

    # save and show the figure
    if save_fig:
        if verbose > 0:
            print("Saving your figure...")
        fig_path = f"results/{dataset_name}/charts/"
        fig_name = f"benchmarking_{datetime}.png"
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        if replace_fig:
            replace_path = get_last_file(fig_path, "benchmarking", insert="_", file_type="png")
            if replace_path is not None:
                os.remove(replace_path)
        fig.write_image(os.path.join(fig_path, fig_name))
    fig.show()
    
    
    
def plot_results(dataset_name:str, scale:int=5, repeat:int=5,
                 save_fig:bool=True, replace_fig:bool=False, 
                 models_per_graph=None, scale_percentile=None, fontsize=15,
                 train_metrics:list=["loss", "acc", "time", "params"],
                 val_metrics:list=["loss"], 
                 col_names:list= ['Validation Loss','Train Loss', 'Train Accuracy', 'Train Time', 'Num Parameters'],
                 errors:str='raise',verbose:int=0) -> None:
    """
    Plot the benchmarking results
    Parameters:
        constants (tuple): contains
            datetime (str): current date and time
            dataset_name (str): name of the dataset
        scale (int): scale of the figure
        repeat (int): number of times to repeat the experiment
        save_fig (bool): whether to save the figure
        replace_fig (bool): whether to replace the old figure
        models_per_graph (int): number of models to plot on each row of graphs
        scale_percentile (float): the percentile of points we want to cut off in our y-axis
        fontsize (int): the fontsize of the text
        errors (str): how to handle errors
        verbose (int): the verbosity level
    """
    assert errors in ["raise", "ignore", "flag"], f"'errors' must be 'raise', 'ignore', or 'flag' not '{errors}'"
    assert type(dataset_name) == str, f"dataset name must be a string, not {type(dataset_name)}"
    assert type(scale) == int, f"'scale' must be an integer not {type(scale)}"
    assert scale > 0 and scale < 20, f"'scale' must be between 0 and 20 not {scale}"
    assert type(repeat) == int, f"'repeat' must be an integer not {type(repeat)}"
    assert repeat > 0, f"'repeat' must be greater than 0 not {repeat}"
    assert type(save_fig) == bool, f"'save_fig' must be a boolean not {type(save_fig)}"
    assert type(replace_fig) == bool, f"'replace_fig' must be a boolean not {type(replace_fig)}"
    
    # Get the constants
    date_format = "%y%m%d@%H%M"
    date_time = dt.datetime.now().strftime(date_format)
    n_epochs = json.load(open(config_path)).get("num_epochs")
    pnt_density = 20
    
    # Load the data
    train_benchmarking, val_benchmarking = rebuild_results({},{}, dataset_name, repeat=repeat, verbose=verbose, train_metrics=train_metrics, val_metrics=val_metrics)
    
    # Figure out how many rows we need
    train_items = list(train_benchmarking.items())
    val_items = list(val_benchmarking.items())
    num_models = len(train_items)
    num_train_plots = len(train_items[0][1]['mean'])
    num_val_plots = len(val_items[0][1]['mean'])
    if models_per_graph is not None:
        num_rows = int(math.ceil(num_models / models_per_graph))
    else:
        num_rows = 1
        models_per_graph = num_models
        
    # calculate the number of columns
    num_cols = num_train_plots + num_val_plots + 1
    if num_cols != len(col_names) + 1:
        print(f"Warning: Must specify the correct number of column names. Expected {num_cols-1} but got {len(col_names)}")
        col_names = ["Val " + name for name in val_metrics] + ["Train " + name for name in train_metrics]
    col_names = ["Model"] + col_names
    
    # Get the names of the models and the colors
    model_names = [model_name for model_name, _ in train_items]
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i / models_per_graph) for i in range(models_per_graph)]
    
    # Find the upper and lower percentiles for the scale
    if scale_percentile is None:
        lower_percentile, upper_percentile = .025, .975
    elif scale_percentile < 0 or scale_percentile > 1:
        raise ValueError("scale_percentile must be between 0 and 1")
    else:
        lower_percentile, upper_percentile = scale_percentile/2, 1 - scale_percentile/2
    scales = {i:[] for i in range(num_cols-1)}
    
    # Loop through the models and find the scales
    for j in range(num_models):
        model_name, train_results = train_items[j]
        _, val_results = val_items[j]
        train_means = train_results['mean']
        val_means = val_results["mean"]
        means = list(val_means.items()) + list(train_means.items())
        
        # Loop through the different types of graphs to plot
        for i, (info_type, means) in enumerate(means):
            if info_type in ["time", "params"]:
                scales[i].append(means)
            else:
                scales[i].append(np.percentile(means, 100*upper_percentile))
                scales[i].append(np.percentile(means, 100*lower_percentile))
                
    # Find the max and min scales
    border = .1
    scales = {i: [np.min(scales[i]) - border*np.abs(np.min(scales[i])), np.max(scales[i]) + border*np.abs(np.max(scales[i]))] for i in range(num_cols-1)}
    # Set up the figure to plot
    plt.figure(figsize=(scale*num_cols, scale*num_rows*.75), dpi=25*scale)
    current_row = 0
    model_name_dictionary = {}
    
    # Loop through the models and get the items from the model
    loop = tqdm(total=num_models*(num_cols-1), position=0, leave=True, disable=verbose<0)
    for j in range(num_models):
        model_name, train_results = train_items[j]
        model_name_dictionary[model_name] = str(j)
        val_results = val_items[j][1]
        train_means = train_results["mean"]
        train_stds = train_results["std"]
        val_means = val_results["mean"]
        val_stds = val_results["std"]
        color_index = j % len(colors)
        
        # Get the means and std list
        means = list(val_means.items()) + list(train_means.items())
        stds =  list(val_stds.items()) + list(train_stds.items())
        
        # Loop through the different graphs and get the right subplot
        for i, (info_type, means), (_, stds) in zip(range(num_cols-1), means, stds):
            plt.subplot(num_rows, num_cols, current_row*num_cols+i+2)
            if current_row == 0:
                plt.title(col_names[i+1], fontsize=fontsize)
            
            # If it is a loss or accuracy, plot it with the standard error bars
            if info_type in ["loss", "acc"]:
                try:
                    start = 0
                    if type(means) == np.float64:
                        means = np.array([means])
                        stds = np.array([stds])
                        start = j
                    if len(means) > pnt_density:
                        # downsample the 1d means data where pnt_density is the number of points desired
                        sep = max(int(len(means) / pnt_density), 1)
                        means = means[::sep]
                        stds = stds[::sep]
                    domain = np.linspace(start, n_epochs, len(means))
                except TypeError as e:
                    if errors == "raise":
                        raise e
                    if errors == "ignore":
                        continue
                    if errors == "flag":
                        print("Error with", model_name, info_type, "-", e)
                        continue
                    
                # Get the standard errors and plot the function 
                standard_errors = stds / np.sqrt(repeat)    
                plt.plot(domain, means, label=model_name, marker='o', color=colors[color_index])
                plt.fill_between(domain, means - 2*standard_errors, means + 2*standard_errors, alpha=0.2, color=colors[color_index])
                plt.ylim(scales[i])
                # plt.xlabel("Epoch")
                # plt.ylabel(info_type)
                
                # If it is accuracy, plot a dotted line at 1
                if info_type == "acc": # This is reversed as of rn with 'acc'
                    plt.axhline(y=1, color='k', linestyle='--')
                    plt.ylim(scales[i])
                    plt.ylim([0, 1.1])
            
            # If it is time, do a log bar plot of the time
            elif info_type == "time":
                plt.bar(model_name_dictionary[model_name], means, yerr=stds, color=colors[color_index], label=model_name)
                # plt.ylabel(info_type)
                # ylimit = scales[i]
                # ylimit[0] = 0
                plt.ylim(scales[i])
                plt.yscale("log")
            
            # If it is params, do a bar plot of the number of parameters
            elif info_type == "params":
                plt.bar(model_name_dictionary[model_name], means, yerr=stds, color=colors[color_index], label=model_name)
                # plt.ylabel(info_type)
                plt.ylim(scales[i])
                
        # If we have collected enough models, record their names in the correct subplot
        if (j+1) % models_per_graph == 0:
            plt.subplot(num_rows, num_cols, current_row*num_cols+1)
            if current_row == 0:
                plt.title("Model Name", fontsize=fontsize)
            current_row += 1
            
            # Get the names of the models and plot empty lines
            name_list = model_names[j+1-models_per_graph:j+1]
            for k,name in enumerate(name_list):
                plt.plot([], [], label=f"{model_name_dictionary[name]}: "+name, color=colors[k], lw=fontsize/2)  # Thicker lines in the legend
            
            # Make the legend and turn off the axes
            plt.legend(loc="center", fontsize=fontsize)
            plt.axis("off")  # Turn off the axes for the legend area
            
        # update the loop
        loop.update()
    loop.close()
    
    # Add our titles and add a little bit of padding
    plt.suptitle(f"Model Benchmarking on '{dataset_name}'", fontsize=fontsize*1.5)

    # save and show the figure
    if save_fig:
        if verbose > 0:
            print("Saving your figure...")
        fig_path = f"results/{dataset_name}/charts/"
        fig_name = f"benchmarking_{date_time}.png"
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        if replace_fig:
            replace_path = get_last_file(fig_path, "benchmarking", insert="_", file_type="png")
            if replace_path is not None:
                os.remove(replace_path)
        plt.savefig(os.path.join(fig_path, fig_name))
    plt.show()
    
    
    
    
    

def compute_plot_scales(scale_percentile:float, num_cols:int, num_models:int, 
                        train_items:list, val_items:list) -> dict:
    """
    Compute the x and y scales for the plots
    Helpful for plotting the data in plotly so that all of the plots are on the same scale
    Parameters:
        scale_percentile (float): the percentile of points we want to cut off in our y-axis
        num_cols (int): the number of columns in the plot
        num_models (int): the number of models
        train_items (list): the training items
        val_items (list): the validation items
    Returns:
        scales (dict): a dictionary containing the scales for the plots
    """
    # modify unset inputs
    lower_percentile = 0.025 if scale_percentile is None else scale_percentile/2
    upper_percentile = 0.975 if scale_percentile is None else 1 - scale_percentile/2
    
    border = .1
    scales = {i:[] for i in range(num_cols-1)}
    for j in range(num_models):
        model_name, train_results = train_items[j]
        _, val_results = val_items[j]
        train_means = train_results['mean']
        val_means = val_results["mean"]
        means = list(val_means.items()) + list(train_means.items())
        
        # Loop through the different types of graphs to plot
        for i, (info_type, means) in enumerate(means):
            if info_type in ["time", "params"]:
                scales[i].append(means)
            else:
                scales[i].append(np.percentile(means, 100*upper_percentile))
                scales[i].append(np.percentile(means, 100*lower_percentile))
    scales = {i: [np.min(scales[i]) - border*np.abs(np.min(scales[i])), 
                  np.max(scales[i]) + border*np.abs(np.max(scales[i]))] 
              for i in range(num_cols-1)}
    return scales


def get_colors(alpha:float=0.2) -> list:
    """
    This function creates a list of colors with an alpha value
    Helpful for making the plotly data look nice
    Parameters:
        alpha (float): the alpha value for the colors
    Returns:
        colors (list): a list of colors with an alpha value
    """
    # check input
    assert isinstance(alpha, float), f"alpha must be a float not {type(alpha)}"
    assert 0 < alpha <= 1, f"alpha must be between 0 and 1 not {alpha}"

    colors = plt.get_cmap('tab20').colors
    max_val = 255
    return [(int(r*max_val), int(g*max_val), int(b*max_val)) for r, g, b in colors]


def count_rows(num_models:int, models_per_graph:int=None) -> tuple:
    """
    This function counts the number of rows and returns the number of rows and the number of models per graph
    Helpful for determining the number of rows and models per graph in the plotly graph of benchmarking results
    Parameters:
        num_models (int): the number of models
        models_per_graph (int): the number of models per graph
    Returns:
        num_rows (int): the number of rows
        models_per_graph (int): the number of models per graph
    """
    if models_per_graph is not None:
        num_rows = int(math.ceil(num_models / models_per_graph))
    else:
        num_rows = 1
        models_per_graph = num_models
    return num_rows, models_per_graph


def count_columns(train_metrics:list, val_metrics:list, col_names:list, 
                  num_train_plots:int, num_val_plots:int) -> tuple:
    """
    This function counts the number of columns and returns the column names
    Helpful for determining the number of columns and the column names in the plotly graph of benchmarking results
    Parameters:
        train_metrics (list): the metrics to plot for the training data
        val_metrics (list): the metrics to plot for the validation data
        col_names (list): the names of the columns
        num_train_plots (int): the number of training plots
        num_val_plots (int): the number of validation plots
    Returns:
        num_cols (int): the number of columns
        col_names (list): the names of the columns
    """
    # check inputs
    assert isinstance(train_metrics, list), f"train_metrics must be a list not {type(train_metrics)}"
    assert isinstance(val_metrics, list), f"val_metrics must be a list not {type(val_metrics)}"
    assert isinstance(col_names, list), f"col_names must be a list not {type(col_names)}"
    assert isinstance(num_train_plots, int), f"num_train_plots must be an integer not {type(num_train_plots)}"
    assert isinstance(num_val_plots, int), f"num_val_plots must be an integer not {type(num_val_plots)}"
    assert num_train_plots > 0, f"num_train_plots must be greater than 0 not {num_train_plots}"
    assert num_val_plots > 0, f"num_val_plots must be greater than 0 not {num_val_plots}"
    
    num_cols = num_train_plots + num_val_plots + 1
    if num_cols != len(col_names) + 1:
        print(f"Warning: Must specify the correct number of column names. Expected {num_cols-1} but got {len(col_names)}")
        col_names = ["Val " + name for name in val_metrics] + ["Train " + name for name in train_metrics]
    col_names = ["Model"] + col_names
    return num_cols, col_names


def process_line(means:np.array, stds:np.array, j:int, repeat:int, 
                 model_name:str, info_type:str, pnt_density:int, n_epochs:int,
                 errors:str='raise') -> tuple:
    """
    This function processes the line for the plotly graph
    Helpful for processing the line to plot in the plotly graph of benchmarking results
    Parameters:
        means (np.array): the means of the data
        stds (np.array): the standard deviations of the data
        j (int): the index of the model
        repeat (int): the number of times to repeat the experiment
        model_name (str): the name of the model
        info_type (str): the type of information
        pnt_density (int): the density of the points
        n_epochs (int): the number of epochs
        errors (str): how to handle errors
    Returns:
        means (np.array): the means of the data
        stds (np.array): the standard deviations of the data
        domain (np.array): the domain of the data
        cont (bool): whether to continue
    """
    cont = False
    try:
        start = 0
        if type(means) == np.float64:
            means = np.array([means])
            stds = np.array([stds])
            start = j
        if len(means) > pnt_density:
            # downsample the 1d means data where pnt_density is the number of points desired
            sep = max(int(len(means) / pnt_density), 1)
            means = means[::sep]
            stds = stds[::sep]
        domain = np.linspace(start, n_epochs, len(means))
    except TypeError as e:
        if errors == "raise":
            raise e
        if errors == "ignore":
            cont = True
        if errors == "flag":
            print("Error with", model_name, info_type, "-", e)
            cont = True
    standard_errors = stds / np.sqrt(repeat)
    return means, standard_errors, domain, cont


def add_model_results(fig:go.Figure, model_name:str, means:np.array, standard_errors:np.array,
                        domain:np.array, colors:np.array, color_idx:int, current_row:int, i:int, 
                        include_legend:bool=True, alpha:float=0.2) -> go.Figure:
    """
    This function adds the model results with standard deviations to the plotly graph
    Helpful for adding the model results to the plotly graph of benchmarking results
    Parameters:
        fig (go.Figure): the plotly figure
        model_name (str): the name of the model
        means (np.array): the means of the data
        standard_errors (np.array): the standard errors of the data
        domain (np.array): the domain of the data
        colors (np.array): the colors of the data
        color_idx (int): the index of the color
        current_row (int): the current row
        i (int): the index
        include_legend (bool): whether to include the legend
    Returns:
        fig (go.Figure): the plotly figure
    """
    sl = current_row==0 and include_legend
    fig.add_trace(
        go.Scatter(x=domain, y=means, mode='lines+markers', name=model_name,
                   line=dict(color=f"rgb{colors[color_idx]}"), marker=dict(size=2), showlegend=sl),
        row=current_row+1, col=i+1
    )
    # fill the area between the standard errors
    rgba = colors[color_idx] + (alpha,)
    fig.add_trace(
        go.Scatter(x=domain, y=means + 2*standard_errors, mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor=f"rgba{rgba}", showlegend=False),
        row=current_row+1, col=i+1
    )
    fig.add_trace(
        go.Scatter(x=domain, y=means - 2*standard_errors, mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor=f"rgba{rgba}", showlegend=False),
        row=current_row+1, col=i+1
    )
    return fig
    
    
def plot_results_plotly(dataset_name:str, scale:int=5, repeat:int=5,
                         save_fig:bool=True, replace_fig:bool=False,
                         models_per_graph=None, scale_percentile=None, fontsize=15, include_legend:bool=True,
                         train_metrics:list=["loss", "acc", "time", "params"],
                         val_metrics:list=["loss"],
                         col_names:list=['Validation Loss', 'Train Loss', 'Train Accuracy', 'Train Time', 'Num Parameters'],
                         errors:str='raise', verbose:int=0) -> None:
    """
    Plot the benchmarking results using Plotly.
    Parameters:
        dataset_name (str): name of the dataset
        scale (int): scale of the figure
        repeat (int): number of times to repeat the experiment
        save_fig (bool): whether to save the figure
        replace_fig (bool): whether to replace the old figure
        models_per_graph (int): number of models to plot on each row of graphs
        scale_percentile (float): the percentile of points we want to cut off in our y-axis
        fontsize (int): the fontsize of the text
        include_legend (bool): whether to include the legend
        train_metrics (list): the metrics to plot for the training data
        val_metrics (list): the metrics to plot for the validation data
        col_names (list): the names of the columns
        errors (str): how to handle errors
        verbose (int): the verbosity level
    """
    assert isinstance(dataset_name, str), f"dataset name must be a string, not {type(dataset_name)}"
    assert isinstance(scale, int), f"'scale' must be an integer not {type(scale)}"
    assert 0 < scale < 20, f"'scale' must be between 0 and 20 not {scale}"
    assert isinstance(repeat, int), f"'repeat' must be an integer not {type(repeat)}"
    assert repeat > 0, f"'repeat' must be greater than 0 not {repeat}"
    assert isinstance(save_fig, bool), f"'save_fig' must be a boolean not {type(save_fig)}"
    assert isinstance(replace_fig, bool), f"'replace_fig' must be a boolean not {type(replace_fig)}"
    assert isinstance(models_per_graph, (int, type(None))), f"'models_per_graph' must be an integer or None not {type(models_per_graph)}"
    assert isinstance(scale_percentile, (float, type(None))), f"'scale_percentile' must be a float or None not {type(scale_percentile)}"
    if scale_percentile is not None:
        assert 0 < scale_percentile < 1, f"'scale_percentile' must be between 0 and 1 not {scale_percentile}"
    assert errors in ["raise", "ignore", "flag"], f"'errors' must be 'raise', 'ignore', or 'flag' not '{errors}'"
    assert isinstance(verbose, int), f"'verbose' must be an integer not {type(verbose)}"

    print("Starting")

    # initialize the lists from the dictionaries of results
    train_benchmarking, val_benchmarking = rebuild_results({}, {}, dataset_name, repeat=repeat, verbose=verbose,
                                                           train_metrics=train_metrics, val_metrics=val_metrics)
    train_items = list(train_benchmarking.items())
    model_names = list(train_benchmarking.keys())
    val_items = list(val_benchmarking.items())
    
    # Get the constants
    date_format = "%y%m%d@%H%M"
    date_time = dt.datetime.now().strftime(date_format)
    n_epochs = json.load(open(config_path)).get("num_epochs")
    pnt_density = 20
    num_models = len(train_items)
    num_train_plots = len(train_items[0][1]['mean'])
    num_val_plots = len(val_items[0][1]['mean'])

    # Set up plotly variables
    num_rows, models_per_graph = count_rows(num_models, models_per_graph)
    num_cols, col_names = count_columns(train_metrics, val_metrics, col_names, num_train_plots, num_val_plots)
    colors = get_colors()
    


    # Create Plotly subplots
    fig = make_subplots(rows=num_rows, cols=num_cols, 
                           subplot_titles=col_names[1:], 
                           vertical_spacing=0.02, horizontal_spacing=0.03)

    # Loop through the models and get the items from the model
    current_row = 0
    model_name_dictionary = {}
    loop = tqdm(total=num_models*(num_cols-1), position=0, leave=True, disable=verbose<0)
    for j in range(num_models):
        model_name, train_results = train_items[j]
        model_name_dictionary[model_name] = str(j)
        val_results = val_items[j][1]
        train_means = train_results["mean"]
        train_stds = train_results["std"]
        val_means = val_results["mean"]
        val_stds = val_results["std"]
        color_idx = j % len(colors)

        # Get the means and std list
        means = list(val_means.items()) + list(train_means.items())
        stds =  list(val_stds.items()) + list(train_stds.items())
        
        # Loop through the different graphs and get the right subplot
        for i, (info_type, means), (_, stds) in zip(range(num_cols-1), means, stds):
            if info_type in ["loss", "acc"]:
                # format data correctly for plotting
                means, standard_errors, domain, cont = process_line(means, stds, j, repeat, model_name, info_type, pnt_density, n_epochs, errors)
                if cont: continue
                
                # plot the data with the standard errors
                fig = add_model_results(fig, model_name, means, standard_errors, domain, 
                                        colors, color_idx, current_row, i, include_legend)
                
                # If it is accuracy, plot a dotted line at 1
                if info_type == "acc":
                    fig.add_hline(y=1, line=dict(color='black', dash='dash'), row=current_row+1, col=i+1)
                    fig.update_yaxes(range=[0, 1.1], row=current_row+1, col=i+1)
            
            # If it is TIME, do a log bar plot of the time
            # If it is PARAMS, do a bar plot of the number of parameters
            elif info_type in ["time", "params"]:
                fig.add_trace(go.Bar(
                    x=[model_name_dictionary[model_name]],
                    y=[means],
                    error_y=dict(type='data', array=np.array([stds]) if np.isscalar(stds) else np.array(stds)),
                    marker=dict(color=f"rgb{colors[color_idx]}"),
                    name=model_name, showlegend=include_legend, 
                ), row=current_row+1, col=i+1)
                if info_type == "time":
                    fig.update_yaxes(type="log", row=current_row+1, col=i+1)
            loop.update()
                
        # If we have collected enough models, record their names in the correct subplot
        if (j+1) % models_per_graph == 0:
            current_row += 1
            if include_legend and current_row == 0:
                fig.update_layout(legend=dict(title="Model Name", orientation="h", y=1.1, x=0.5, xanchor='center'))
    loop.close()
    
    # Add our titles and add a little bit of padding
    fig.update_layout(title=f"Model Benchmarking on '{dataset_name}'")
    fig.update_layout(height=200*num_rows, width=1200)
    
    if save_fig:
        if verbose > 0:
            print("Saving your figure...")
        fig_path = f"results/{dataset_name}/charts/"
        fig_name = f"benchmarking_{date_time}.html"
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        if replace_fig:
            old_files = [f for f in os.listdir(fig_path) if "benchmarking" in f and f.endswith(".html")]
            for f in old_files:
                os.remove(os.path.join(fig_path, f))
        fig.write_html(os.path.join(fig_path, fig_name))
    fig.show()
