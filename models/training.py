import torch                    # type: ignore
import torch.nn as nn           # type: ignore
import torch.optim as optim     # type: ignore
import torch.nn.functional as F # type: ignore
import numpy as np # type: ignore
from tqdm import tqdm # type: ignore
from torch.utils.data import TensorDataset, DataLoader # type: ignore
from matplotlib import pyplot as plt # type: ignore




################################# Data Prep #####################################   
     
def load_data(x_data, y_data, batch_size=32, shuffle=True) -> torch.utils.data.DataLoader:
    """
    This function loads the data into a DataLoader object.
    Parameters:
        x_data (np.ndarray) - The input data
        y_data (np.ndarray) - The labels
        batch_size (int) - The batch size
        shuffle (bool) - Whether to
    Returns:
        DataLoader - The DataLoader object
    """
    # Convert x and y data to PyTorch tensors if they aren’t already
    x_tensor = torch.tensor(x_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.long)

    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

################################# Learning Rate Scheduling #################################
class NoamScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, model_size, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.model_size = model_size
        self.optimizer = optimizer
        super(NoamScheduler, self).__init__(self.optimizer, last_epoch)
        
    def get_lr(self) -> list:
        """
        This function calculates the learning rate based on the current step number.
        Returns:
            list - The learning rate for each parameter group
        """
        step_num = self.last_epoch + 1
        lr = self.optimizer.defaults['lr'] * (self.model_size ** (-0.5)) * min(step_num ** (-0.5), step_num * self.warmup_steps ** (-1.5))
        return [lr for _ in self.base_lrs]
    
class LRSchedulerWrapper:
    def __init__(self, scheduler=None):
        self.scheduler = scheduler

    def step(self):
        if self.scheduler is not None:
            self.scheduler.step()



################################# Helper functions for training #################################

def check_accuracy(y_hat:torch.Tensor, y:torch.Tensor) -> float:
    """
    This function checks the accuracy of the model.
    Parameters:
        y_hat (torch.Tensor) - The predicted values
        y (torch.Tensor) - The true values
    Returns:
        accuracy (float) - The accuracy of the model
    """
    # Get the predicted class by finding the max logit (highest score) along the last dimension
    predictions = y_hat.argmax(dim=1)
    
    # Compare predictions to the true labels and calculate accuracy
    correct = (predictions == y).sum().item()
    accuracy = correct / y.size(0)
    return accuracy


def validate(net, val_dataloader:torch.utils.data.DataLoader, DEVICE:torch.device) -> tuple:
    """
    This function calculates the validation loss and accuracy.
    Parameters:
        net (nn.Module) - The neural network
        val_dataloader (DataLoader) - The DataLoader object for the validation set
        DEVICE (torch.device) - The device to run the calculations on
    Returns:
        avg_val_loss (float) - The average validation loss
        avg_val_accuracy (float) - The average validation accuracy
    """
    net.eval()
    val_loss = []
    val_accuracy = []

    with torch.no_grad():  # Disable gradient calculations
        for x, y in val_dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # Forward pass
            y_hat = net(x)
            loss = F.cross_entropy(y_hat, y.long())
            val_loss.append(loss.item())

            # Calculate accuracy
            accuracy = check_accuracy(y_hat, y)
            val_accuracy.append(accuracy)

    # Calculate mean loss and accuracy over the entire validation set
    avg_val_loss = np.mean(val_loss)
    avg_val_accuracy = np.mean(val_accuracy)
    return avg_val_loss, avg_val_accuracy




################################# Training function #################################
def train(model, optimizer:torch.optim.Optimizer, 
          train_dataloader:torch.utils.data.DataLoader, 
          val_dataloader:torch.utils.data.DataLoader, 
          epochs:int=100, DEVICE:torch.device=None, 
          validate_rate:float=0, verbose:int=0, lr_schedule:LRSchedulerWrapper=None):
    """
    This function trains the neural network.
    Parameters:
        net (nn.Module) - The neural network
        optimizer (torch.optim.Optimizer) - The optimizer
        train_dataloader (DataLoader) - The DataLoader object for the training set
        val_dataloader (DataLoader) - The DataLoader object for the validation set
        epochs (int) - The number of epochs to train the model (default: 100)
        DEVICE (torch.device) - The device to run the calculations on
        validate_rate (float) - The rate at which to validate the model (default: 0)
        verbose (int) - The verbosity level (default: 0)
    Returns:
        train_losses (list) - The training losses for each epoch
        val_losses (list) - The validation losses for each epoch
        train_accuracies (list) - The training accuracies for each epoch
        val_accuracies (list) - The validation accuracies for each epoch
    """
    # Get the device if it is not already defined
    DEVICE = DEVICE or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose > 2:
        print(f"Working Device: {DEVICE}")
    model.to(DEVICE)
    
    # Initialize the lists to keep track of the losses and accuracies
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    learning_rates = []
    number_batches = len(train_dataloader)
    val_separation = int(validate_rate * epochs)
    lr_scheduler = lr_schedule

    # Define a loop object to keep track of training and loop through the epochs
    loop = tqdm(desc="Training", total=epochs*number_batches, position=0, leave=True, disable=verbose<=1)

    for i in range(epochs):
        epoch_loss, correct_preds, total_preds = 0, 0, 0
        model.train()
        
        # Loop through the train_dataloader
        for batch_num, (x, y) in enumerate(train_dataloader, 1):
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # Get the prediction and calculate the loss
            y_hat = model(x)
            loss = F.cross_entropy(y_hat, y)

            # Backpropagate the loss and update the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Append accuracies and losses
            loss_value = loss.item()
            epoch_loss += loss_value
            correct_preds += (y_hat.argmax(dim=1) == y).sum().item()
            total_preds += y.size(0)
            if verbose > 1:
                loop.set_description('epoch:{}/{}, batch: {}/{}, loss:{:.4f}'.format(i+1, epochs, batch_num, number_batches, loss_value))
                loop.update()

        lr_scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        learning_rates.append(lr)
        
        # Get the average loss for the epoch
        train_losses.append(epoch_loss / number_batches)
        train_accuracies.append(correct_preds / total_preds)
        
        # Calculate the validation loss and accuracy
        if val_separation > 0 and (i+1)% val_separation == 0:
            val_loss, val_accuracy = validate(model, val_dataloader, DEVICE)

            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
    
    # Close the loop and return the losses and accuracies
    loop.close()
        
    if lr_scheduler is not None:
        return train_losses, val_losses, train_accuracies, val_accuracies, learning_rates
    return train_losses, val_losses, train_accuracies, val_accuracies




################################# Analysis and plotting #################################
def plot_model(train_losses, val_losses, train_accuracies, val_accuracies):
    # plot the losses and accuracies
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='train')
    plt.plot(val_accuracies, label='val')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()