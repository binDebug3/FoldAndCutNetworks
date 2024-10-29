import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import torch.nn.functional as F # type: ignore
import numpy as np # type: ignore
from tqdm import tqdm # type: ignore
from torch.utils.data import TensorDataset, DataLoader # type: ignore
from matplotlib import pyplot as plt # type: ignore




################################# Data Prep #####################################        
def load_data(x_data, y_data, batch_size=32, shuffle=True) -> torch.utils.data.DataLoader:
    # Convert x and y data to PyTorch tensors if they aren’t already
    x_tensor = torch.tensor(x_data, dtype=torch.float32)  # Specify float32 for model compatibility
    y_tensor = torch.tensor(y_data, dtype=torch.long)     # Specify long for classification labels

    # Create a TensorDataset
    dataset = TensorDataset(x_tensor, y_tensor)
    
    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# TODO: Fix the following function to robustly handle different types of label data
# def encode_y(y, DEVICE) -> torch.Tensor:
#         """
#         Encodes the labels into one-hot format.
#         Parameters:
#             y (np.ndarray) - The labels
#         """
#         # if type(y) == np.ndarray:
#         #     y = torch.tensor(y)
#         y = y.clone().detach().to(DEVICE, dtype=torch.long) if isinstance(y, torch.Tensor) \
#             else torch.tensor(y, dtype=torch.long).to(DEVICE)
#         classes = torch.unique(y)
#         num_classes = len(classes)
#         one_hot = F.one_hot(y, num_classes).float()
#         return one_hot




################################# Learning Rate Scheduling #################################
class NoamScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, model_size, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.model_size = model_size
        super(NoamScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step_num = self.last_epoch + 1
        lr = self.optimizer.defaults['lr'] * (self.model_size ** (-0.5)) * min(step_num ** (-0.5), step_num * self.warmup_steps ** (-1.5))
        return [lr for _ in self.base_lrs]




################################# Helper functions for training #################################
def check_accuracy(y_hat, y):
    # Get the predicted class by finding the max logit (highest score) along the last dimension
    predictions = y_hat.argmax(dim=1)
    
    # Compare predictions to the true labels and calculate accuracy
    correct = (predictions == y).sum().item()
    accuracy = correct / y.size(0)
    return accuracy


def validate(net, val_dataloader, DEVICE):
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
def train(net, optimizer, train_dataloader, val_dataloader, epochs = 100, DEVICE=None):
    # Get the device if it is not already defined
    if DEVICE is None:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Working Device: {DEVICE}")
    net.to(DEVICE)
    
    # Initialize the lists to keep track of the losses and accuracies
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    number_batches = len(train_dataloader)

    #define a loop object to keep track of training and loop through the epochs
    loop = tqdm(total=epochs*number_batches, position=0)
    for i in range(epochs):
        epoch_loss = []
        epoch_accuracy = []
        net.train()
        
        # Loop through the train_dataloader
        batch_num = 1
        for x, y in train_dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # Get the prediction and calculate the loss
            y_hat = net(x)
            loss = F.cross_entropy(y_hat, y.long())

            # Backpropagate the loss and update the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Append accuracies and losses
            loss_value = loss.item()
            epoch_loss.append(loss_value)
            epoch_accuracy.append(check_accuracy(y_hat.detach(), y))
            
            # Update the loop
            if batch_num<10:
                disp_batch_num = '0'+str(batch_num)
            else:
                disp_batch_num = str(batch_num)
            loop.set_description('epoch:{}/{}, batch: {}/{}, loss:{:.4f}'.format(i+1, epochs, disp_batch_num, number_batches, loss_value))
            loop.update()
            batch_num += 1
            
        # Get the average loss for the epoch
        train_losses.append(np.mean(epoch_loss))
        train_accuracies.append(np.mean(epoch_accuracy))
        
        # Calculate the validation loss and accuracy
        val_loss, val_accuracy = validate(net, val_dataloader, DEVICE)
        
        # Append the validation loss and accuracy
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
    
    # Close the loop and return the losses and accuracies
    loop.close()
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