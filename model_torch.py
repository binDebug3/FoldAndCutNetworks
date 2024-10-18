import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

# import accuracy_score from sklearn.metrics
from sklearn.metrics import accuracy_score

class OrigamiNetwork(torch.nn.Module):
    """Put in our final docstring here"""
    
    ################################### Initialization ##################################
    def __init__(self, layers:int=3, width:int=None, learning_rate:float=0.001, reg:float=10, sigmoid=False,
                 optimizer:str="grad", batch_size:int=32, epochs:int=100, leak:float=0, crease:float=1):
        super(OrigamiNetwork, self).__init__()
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.reg = reg
        self.layers = layers
        self.width = width
        self.leak = leak
        self.crease = crease
        self.sigmoid = sigmoid
        
        # if the model is a sigmoid model, set the leak to 0
        if self.sigmoid:
            self.leak = 0

        # Variables to store
        self.X = None
        self.y = None
        self.n = None
        self.d = None
        self.feature_names = None
        self.classes = None
        self.num_classes = None
        self.y_dict = None
        self.one_hot = None
        self.fold_vectors = None
        self.output_layer = None
        self.input_layer = None
        self.b = None
        
        # Check if the model has an expand matrix
        self.has_expand = self.width is not None

        # Validation variables
        self.validate = False
        self.X_val_set = None
        self.y_val_set = None
        self.class_index = []
        self.val_history = []
        self.train_history = []
        self.fold_history = []
        self.cut_history = []
        self.expand_history = []
        self.learning_rate_history = []    

        # Initialize the fold vectors, output layer, and biases
        self.fold_vectors = [torch.nn.Parameter(torch.randn(width) if width else torch.randn(1)) for _ in range(layers)]
        self.output_layer = torch.nn.Parameter(torch.randn(self.num_classes, width) if width else torch.randn(self.num_classes, 1))
        self.b = torch.nn.Parameter(torch.zeros(self.num_classes))

    ################################### Class Helper Functions ##################################
    def he_init(self, shape:tuple) -> torch.Tensor:
        # Calculates the standard deviation
        stddev = torch.sqrt(torch.tensor(2.0) / shape[0])
        # Initializes weights from a normal distribution with mean 0 and calculated stddev
        return torch.normal(0, stddev, size=shape)
    
    
    def encode_y(self, y:torch.Tensor) -> None:
        """
        Encode the labels of the data.
        Parameters:
            y (n,) Tensor - The labels of the data
        """
        # Check if the input is a list
        if isinstance(y, list):
            y = torch.tensor(y)
        elif isinstance(y, np.ndarray):
            y = torch.tensor(y)
        elif not isinstance(y, torch.Tensor):
            raise ValueError("y must be a list, numpy array, or a tensor")
        
        # If it is not integers, give it a dictionary
        if y.dtype != torch.int64:
            self.classes = torch.unique(y)
            self.y_dict = {label.item(): i for i, label in enumerate(self.classes)}

        # If it is, still make it a dictionary
        else:
            self.classes = torch.arange(torch.max(y) + 1)
            self.y_dict = {i: i for i in self.classes}
        self.num_classes = len(self.classes)

        # Create an index array
        for i in range(self.num_classes):
            self.class_index.append(torch.where(y == self.classes[i])[0])

        # Make a one hot encoding
        self.one_hot = torch.zeros((self.n, self.num_classes), dtype=torch.float32)
        for i in range(self.n):
            self.one_hot[i, self.y_dict[y[i].item()]] = 1

    def get_batches(self) -> list:
        """
        Randomize the batches for stochastic gradient descent
        Returns:
            batches (list) - A list of batches of indices for training
        """
        # Get randomized indices and calculate the number of batches
        indices = torch.randperm(self.n).tolist()
        num_batches = self.n // self.batch_size

        # Loop through the different batches and get the batches
        batches = [indices[i*self.batch_size:(i+1)*self.batch_size] for i in range(num_batches)]

        # Handle the remaining points
        remaining_points = indices[num_batches*self.batch_size:]
        counter = len(remaining_points)
        i = 0

        # Fill the remaining points into the batches
        while counter > 0:
            batches[i % len(batches)].append(remaining_points[i])
            i += 1
            counter -= 1

        # Return the batches (a list of lists)
        return batches

    ############################## Training Helper Functions ##############################
    def learning_rate_decay(self, epoch:int) -> float:
        """
        Calculate the learning rate decay

        Parameters:
            epoch (int) - The current epoch
        Returns:
            None
        """ 
        # Set hyperparameters
        start_decay = .2
        scale_rate = 3
        
        # Get the progress of the training
        progress = epoch / self.epochs
        
        # If the progress is less than the start decay, use the scale rate
        if progress < start_decay:
            rate = scale_rate * self.learning_rate**(2 - progress/start_decay)
        # Otherwise, use the exponential decay, append the learning rate to the history, and return it
        else:
            rate = self.learning_rate * (1 + (scale_rate-1) * torch.exp(-(epoch - self.epochs * start_decay) / torch.sqrt(torch.tensor(self.epochs))))
        self.learning_rate_history.append(rate)
        return rate

    def fold(self, Z:torch.Tensor, n:torch.Tensor, leaky:float=None) -> torch.Tensor:
        """
        This function folds the data along the hyperplane defined by the normal vector n
        
        Parameters:
            Z (n,d) Tensor - The data to fold
            n (d,) Tensor - The normal vector of the hyperplane
            leaky (float) - The amount of leak in the fold
        Returns:
            folded (n,d) Tensor - The folded data
        """
        # Make the scaled inner product and the mask
        leaky = self.leak if leaky is None else leaky
        if torch.dot(n, n) == 0:
            n = n + 1e-5
        scales = (Z @ n) / torch.dot(n, n)
        indicator = scales > 1
        indicator = indicator.float()
        indicator = indicator + (1 - indicator) * leaky
        
        # Make the projection and flip the points that are beyond the fold (mask)
        projected = scales.unsqueeze(1) * n
        folded = Z + 2 * indicator.unsqueeze(1) * (n - projected) 
        return folded

    def auto_diff_fold(self, Z:torch.Tensor, n:torch.Tensor, leaky:float=None) -> torch.Tensor:
        """
        This function calculates the derivative of the fold operation
        using PyTorch autodifferentiation
        
        Parameters:
            Z (n,d) Tensor - The data to fold
            n (d,) Tensor - The normal vector of the hyperplane
            leaky (float) - The amount of leak in the fold
        Returns:
            derivative (n,d,d) Tensor - The derivative of the fold operation as a PyTorch tensor
        """
        # return jacrev(self.fold, argnums=1)(Z, n)
        Z.requires_grad_(True)
        n.requires_grad_(True)
        folded = self.fold(Z, n, leaky)
        torch.autograd.backward(folded, torch.ones_like(folded))
        return Z.grad

    def derivative_fold(self, Z:torch.Tensor, n:torch.Tensor, leaky:float=None) -> torch.Tensor:
        """
        This function calculates the derivative of the fold operation
        
        Parameters:
            Z (n,d) Tensor - The data to fold
            n (d,) Tensor - The normal vector of the hyperplane
            leaky (float) - The amount of leak in the fold
        Returns:
            derivative (n,d,d) Tensor - The derivative of the fold operation
        """
        leaky = self.leak if leaky is None else leaky
        # Get the scaled inner product, mask, and make the identity stack
        if torch.dot(n, n) == 0:   
            n = n + 1e-5
        quad_normal = n / torch.dot(n, n)
        scales = Z @ quad_normal
        indicator = scales > 1
        indicator = indicator.float()
        indicator = indicator + (1 - indicator) * leaky
        identity = torch.eye(self.width)

        # Use broadcasting to apply scales along the first axis
        first_component = (1 - scales).unsqueeze(1).unsqueeze(2) * identity
        
        # Calculate the outer product of n and helper, then subtract the input
        outer_product = (2 * scales.unsqueeze(1) * n) - Z
        second_component = outer_product.unsqueeze(2) * quad_normal.unsqueeze(0)
        
        # Return the derivative
        derivative = 2 * indicator.unsqueeze(1).unsqueeze(2) * (first_component + second_component)
        return derivative

    def forward_pass(self, D: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the network
        
        Parameters:
            Z (n,d) Tensor - The data to pass through the network
            
        Returns:
            output (n,num_classes) Tensor - The output from the model
        """
        # Expand to a higher dimension if necessary
        if self.has_expand:
            D = Z.copy()
            Z = Z @ self.input_layer.T 
            output = [D, Z] 
            output = [Z] 
            
        # Get the correct fold function
        if self.sigmoid:
            fold = self.sig_fold
        else:
            fold = self.fold
        
        # Pass through the layers
        for layer in range(self.layers):
            Z = fold(Z, self.fold_vectors[layer])
            output.append(Z)
        
        # Softmax to the output layer
        Z = Z @ self.output_layer + self.b # check for dimension error with b
        output.append(F.softmax(Z, dim=1))
        return output


    def back_propagation(self, indices:torch.Tensor, Z:torch.Tensor=None, freeze_folds:bool=False, freeze_cut:bool=False) -> torch.Tensor:
        """
        The backpropagation step for the network
        
        Parameters:
            Z (n,d) Tensor - The data to pass through the network
            indices (n,) Tensor - The indices of the data
            freeze_folds (bool) - Whether to freeze the folds
            freeze_cut (bool) - Whether to freeze the cut
        Returns:
            gradients (list) - The gradients for all parameters in the network
        """
        if Z is None:
            Z = self.X
        if indices is not None:
            Z = Z[indices]
        one_hot = self.one_hot[indices]
        gradients = []
        
        forward = self.forward_pass(Z)
        softmax = forward[-1]
        output_error = softmax - one_hot

        # Calculate the gradients for the output layer
        # TODO: this is not right here
        if not freeze_cut:
            output_gradient = torch.einsum('ij,ik->ijk', output_error, forward[-2])
            gradients.append(output_gradient)

        # Backpropagate through the layers
        if not freeze_folds:
            # Get the current fold vector
            derivative = self.sig_derivative_fold if self.sigmoid else self.derivative_fold
            sidx = 1 if self.has_expand else 0
            fold_grads = [derivative(forward[i+sidx], self.fold_vectors[i]) for i in range(self.layers)]

            backprop_start = output_error @ self.output_layer.T
            for i in range(self.layers):
                backprop_start = torch.einsum('ij,ijk->ik', backprop_start, fold_grads[-(i+1)])
                gradients.append(torch.sum(backprop_start, dim=0))
                
            if self.has_expand:
                gradients.append(torch.sum(backprop_start @ forward[0], dim=0))
        return gradients



    def descend(self, indices: torch.Tensor, epoch: int, freeze_folds: bool = False, freeze_cut: bool = False) -> list:
        """
        Perform gradient descent on the model.

        Parameters:
            indices (Tensor) - The indices of the data to backpropagate.
            epoch (int) - The current epoch.
            freeze_folds (bool) - Whether to freeze the folds during backpropagation.
            freeze_cut (bool) - Whether to freeze the cut during backpropagation.
            
        Returns:
            gradient (list) - The computed gradients.
        """
        # Get the gradient and learning rate decay
        gradient = self.back_propagation(indices)
        learning_rate = self.learning_rate_decay(epoch)

        # Update the weights of the cut matrix and the cut biases
        if not freeze_cut:
            self.output_layer -= learning_rate * gradient[0]
            self.b -= learning_rate * gradient[1]
        self.cut_history.append(self.output_layer.detach().clone())  # Store a copy of the output layer
        self.b_history.append(self.b.detach().clone())  # Store a copy of the bias

        # Update the fold vectors
        if not freeze_folds:
            for i in range(self.layers):
                self.fold_vectors[i] -= learning_rate * gradient[i + 2]
            self.fold_history.append([vec.detach().clone() for vec in self.fold_vectors])  # Store copies of the fold vectors
        
        # Update the expand matrix if necessary
        if self.has_expand:
            self.input_layer -= learning_rate * gradient[-1]
            self.expand_history.append(self.input_layer.detach().clone())  # Store a copy of the input layer

        # Return the gradient
        return gradient


    ############################## Train the Model ##############################
    def gradient_descent(self, validate:bool=None, freeze_folds:bool=False, freeze_cut:bool=False, epochs=None, verbose=0):
        """
        Perform gradient descent on the model
        Parameters:
            validate (bool - Whether to validate the model
            freeze_folds (bool) - Whether to freeze the folds during back propogation
            freeze_cut (bool) - Whether to freeze the cut during back propogation
            epochs (int): The number of iterations to run
            verbose (int) - Whether to show the progress of the training (default is 0)
        """
        # Robustly handle variables and initialize the loop
        epochs = self.epochs if epochs is None else epochs
        validate = self.validate if validate is None else validate
        val_update_wait = max(epochs // 100, 1)
        loop = tqdm(total=epochs, position=0, leave=True, desc="Training Progress", disable=verbose==0)

        for epoch in range(epochs):
            # Update the gradient on all the data
            gradient = self.descend(np.arange(self.n), epoch, freeze_folds=freeze_folds, freeze_cut=freeze_cut)
            
            # If there is a validation set, validate the model
            if validate and epoch % val_update_wait == 0:
                
                # predict the validation set and get the accuracy
                predictions = self.predict(self.X_val_set)
                val_acc = accuracy_score(predictions, self.y_val_set)
                self.val_history.append(val_acc)

                # Get the training accuracy, append it to the history, set loop description
                train_acc = accuracy_score(self.predict(self.X), self.y)
                self.train_history.append(train_acc)
                loop.set_description(f"Epoch {epoch+1}/{self.epochs} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")
            elif not validate:
                loop.set_description(f"Epoch {epoch+1}/{self.epochs}")
            
            # Freeze the folds if the gradient is small
            if not freeze_folds:
                ipct = 20
                grad_threshold = 5
                rate_increase = 2
                update_rate = max(ipct, epoch // ipct)
                if epoch % update_rate == 0:
                    avg_gradient = torch.mean(torch.stack([torch.mean(torch.abs(grad)) for grad in gradient[2:]]))
                    if avg_gradient < grad_threshold:
                        self.learning_rate *= rate_increase
                        
            # Update the loop
            loop.update()
        loop.close()
        
        # Set up the plot if you want it to validate
        if validate:
            fig, ax = plt.subplots()
            train_line, = ax.plot([], [], label="Training Accuracy", color="blue")
            val_line, = ax.plot([], [], label="Validation Accuracy", color="orange")
            ax.set_xlim(0, self.epochs)
            ax.set_ylim(0, 1)  # Accuracy values between 0 and 1
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.set_yticks(np.arange(0, 1.1, .1))
            ax.legend(loc="lower right")
            ax.set_title(f"Opt: {self.optimizer} -- LR: {self.learning_rate} -- Reg: {self.reg} -- Width: {self.width}")
            
            # Set the data for the plot and show it
            x_data = np.arange(len(self.train_history))
            x_data = x_data/x_data[-1] * self.epochs
            train_line.set_xdata(x_data)
            train_line.set_ydata(self.train_history)
            val_line.set_xdata(x_data)
            val_line.set_ydata(self.val_history)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            plt.show()

