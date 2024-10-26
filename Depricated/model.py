from tqdm import tqdm
import copy
import pdb
import pickle

import numpy as np
from jax import jacrev, jacfwd
import jax.numpy as jnp

from scipy.special import erf
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold

from matplotlib import pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots




class OrigamiNetwork():
    """Put in our final docstring here"""
    
    ################################### Initialization ##################################
    def __init__(self, layers:int=3, width:int=None, learning_rate:float=0.001, reg:float=10, sigmoid = False,
                 optimizer:str="grad", batch_size:int=32, epochs:int=100, leak:float=0, crease:float=1):
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




    ################################### Class Helper Functions ##################################
    # initialization method
    def he_init(self, shape:tuple) -> np.ndarray:
        # Calculates the standard deviation
        stddev = np.sqrt(2 / shape[0])
        # Initializes weights from a normal distribution with mean 0 and calculated stddev
        return np.random.normal(0, stddev, size=shape)
    
    
    def encode_y(self, y:np.ndarray) -> None:
        """
        Encode the labels of the data.
        Parameters:
            y (n,) ndarray - The labels of the data
        """
        # Check if the input is a list
        if isinstance(y, list):
            y = np.array(y)

        # Make sure it is a numpy array
        elif not isinstance(y, np.ndarray):
            raise ValueError("y must be a list or a numpy array")
        
        # If it is not integers, give it a dictionary
        if y.dtype != int:
            self.classes = np.unique(y)
            self.y_dict = {label: i for i, label in enumerate(np.unique(y))}

        # If it is, still make it a dictionary
        else:
            self.classes = np.arange(np.max(y)+1)
            self.y_dict = {i: i for i in self.classes}
        self.num_classes = len(self.classes)

        # Create an index array
        for i in range(self.num_classes):
            self.class_index.append(np.where(y == self.classes[i])[0])

        # Make a one hot encoding
        self.one_hot = np.zeros((self.n, self.num_classes))
        for i in range(self.n):
            self.one_hot[i, self.y_dict[y[i]]] = 1

        
    def get_batches(self) -> list:
        """
        Randomize the batches for stochastic gradient descent
        Returns:
            batches (list) - A list of batches of indices for training
        """
        # Get randomized indices and calculate the number of batches
        indices = np.arange(self.n)
        np.random.shuffle(indices)
        num_batches = self.n // self.batch_size

        # Loop through the different batches and get the batches
        batches = [indices[i*self.batch_size:(i+1)*self.batch_size].tolist() for i in range(num_batches)]

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
            rate = self.learning_rate*(1 + (scale_rate-1)*np.exp(-(epoch-self.epochs*start_decay)/np.sqrt(self.epochs)))
        self.learning_rate_history.append(rate)
        return rate # self.learning_rate


    def fold(self, Z:np.ndarray, n:np.ndarray, leaky:float=None) -> np.ndarray:
        """
        This function folds the data along the hyperplane defined by the normal vector n
        
        Parameters:
            Z (n,d) ndarray - The data to fold
            n (d,) ndarray - The normal vector of the hyperplane
            leaky (float) - The amount of leak in the fold
        Returns:
            folded (n,d) ndarray - The folded data
        """
        # Make the scaled inner product and the mask
        leaky = self.leak if leaky is None else leaky
        if np.dot(n, n) == 0:
            n = n + 1e-5
        try:
            scales = (Z@n)/np.dot(n, n)
        except ValueError:
            print("Divide by zero error I think")
            print("Z shape:", Z.shape)
            print("n shape:", n.shape)
            print("n@n:", type(np.dot(n, n)))
        indicator = scales > 1
        indicator = indicator.astype(int)
        indicator = indicator + (1 - indicator) * leaky
        
        # Make the projection and flip the points that are beyond the fold (mask)
        projected = np.outer(scales, n)
        folded = Z + 2 * indicator[:,np.newaxis] * (n - projected)
        return folded


    def auto_diff_fold(self, Z:np.ndarray, n:np.ndarray, leaky:float=None) -> jnp.ndarray:
        """
        This function calculates the derivative of the fold operation
        using jax autodifferentiation
        
        Parameters:
            Z (n,d) ndarray - The data to fold
            n (d,) ndarray - The normal vector of the hyperplane
            leaky (float) - The amount of leak in the fold
        Returns:
            derivative (n,d,d) jndarray - The derivative of the fold operation as a jax tensor
        """
        return jacrev(self.fold, argnums=1)(Z, n)
    
    
    def derivative_fold(self, Z:np.ndarray, n:np.ndarray, leaky:float=None) -> np.ndarray:
        """
        This function calculates the derivative of the fold operation
        
        Parameters:
            Z (n,d) ndarray - The data to fold
            n (d,) ndarray - The normal vector of the hyperplane
            leaky (float) - The amount of leak in the fold
        Returns:
            derivative (n,d,d) ndarray - The derivative of the fold operation
        """
        leaky = self.leak if leaky is None else leaky
        # Get the scaled inner product, mask, and make the identity stack
        if np.dot(n, n) == 0:   
            n = n + 1e-5
        quad_normal = n / np.dot(n, n)
        scales = Z @ quad_normal
        indicator = scales > 1
        indicator = indicator.astype(int)
        indicator = indicator + (1 - indicator) * leaky
        identity = np.eye(self.width)

        # Use broadcasting to apply scales along the first axis
        first_component = (1 - scales)[:, np.newaxis, np.newaxis] * identity
        
        # Calculate the outer product of n and helper, then subtract the input
        outer_product = np.outer(2 * scales, n) - Z
        second_component = np.einsum('ij,k->ijk', outer_product, quad_normal)
        
        # Return the derivative
        derivative = 2 * indicator[:,np.newaxis, np.newaxis] * (first_component + second_component)
        return derivative


    def forward_pass(self, D:np.ndarray) -> list:
        """
        Perform a forward pass of the data through the model

        Parameters:
            D (n,d) ndarray - The data to pass through the model
        
        Returns:
            output list - The output of the model at each layer    
        """
        # Expand to a higher dimension if necessary
        if self.has_expand:
            Z = D @ self.input_layer.T
            output = [D, Z]
            input = Z
        
        # If there is no expand matrix, just use the data
        else:
            output = [D]
            input = D
        
        # Get the correct fold function
        if self.sigmoid:
            fold = self.sig_fold
        else:
            fold = self.fold
        
        # Loop through the different layers and fold the data
        for i in range(self.layers):
            folded = fold(input, self.fold_vectors[i])
            output.append(folded)
            input = folded
        
        # make the final cut with the softmax
        cut = input @ self.output_layer.T + self.b[np.newaxis,:]
        # normalize the cut for softmax
        cut -= np.max(cut, axis=1, keepdims=True)
        exponential = np.exp(cut)
        softmax = exponential / np.sum(exponential, axis=1, keepdims=True)
        output.append(softmax)
        return output
    
    
    def back_propagation(self, indices:np.ndarray, freeze_folds:bool=False, freeze_cut:bool=False) -> list:
        """
        Perform a back propagation of the data through the model

        Parameters:
            indices (ndarray) - The indices of the data to back propagate
            freeze_folds (bool) - Whether to freeze the folds during back propogation
            freeze_cut (bool) - Whether to freeze the cut during back propogation
        Returns:
            gradient list - The gradient of the model (ndarrays)
        """
        # Get the correct one hot encoding and the correct data and initialize the gradient
        D = self.X[indices]
        one_hot = self.one_hot[indices]
        gradient = []
        
        # Run the forward pass and get the softmax and outer layer
        forward = self.forward_pass(D)
        softmax = forward[-1]
        outer_layer = softmax - one_hot
        
        # Make the b and W gradient and append them to the gradient
        if not freeze_cut:
            dW = np.einsum('ik,id->kd', outer_layer, forward[-2])
            db = np.sum(outer_layer, axis=0)
            gradient.append(dW)
            gradient.append(db)
        
        
        # Calculate the gradients of each fold using the forward propogation
        if not freeze_folds:
            # Get the correct fold function
            derivative = self.derivative_fold if not self.sigmoid else self.sig_derivative_fold
            start_index = 1 if self.has_expand else 0
            fold_grads = [derivative(forward[i+start_index], self.fold_vectors[i]) for i in range(self.layers)]
        
            # Perform the back propogation for the folds
            backprop_start = outer_layer @ self.output_layer
            for i in range(self.layers):
                backprop_start = np.einsum('ij,ijk->ik', backprop_start, fold_grads[-i-1])
                gradient.append(np.sum(backprop_start, axis=0))
            
            # If there is an expand matrix, calculate the gradient for that
            if self.has_expand:
                dE = np.einsum('ik,id->kd', backprop_start, forward[0])
                gradient.append(dE)
            
        return gradient
      
        
        
    
    ########################## Optimization and Training Functions ############################
    def descend(self, indices:np.ndarray, epoch:int, freeze_folds:bool=False, freeze_cut:bool=False) -> list:
        """
        Perform gradient descent on the model
        Parameters:
            indices (ndarray) - The indices of the data to back propagate
            epoch (int) - The current epoch
            freeze_folds (bool) - Whether to freeze the folds during back propogation
            freeze_cut (bool) - Whether to freeze the cut during back propogation
        """
        # Get the gradient and learning rate decay
        gradient = self.back_propagation(indices)
        learning_rate = self.learning_rate_decay(epoch)

        # Update the weights of the cut matrix and the cut biases
        if not freeze_cut:
            self.output_layer -=  learning_rate * gradient[0]
            self.b -= learning_rate * gradient[1]
        self.cut_history.append([self.output_layer.copy(), self.b.copy()])

        # Update the fold vectors
        if not freeze_folds:
            for i in range(self.layers):
                self.fold_vectors[i] -= learning_rate * gradient[i+2]
        self.fold_history.append(self.fold_vectors.copy())
            
        # Update the expand matrix if necessary
        if self.has_expand:
            self.input_layer -= learning_rate * gradient[-1]
            self.expand_history.append(self.input_layer.copy())
        
        # Return  the gradient
        return gradient


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
                    avg_gradient = [np.mean(np.abs(g.ravel()), axis=0) for g in gradient]
                    if np.linalg.norm(avg_gradient) < grad_threshold:
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


    def stochastic_gradient_descent(self, validate:bool=None, freeze_folds:bool=False, freeze_cut:bool=False, epochs=None, verbose=1):
        """
        Perform stochastic gradient descent on the model
        Parameters:
            validate (bool) - Whether to validate the model
            freeze_folds (bool) - Whether to freeze the folds during back propogation
            freeze_cut (bool) - Whether to freeze the cut during back propogation
            epochs (int) - The number of iterations to run
            verbose (int) - Whether to show the progress of the training (default is 1)
        """
        
        # Loop through the epochs, get the batches, and update the loop
        epochs = self.epochs if epochs is None else epochs
        validate = self.validate if validate is None else validate
        val_update_wait = max(epochs // 100, 1)
        loop = tqdm(total=epochs, position=0, leave=True, disable=verbose==0)

        # Loop through the epochs and get the batches
        for epoch in range(epochs):
            batches = self.get_batches()
            
            # Loop through the batches and descend
            for batch in batches:
                self.descend(batch, epoch, freeze_folds=freeze_folds, freeze_cut=freeze_cut)
            
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
                
                # Update the loop
            loop.update(1)
        loop.close()
        
        # Set up the plot if you want it to validate
        if validate:
            fig, ax = plt.subplots()
            train_line, = ax.plot([], [], label="Training Accuracy", color="blue")
            val_line, = ax.plot([], [], label="Validation Accuracy", color="orange")
            ax.set_xlim(0, epochs)
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
    
     
    def adam(self):
        raise NotImplementedError("Adam is not implemented yet")


    def fit(self, X:np.ndarray=None, y:np.ndarray=None, X_val_set=None, y_val_set=None, 
            freeze_folds:bool=False, freeze_cut:bool=False, epochs=None, verbose=1):
        """
        Fit the model to the data

        Parameters:
            X (n,d) ndarray - The data to fit the model on
            y (n,) ndarray - The labels of the data
            X_val_set (n_val,d) ndarray - The validation set for the data
            y_val_set (n_val,) ndarray - The validation labels for the data
            freeze_folds (bool) - Whether to freeze the folds during back propogation
            freeze_cut (bool) - Whether to freeze the cut during back propogation
            epochs (int) - The maximum number of iterations to run
            verbose (int) - Whether to show the progress of the training (default is 1)
        Returns:
            train_history (list) - A list of training accuracies
            val_history (list) - A list of validation accuracies
            fold_history (list) - A list of the fold vectors at each iteration
        """
        # Robustly handle the variables
        epochs = self.epochs if epochs is None else epochs
        if X is None and self.X is None:
            raise ValueError("X must be provided")
        if y is None and self.y is None:
            raise ValueError("y must be provided")
        
        # Save the data as variables and encode y
        self.X = np.array(X) if X is not None else self.X
        self.y = np.array(y) if y is not None else self.y
        self.n = self.X.shape[0]
        self.d = self.X.shape[1]
        self.encode_y(self.y)
        #test comment

        # Initialize the expand matrix if necessary
        if self.has_expand:
            self.input_layer = self.he_init((self.width,self.d))
        else:
            self.width = self.d
            
        # Initialize the cut matrix, fold vectors, and biases
        if not freeze_cut:
            self.output_layer = self.he_init((self.num_classes, self.width))
            self.b = np.random.rand(self.num_classes)
        elif self.output_layer is None:
            raise ValueError("Output layer must be initialized")
        if not freeze_folds and self.fold_vectors is None:
            self.fold_vectors = [] if self.layers == 0 else self.he_init((self.layers, self.width))

        # If there is a validation set, save it
        if X_val_set is not None and y_val_set is not None:
            self.X_val_set = X_val_set
            self.y_val_set = y_val_set
            self.validate = True

        # Run the optimizer
        if self.optimizer == "sgd":
            self.stochastic_gradient_descent(freeze_folds=freeze_folds, freeze_cut=freeze_cut, epochs=epochs, verbose=verbose, validate=self.validate)
        elif self.optimizer == "grad":
            self.gradient_descent(freeze_folds=freeze_folds, freeze_cut=freeze_cut, epochs=epochs, verbose=verbose, validate=self.validate)
        elif self.optimizer == "adam":
            self.adam()
        else:
            raise ValueError("Optimizer must be 'sgd', 'grad', or 'adam'")
        return self.get_history()
        
        
        
        
    ############################## Evaluation Functions #############################
    def predict(self, points:np.ndarray=None, show_probabilities=False):
        """
        Predict the labels of the data

        Parameters:
            points (n,d) ndarray - The data to predict the labels of
            show_probabilities (bool) - Whether to show the probabilities of the classes
        Returns:
            predictions (n,) ndarray - The predicted labels of the data
        """
        # Get the probabilities of the classes
        points = self.X if points is None else points
        probabilities = self.forward_pass(points)[-1]
        if show_probabilities:
            return probabilities
        
        else:        
            # Get the predictions
            predictions = np.argmax(probabilities, axis=1)
            # Get the dictionary of the predictions
            return np.array([self.classes[prediction] for prediction in predictions])
    
        
        
        
    ################################## Analysis Functions #################################
    
    
    
    
    ############################## Visualitzation Functions ###############################

    def plot_history(self, X:np.ndarray=None, Y:np.ndarray=None, 
                    n_folds:int=50, resolution:float=0.02, include_cut:bool=True) -> None:
        """
        Plot the decision boundaries of a model as lines.

        Parameters:
            X (n,2) ndarray: The input data used to define the range of the plot (only works in 2D).
            y (n,) ndarray: The true labels of the data (for plotting the points).
            n_folds (int): The number of folds to plot.
            resolution (float): The resolution of the grid for plotting the decision boundary.
            include_cut (bool): Whether to include the cut layer in the plot.
        """
        # initialize input
        X = self.X if X is None else X
        Y = self.y if Y is None else Y
        mod_number = self.epochs // n_folds
        scalor = 1 / (2 * self.epochs / mod_number)
        
        # get pure colors for color scale
        length = 255
        cmap = [plt.get_cmap('spring')(i) for i in range(0, length, length//n_folds)]
        cmap = np.array([np.array(cmap[i][:-1])*length for i in range(n_folds)], dtype=int)
        colors = ['#%02x%02x%02x' % tuple(cmap[i]) for i in range(n_folds)]

        # set up grid
        if include_cut:
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                                np.arange(y_min, y_max, resolution))
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            og_output_layer = copy.deepcopy(self.output_layer)
            og_b_thing = copy.deepcopy(self.b)
            
        """# Setup subplots grid (2 columns, dynamic number of rows based on layers)
        n_rows = (self.layers + 1) // 2
        fig = make_subplots(rows=n_rows, cols=2, subplot_titles=[f"Layer {i+1}" for i in range(self.layers)])

        # Loop over each layer and fold
        out = self.forward_pass(X)
        progress = tqdm(total=n_folds*self.layers, desc="Plotting")
        for fold_idx in range(self.layers):
            outx = out[fold_idx][:, 0]
            outy = out[fold_idx][:, 1]
            subplot_idx = fold_idx + 1
            row_idx = (subplot_idx + 1) // 2
            col_idx = (subplot_idx % 2)
            
            # Plot every mod_number fold and decision boundary
            for hist_idx in range(0, self.epochs, mod_number):
                color_idx = hist_idx // mod_number
                fold = self.fold_history[hist_idx][fold_idx]
                hyperplane_plot = self.idraw_fold(fold, outx, outy, color=colors[color_idx])

                # Add hyperplane plot to the subplot
                # fig.add_trace(hyperplane_plot, row=row_idx, col=col_idx)
                
                if include_cut:
                    self.output_layer = self.cut_history[hist_idx][0]
                    self.b = self.cut_history[hist_idx][1]
                    Z = self.predict(grid_points)
                    Z = Z.reshape(xx.shape)
                    
                    # Add decision boundary as contour lines (approximated with scatter plot)
                    boundary_trace = go.Contour(x=np.arange(x_min, x_max, resolution), 
                                                y=np.arange(y_min, y_max, resolution),
                                                z=Z, showscale=False,
                                                # contours_coloring='lines', 
                                                colorscale='Viridis', opacity=1)
                    fig.add_trace(boundary_trace, row=row_idx, col=col_idx)
                # set xlim and ylim
                fig.update_xaxes(range=[np.min(outx), np.max(outx)], row=row_idx, col=col_idx)
                progress.update(1)
                break
        progress.close()

        # Reset the output layer and bias term
        self.output_layer = og_output_layer.copy()
        self.b = og_b_thing.copy()
        hyperplane = np.round(self.fold_vectors[-1], 2)
        last_plot_col = 1 if self.layers % 2 == 0 else 2
        
        if include_cut:
            # Z = self.predict(grid_points)
            # Z = Z.reshape(xx.shape)
            # boundary_trace = go.Contour(x=np.arange(x_min, x_max, resolution), 
            #                             y=np.arange(y_min, y_max, resolution),
            #                             z=Z, showscale=False,
            #                             line_smoothing=0.85,
            #                             colorscale='Greys', opacity=0.5)
            # fig.add_trace(boundary_trace, row=n_rows, col=last_plot_col)
            pass
        outx = out[-2][:,0]
        outy = out[-2][:,1]
        fig.add_trace(go.Scatter(x=outx, y=outy, mode='markers', marker=dict(color=Y)), 
                      row=n_rows, col=last_plot_col)
        fig.update_layout(height=500 * n_rows, width=1000, title="Decision Boundaries and Hyperplanes", showlegend=False)
        fig.show()"""

        # loop over each fold
        out = self.forward_pass(X)
        plt.figure(figsize=(8, 3.5*self.layers), dpi=150)
        n_cols = 2
        n_rows = self.layers // n_cols + 1
        for layer in range(self.layers):
            outx = out[layer][:,0]
            outy = out[layer][:,1]
            progress = tqdm(total=n_folds, desc=f"Plotting Layer {layer+1}")
            
            # plot every mod_number fold and decision boundary
            plt.subplot(n_rows, n_cols, layer+1)
            for step in range(0, len(self.fold_history), mod_number):
                color_idx = step // mod_number
                self.draw_fold(self.fold_history[step][layer], outx, outy, color=colors[color_idx], name=None)
                if include_cut:
                    self.output_layer = self.cut_history[step][0]
                    self.b = self.cut_history[step][1]
                    Z = self.predict(grid_points)
                    Z = Z.reshape(xx.shape)
                    plt.contourf(xx, yy, Z, alpha=scalor*color_idx, cmap=plt.cm.YlGnBu)
                progress.update(1)
            plt.ylim(np.min(outy), np.max(outy))
            plt.xlim(np.min(outx), np.max(outx)) 
            plt.tick_params(axis='both', which='major', labelsize=6)
            hyperplane = np.round(self.fold_vectors[layer], 2)
            plt.title(f"Layer {layer+1}: {hyperplane}", fontsize=8)
            if layer % n_cols == 0:
                plt.ylabel("Feature 2", fontsize=6)
            if layer >= n_cols * (n_rows - 1):
                plt.xlabel("Feature 1", fontsize=6)
            progress.close()

        # reset the output layer and b
        self.output_layer = og_output_layer.copy()
        self.b = og_b_thing.copy()
        
        # plot the final decision boundary
        plt.subplot(n_rows, n_cols, n_rows*n_cols)
        if include_cut:
            Z = self.predict(grid_points)
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.Greens)
        # plot the points
        outx = out[-2][:,0]
        outy = out[-2][:,1]
        plt.scatter(outx, outy, c=Y)
        plt.ylim(np.min(outy), np.max(outy))
        plt.xlim(np.min(outx), np.max(outx))
        plt.title(f"Layer {layer+1}: {hyperplane}")
        plt.xlabel("Feature 1", fontsize=6)
        plt.title("Final Decision Boundary", fontsize=8)
        plt.tick_params(axis='both', which='major', labelsize=6)
        plt.tight_layout()
        plt.show()
        
    
    
    
    def draw_fold(self, hyperplane, outx, outy, color, name):
        """
        This function draws a hyperplane on a plot
        
        Parameters:
            hyperplane (list) - The hyperplane to draw
            outx (list) - The x values of the data
            outy (list) - The y values of the data
            color (str) - The color of the hyperplane
            name (str) - The name of the hyperplane
        """
        plane_domain = np.linspace(np.min(outx), np.max(outx), 100)
        if hyperplane[1] == 0:
            plt.plot([hyperplane[0], hyperplane[0]], [np.min(outy), np.max(outy)], color=color, lw=2, label=name)
        elif hyperplane[0] == 0:
            plt.plot([np.min(outx), np.max(outx)], [hyperplane[1], hyperplane[1]], color=color, lw=2, label=name)
        else:
            a, b = hyperplane
            slope = -a / b
            intercept = b - slope * a
            plane_range = slope * plane_domain + intercept
            # set values outside y range to NaN
            plane_range = np.where((plane_range > np.min(outy)) & (plane_range < np.max(outy)), plane_range, np.nan)
            plt.plot(plane_domain, plane_range, color=color, lw=2, label=name)
    
    
    
    def idraw_fold(self, hyperplane, outx, outy, color, name=None):
        """
        This function draws a hyperplane on a plot using Plotly
        
        Parameters:
            hyperplane (list) - The hyperplane to draw
            outx (list) - The x values of the data
            outy (list) - The y values of the data
            color (str) - The color of the hyperplane
            name (str) - The name of the hyperplane
        """
        plane_domain = np.linspace(np.min(outx), np.max(outx), 100)
        if hyperplane[1] == 0:
            return go.Scatter(x=[hyperplane[0], hyperplane[0]], y=[np.min(outy), np.max(outy)], 
                            mode="lines", line=dict(color=color, width=2), name=name)
        elif hyperplane[0] == 0:
            return go.Scatter(x=[np.min(outx), np.max(outx)], y=[hyperplane[1], hyperplane[1]], 
                            mode="lines", line=dict(color=color, width=2), name=name)
        else:
            a, b = hyperplane
            slope = -a / b
            intercept = b - slope * a
            plane_range = slope * plane_domain + intercept
            # Keep values inside y range
            plane_range = np.where((plane_range > np.min(outy)) & (plane_range < np.max(outy)), plane_range, np.nan)
            return go.Scatter(x=plane_domain, y=plane_range, mode="lines", 
                            line=dict(color=color, width=2), name=name)


    def iscore_landscape(self, score_layers:int=None, X:np.ndarray=None, y:np.ndarray=None, 
                       feature_mins:list=None, feature_maxes:list=None, density:int=10, 
                       f1id:int=0, f2id:int=1, create_plot:bool=False, png_path:str=None, theme:str="viridis",
                       learning:bool=False, verbose:int=0):
        """
        This function visualizes the score landscape of the model for a given layer and two features.
        
        Parameters:
            score_layers (int) - The layer to calculate the score landscape for
            X (n,d) ndarray - The data to calculate the score landscape on
            y (n,) ndarray - The labels of the data
            feature_mins (list) - The minimum values for each feature
            feature_maxes (list) - The maximum values for each feature
            density (int) - The number of points to calculate the score for
            f1id (int) - The id of the first feature to calculate the score for
            f2id (int) - The id of the second feature to calculate the score for
            create_plot (bool) - Whether to create a plot of the score landscape
            png_path (str) - The path to save the plot to
            theme (str) - The theme of the plot
            learning (bool) - Whether to learn from the maximum score and features
            verbose (int) - Whether to show the progress of the training (default is 1)
        Returns:
            max_score (float) - The maximum score of the model
            max_features (list) - The features that produced the maximum score
        """
        # set default values
        X = self.X
        y = self.y
        density = [density]*self.d if density is not None else [10]*self.d
        feature_mins = feature_mins if feature_mins is not None else np.min(X, axis=0)
        feature_maxes = feature_maxes if feature_maxes is not None else np.max(X, axis=0)
        score_layers = score_layers if type(score_layers) == list else [score_layers] if type(score_layers) == int else [l for l in range(self.layers)]
        og_fold_vectors = copy.deepcopy(self.fold_vectors)
        og_output_layer = copy.deepcopy(self.output_layer)
        og_b = copy.deepcopy(self.b)

        # input error handling
        assert type(X) == np.ndarray and X.shape[0] > 0 and X.shape[1] > 0, f"X must be a 2D numpy array. Instead got {type(X)}"
        assert type(y) == np.ndarray, f"y must be a numpy array. Instead got {type(y)}"
        assert type(score_layers) == int or (type(score_layers) == list and len(score_layers) > 0 and type(score_layers[0]) == int), f"score_layer must be an integer. instead got {score_layers}"
        assert type(density) == list or (len(density) > 0 and type(density[0]) == int), f"Density must be a list of integers. Instead got {density}"
        
        # create a grid of features
        feature_folds = []
        for mins, maxes, d in zip(feature_mins, feature_maxes, density):
            feature_folds.append(np.linspace(mins, maxes, d))
        feature_combinations = np.array(np.meshgrid(*feature_folds)).T.reshape(-1, self.d)            
        
        # compute scores for each feature combination and each layer
        max_scores = []
        max_features_list = []
        for score_layer in score_layers:
            scores = []
            for features in tqdm(feature_combinations, position=0, leave=True, disable=verbose==0, desc=f"score Layer {score_layer}"):
                self.fold_vectors[score_layer] = features
                self.output_layer = og_output_layer.copy()
                # self.fit(epochs=50, freeze_folds=True, verbose=0)
                scores.append(self.score(X, y))
                
            # find the maximum score and the features that produced it
            scores = np.array(scores)
            max_score = np.max(scores)
            max_index = np.argmax(scores)
            max_scores.append(max_score)
            max_features_list.append(feature_combinations[max_index])
            
            # create a heatmap of the score landscape for features f1id and f2id
            if create_plot:
                f1 = feature_combinations[:,f1id]
                f2 = feature_combinations[:,f2id]
                f1_folds = feature_folds[f1id]
                f2_folds = feature_folds[f2id]

                # Get the heatmap data
                mesh = np.zeros((len(f2_folds), len(f1_folds)))
                for i, f1_val in enumerate(f1_folds):
                    for j, f2_val in enumerate(f2_folds):
                        mesh[j, i] = scores[np.where((f1 == f1_val) & (f2 == f2_val))[0][0]]

                offset = 1 if self.has_expand else 0
                self.fold_vectors = og_fold_vectors.copy()
                paper = self.forward_pass(X)
                outx = paper[offset + score_layer][:,f1id]
                outy = paper[offset + score_layer][:,f2id]

                # Create subplots
                fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Input Data", f"Score Landscape"), 
                                    specs=[[{"type": "scatter"}, {"type": "heatmap"}]])

                # Scatter plot with colors
                
                fig.add_trace(go.Scatter(x=outx, y=outy, mode='markers', 
                                        marker=dict(color=y.astype(int)*.5+0.2, colorscale=theme, size=8), 
                                        name="Data", showlegend=False), row=1, col=1)

                # Add predicted and maximum folds
                pred_fold = self.fold_vectors[score_layer]
                best_fold = max_features_list[-1]
                fig.add_trace(self.idraw_fold(pred_fold, outx, outy, color="red", 
                                        name=f"Predicted Fold ({round(pred_fold[f1id], 2)}, {round(pred_fold[f2id], 2)})"), row=1, col=1)
                fig.add_trace(self.idraw_fold(best_fold, outx, outy, color="black", 
                                        name=f"Maximum Score Fold ({round(best_fold[f1id], 2)}, {round(best_fold[f2id], 2)})"), row=1, col=1)

                # Heatmap
                fig.add_trace(go.Heatmap(z=mesh, x=f1_folds, y=f2_folds, colorscale=theme, zmin=np.min(mesh)*0.99, zmax=np.max(mesh)*1.01,), row=1, col=2)

                # Point on the max score
                max_index = np.unravel_index(np.argmax(mesh), mesh.shape)
                max_x = f1_folds[max_index[1]]
                max_y = f2_folds[max_index[0]]
                fig.add_trace(go.Scatter(x=[max_x], y=[max_y], mode='markers', 
                                        marker=dict(color='red', size=8), name=f"Max={round(max_score, 2)}", showlegend=False),
                            row=1, col=2)

                # Update layout
                fig.update_xaxes(title_text=f"Feature {f1id}", row=1, col=1)
                fig.update_yaxes(title_text=f"Feature {f2id}", row=1, col=1)
                fig.update_xaxes(title_text=f"Feature {f1id}", row=1, col=2)
                fig.update_yaxes(title_text=f"Feature {f2id}", row=1, col=2)

                fig.update_layout(height=500, width=1000, 
                                  title_text=f"Layer {score_layer} Visualization", 
                                  showlegend=True, 
                                  legend=dict(
                                        x=0.5, 
                                        y=-0.2,
                                        xanchor="center",
                                        yanchor="bottom" ))

                # Save plot if png_path is provided
                if png_path:
                    fig.write_image(png_path)
                fig.show()
            else:
                self.fold_vectors = og_fold_vectors
                self.output_layer = og_output_layer
                self.b = og_b
        
            if learning:
                # update weights with the best fold for this layer
                self.fold_vectors[score_layer] = max_features_list[-1].copy()
                # update input and output layers
                self.fit(freeze_folds=True, verbose=0, epochs=50)
                if max_scores[-1] >= 0.98:
                    break
        
        if len(max_scores) == 1:
            return max_scores[0], max_features_list[0]
        return max_scores, max_features_list
    
    
    
    
    
    
    
    
    ############################## Saving and Loading Functions ###########################
    
    
    def set_params(self, **kwargs):
        """
        Set the parameters of the model
        Parameters:
            **kwargs - The parameters to set
        """
        # TODO: Test this function
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except Exception as e:
                print(f"Could not set {key} to {value}. Error: {e}")
    

    def get_params(self) -> dict:
        """
        Get the parameters of the model
        Returns:
            params (dict) - The parameters of the model
        """
        params = self.__dict__.copy()
        
        # remove anything too big
        pop_list = []
        max_size = max(self.d, self.width, self.layers)
        for attr in params:
            val = params[attr]
            if type(val) == np.ndarray or type(val) == list:
                if type(val) == list:
                    check = len(val[0]) if len(val) > 0 and (type(val[0]) == np.ndarray or type(val[0]) == list) else 1
                    size = max(len(val), check)
                else:
                    size = val.shape[0] * val.shape[1] if len(val.shape) > 1 else val.shape[0]
                if size > max_size:
                    pop_list.append(attr)
        for attr in pop_list:
            params.pop(attr, None)
        return params
    
    
    
    def get_history(self, history:str=None):
        """
        Get the history of the model
        Parameters:
            history (str) - The history to get
        Returns:
            history (list) - The history of the model
        """
        libary = ["train", "val", "fold", "cut", "expand", "learning_rate"]
        if history is None:
            return self.train_history, self.val_history, self.fold_history, self.cut_history, self.expand_history, self.learning_rate_history
        elif history.lower() in libary:
            return getattr(self, f"{history}_history")
    


    def score(self, X:np.ndarray=None, y:np.ndarray=None):
        """
        Get the accuracy of the model on the data
        
        Parameters:
            X (n,d) ndarray - The data to score the model on
            y (n,) ndarray - The labels of the data
        Returns:
            accuracy (float) - The accuracy of the model on the data
        """
        # If the data is not provided, use the training data
        if X is None:
            X = self.X
            y = self.y

        # Get the predictions and return the accuracy
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
    
    

    def cross_val_score(self, X: np.ndarray = None, y: np.ndarray = None, cv=5) -> list:
        """
        Get the cross-validated accuracy of the model on the data.

        Parameters:
            X (n,d) ndarray - The data to score the model on
            y (n,) ndarray - The labels of the data
            cv (int) - The number of cross-validation splits
        Returns:
            scores (list) - The accuracy of the model on the data for each split
        """
        X = self.X if X is None else X
        y = self.y if y is None else y
        
        # Initialize the k-fold cross-validation
        kf = KFold(n_splits=cv)
        scores = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Fit the model on the training data and get the score on the test data
            self.fit(X_train, y_train)
            scores.append(self.score(X_test, y_test))
        return scores
    


    def confusion_matrix(self, X:np.ndarray=None, y:np.ndarray=None):
        """
        Get the confusion matrix of the model on the data
        
        Parameters:
            X (n,d) ndarray - The data to get the confusion matrix for
            y (n,) ndarray - The labels of the data
        Returns:
            confusion_matrix (num_classes,num_classes) ndarray - The confusion matrix of the model
        """
        # If the data is not provided, use the training data
        X = self.X if X is None else X
        y = self.y if y is None else y

        # Get the predictions and return the confusion matrix
        predictions = self.predict(X)
        return confusion_matrix(y, predictions, labels=self.classes)



    def copy(self, deep=False):
        """
        Create a copy of the model

        Parameters:
            None
        Returns:
            new_model (model3 class) - A copy of the model
        """
        # Initialize a new model
        new_model = OrigamiNetwork()
        
        # Copy the other attributes
        param_list = self.__dict__.copy() if deep else self.get_params()
        for param in param_list:
            setattr(new_model, param, param_list[param])
        return new_model
    


    def save_weights(self, file_path:str="origami_weights", save_type:str="standard", verbose:int=0):
        """
        Save the weights of the model to a file so that it can be loaded later
        Parameters:
            file_path (str) - The name of the file to save the weights to
            save_type (str) - How much of the model to save
                "full" - Save the full model and all of its attributes
                "standard" - Save the standard attributes of the model
                "weights" - Save only the weights of the model
            verbose (int) - Whether to show the result success
        """
        if save_type not in ["full", "standard", "weights"]:
            raise ValueError("save_type must be 'full', 'standard', or 'weights'")
        
        preferences = {"fold_vectors": self.fold_vectors,
                       "input_layer": self.input_layer,
                       "output_layer": self.output_layer,
                       "b": self.b,
                       "save_type": save_type}
        if save_type == "standard":
            preferences.update(self.get_params())
        
        if save_type == "full":
            preferences.update(self.__dict__)

        try:
            if "." in file_path:
                file_path = file_path.split(".")[0]
            with open(f'{file_path}.pkl', 'wb') as f:
                pickle.dump(preferences, f)
        except Exception as e:
            print(e)
            raise ValueError(f"The file '{file_path}.pkl' could not be saved.")
        if verbose > 0:
            print(f"The weights were saved to '{file_path}.pkl'")
    
    

    def load_weights(self, file_path:str="origami_weights", verbose:int=0):
        """
        Load the weights of the model from a file
        Parameters:
            file_path (str) - The name of the file to load the weights from
            verbose (int) - Whether to show the result success
        """
        try:
            with open(f'{file_path}.pkl', 'rb') as f:
                data = pickle.load(f)

            for key, value in data.items():
                try:
                    setattr(self, key, value)
                except Exception as e:
                    print(f"Could not set {key} to {value}.\nError: {e}")
            
            if verbose > 0:
                print(f"The weights were loaded from '{file_path}.pkl'")
        except Exception as e:
            print(e)
            raise ValueError(f"The file '{file_path}.pkl' could not be loaded")
    
    
        
    ############################## Functions In Development ###############################
    
    def beam_search(self, X:np.ndarray=None, y:np.ndarray=None, beam_width:int=5, max_depth:int=5, 
                    iter:int=10, auto_update:bool=True, verbose=1):
        """
        This function uses beam search to find the best folds for the model.
        Parameters:
            X (n,d) ndarray: The input data to use for the search.
            y (n,) ndarray: The labels of the data.
            beam_width (int): The number of folds to keep at each iteration.
            max_depth (int): The maximum number of iterations to run.
            auto_update (bool): Whether to update the model with the best fold found.
        Returns:
            best_fold (list): The best fold found by the search.
        """
        self.X = X if X is not None else self.X
        self.y = y if y is not None else self.y
        assert self.X is not None, "X must be provided"
        assert self.y is not None, "y must be provided"
        assert self.layers > 0, "The model must have at least one layer"

        def jostle(particle, scale):
            """Slightly perturb the particle with Gaussian noise."""
            return particle + np.random.normal(0, scale, particle.shape)

        # Initialize the beam search with random fold vectors
        beams = [np.random.rand(self.layers, self.X.shape[1]) for _ in range(beam_width)]
        best_folds = None
        best_cut = None
        best_score = -np.inf
        base_scale = np.std(self.X) / 20

        # Run the beam search for the specified depth
        progress = tqdm(total=max_depth*beam_width**2, desc="Beam Search", disable=verbose==0)
        for depth in range(max_depth-1, -1, -1):
            shake_splits = []
            split_scores = []
            scale_factors = [base_scale * i * (depth + 1) for i in range(beam_width)]

            for i, folds in enumerate(beams):
                for k in range(beam_width):
                    if i == 0 and k == 0:
                        # For the first beam, use the current fold vectors without shaking
                        self.fold_vectors = folds
                        self.fit(epochs=iter, verbose=0)
                        shake_splits.append(folds)
                    else:
                        # For other beams, shake the fold vectors based on the scale factor
                        shaken_folds = [jostle(fold, scale_factors[i]) for fold in folds]
                        self.fold_vectors = shaken_folds
                        self.fit(epochs=iter, freeze_folds=True, verbose=0)
                        shake_splits.append(shaken_folds)

                    score = self.score()
                    split_scores.append(score)
                    if score > best_score:
                        best_folds = self.fold_vectors
                        best_score = score
                        best_cut = self.output_layer
                    progress.update(1)
                    progress.set_description(f"Beam Search Depth {depth} | Score: {best_score:.2f}")

            # Keep only the top 'beam_width' beams based on the scores
            beams = [x for _, x in sorted(zip(split_scores, shake_splits), key=lambda pair: pair[0], reverse=True)][:beam_width]
            # for beam in beams:
            #     print(np.round(beam, 2), end=", ")
            # print()
            # for score in split_scores:
            #     print(score, end=", ")
            # print()
            # print("max depth", max_depth, "depth", depth)
        
        progress.close()
        if auto_update:
            self.fold_vectors = best_folds
            self.output_layer = best_cut
        return best_folds, best_cut, best_score

        
        
    
    def sig_fold(self, Z:np.ndarray, n:np.ndarray) -> np.ndarray:
        """
        This function does a soft fold of the data along the hyperplane defined by the normal vector n
        Parameters:
            Z (n,d) ndarray - The data to fold
            n (d,) ndarray - The normal vector of the hyperplane
        Returns:
            folded (n,d) ndarray - The folded data
        """
        # Get the helpful terms to substitute into our fold function
        z_dot_x = (Z@n)
        n_dot_n = np.dot(n, n)
        scales = z_dot_x / n_dot_n
        p = self.crease * (z_dot_x - n_dot_n)
        sigmoid = 1/(1 + np.exp(-p))
        
        # Make the projection and flip the points that are beyond the fold
        projected = np.outer(1-scales, n)
        return Z + 2*sigmoid[:,np.newaxis] * projected


    def sig_derivative_fold(self, Z:np.ndarray, n:np.ndarray) -> np.ndarray:
        """
        This function calculates the derivative of the soft fold operation
        Parameters:
            Z (n,d) ndarray - The data to fold
            n (d,) ndarray - The normal vector of the hyperplane
        Returns:
            derivative (n,d,d) ndarray - The derivative of the fold operation
        """
        # Get the helpful terms to substitute into our derivative fold function
        z_dot_x = (Z@n)
        n_dot_n = np.dot(n, n)
        scales = z_dot_x / n_dot_n
        p = self.crease * (z_dot_x - n_dot_n)
        sigmoid = (1/(1 + np.exp(-p)))[:,np.newaxis, np.newaxis]
        u = n / n_dot_n
        identity_stack = np.stack([np.eye(self.width) for _ in range(len(Z))])
        one_minus_scales = 1 - scales[:,np.newaxis, np.newaxis]
        
        # Calculate the first component and the second, then combine them
        first_component = one_minus_scales * identity_stack
        second_component = np.einsum('ij,k->ikj', np.outer(2*Z@n, u) - Z, u)
        first_half = 2 * sigmoid * (first_component + second_component)
        
        # Calculate the second half of the derivative
        second_half = 2 * self.crease * one_minus_scales * sigmoid * (1-sigmoid) * np.einsum('ij,k->ikj', Z - 2*n[np.newaxis,:], n)
        return first_half + second_half


    # possible use of activation functions in the fold derivative fold
    def gelu(self, x:np.ndarray) -> np.ndarray:
        """
        This function calculates the Gaussian Error Linear Unit of the input 
        It is an alternative to the ReLU activation function that is differentiable at zero
        Parameters:
            x (n,) ndarray - The input to the activation function
        Returns:
            y (n,) ndarray - The output of the activation function
        """
        cdf = 0.5 * (1.0 + erf(x / np.sqrt(2)))
        return x * cdf

