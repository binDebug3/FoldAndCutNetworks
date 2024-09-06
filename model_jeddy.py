# Class dependencies
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import pickle


# Other analysis libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


"""
To import this class into an ipynb file in the same folder:

from model import OrigamiNetwork    
"""



class OrigamiNetwork():
    def __init__(self, layers = 3, width = None, temp = .5, max_iter=1000, tol=1e-8, learning_rate=0.01, reg=.5, optimizer="grad", batch_size=32, epochs=100):
        # Hyperparameters
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.reg = reg
        self.layers = layers
        self.width = width
        self.temp = temp

        # Variables to store
        self.X = None
        self.y = None
        self.n = None
        self.d = None
        self.classes = None
        self.num_classes = None
        self.y_dict = None
        self.one_hot = None
        self.fold_vectors = None
        self.output_layer = None
        self.input_layer = None
        
        # Check if the model has an expand matrix
        if self.width is not None:
            self.has_expand = True
        else:
            self.has_expand = False

        # Validation variables
        self.X_val_set = None
        self.y_val_set = None
        self.class_index = []
        self.val_history = []
        self.train_history = []
        self.weights_history = []
        

    ############################## Helper Functions ###############################
    def encode_y(self, y:np.ndarray):
        """
        Encode the labels of the data.

        Parameters:
            y (n,) ndarray - The labels of the data
        Returns:
            None
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

        
    def randomize_batches(self):
        """
        Randomize the batches for stochastic gradient descent
        Parameters:
            None
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

        # Return the batches
        return batches
    

    def fold(self, Z, n):
        # Make the scaled inner product and the mask
        scales = (Z@n)/np.dot(n, n)
        indicator = scales > 1
        
        # Make the projection and flip the points that are beyond the fold (mask)
        projected = np.outer(scales, n)
        return Z + 2 * indicator[:,np.newaxis] * (n - projected)
    
    
    def derivative_fold(self, Z, n):
        # Get the scaled inner product, mask, and make the identity stack
        n_normal = n / np.dot(n,n)
        scales = Z@n_normal
        indicator = scales > 1

        # Use broadcasting to apply scales along the first axis
        first_component = (1 - scales)[:, np.newaxis, np.newaxis] * np.eye(self.width)
        
        # Calculate the outer product of n and helper, then subtract the input
        outer_product = np.outer(2 * scales, n_normal) - Z
        second_component = np.einsum('ij,k->ikj', outer_product, n_normal)
        
        # Return the derivative
        return 2 * indicator[:,np.newaxis, np.newaxis] * (first_component + second_component)
    
    
    ############################## Training Calculations ##############################
    def forward_pass(self, D:np.ndarray, verbose=0):
        """
        Perform a forward pass of the data through the model

        Parameters:
            D (n,d) ndarray - The data to pass through the model"""
        # Expand to a higher dimension if necessary
        if self.has_expand:
            Z = D @ self.input_layer.T
            output = [D, Z]
            input = Z
        
        # If there is no expand matrix, just use the data
        else:
            output = [D]
            input = D
        
        # Loop through the different layers and fold the data
        for i in range(self.layers):
            folded = self.fold(input, self.fold_vectors[i])
            output.append(folded)
            input = folded
        
        # add a column of ones and make the final cut with the softmax
        cut = np.concatenate((input,np.ones((input.shape[0],1))), axis=1) @ self.output_layer.T
        
        # Normalize the cut and get the softmax
        cut = (cut / np.max(np.abs(cut))) * self.temp * 200
        if verbose:
            print(cut)
        exponential = np.exp(cut)
        softmax = exponential / np.sum(exponential, axis=1, keepdims=True)
        output.append(softmax)

        # Return the output
        return output
    
    def back_propagation(self, indices:np.ndarray):
        """
        Perform a back propagation of the data through the model

        Parameters:
            indices (ndarray) - The indices of the data to back propagate
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
        outer_layer = -(one_hot - softmax) # flipped the sign
        
        # Append ones to the forward pass and calculuate the W gradient, appending to the list
        second_stage_forward = np.concatenate((forward[-2],np.ones((forward[-2].shape[0],1))), axis=1)
        dW = np.einsum('ik,id->kd', outer_layer, second_stage_forward)
        gradient.append(dW)
        
        # Calculate the gradients of each fold using the forward propogation
        start_index = 1 if self.has_expand else 0
        fold_grads = [self.derivative_fold(forward[i + start_index], self.fold_vectors[i]) for i in range(self.layers)]
        
        # Perform the back propogation for the folds
        backprop_start = outer_layer @ self.output_layer[:,:-1]
        for i in range(self.layers):
            backprop_start = np.einsum('ij,ijk->ik', backprop_start, fold_grads[-i-1])
            gradient.append(np.sum(backprop_start, axis=0))
            
        # If there is an expand matrix, calculate the gradient for that
        if self.has_expand:
            dE = np.einsum('ik,id->kd', backprop_start, forward[0])
            gradient.append(dE)
            
        # Return the gradient
        return gradient
        
        
    
    ########################## Optimization and Training Functions ############################
    def gradient_descent(self):
        """
        Perform gradient descent on the model
        Parameters:
            None
        Returns:
            None
        """
        # Loop through and get the gradient
        progress = tqdm(range(self.max_iter), desc="Training", leave=True)
        for i in range(self.max_iter):
            gradient = self.back_propagation(np.arange(self.n))
            
            # Clip any gradients that are too large
            max_norm = 5.0
            for g in gradient:
                np.clip(g, -max_norm, max_norm, out=g)
            
            # Save the weights if regualarization is not 0
            if self.reg != 0:
                cut_reg = self.reg * self.output_layer
                fold_reg = [self.reg * fold for fold in self.fold_vectors]
                if self.has_expand:
                    expand_reg = self.reg * self.input_layer

            # Update the weights of the cut matrix and the cut biases
            self.output_layer -= self.learning_rate * gradient[0]

            # Update the fold vectors
            for i in range(self.layers):
                self.fold_vectors[i] -= self.learning_rate * gradient[i+1]
                
            # Update the expand matrix if necessary
            if self.has_expand:
                self.input_layer -= self.learning_rate * gradient[-1]
                
            # Regularize the weights if it is not 0
            if self.reg != 0:
                self.output_layer -= cut_reg
                for i in range(self.layers):
                    self.fold_vectors[i] -= fold_reg[i]
                if self.has_expand:
                    self.input_layer -= expand_reg
            progress.update(1)
        progress.close()


    def fit(self, X:np.ndarray, y:np.ndarray, X_val_set=None, y_val_set=None):
        """
        Fit the model to the data

        Parameters:
            X (n,d) ndarray - The data to fit the model on
            y (n,) ndarray - The labels of the data
            X_val_set (n_val,d) ndarray - The validation set for the data
            y_val_set (n_val,) ndarray - The validation labels for the data
        Returns:
            train_history (list) - A list of training accuracies
            val_history (list) - A list of validation accuracies
        """
        # Save the data as variables and encode y
        self.X = np.array(X)
        self.y = np.array(y)
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.encode_y(y)

        # Initialize the expand matrix if necessary
        if self.has_expand:
            self.input_layer = .1 * np.random.randn(self.width, self.d)
        else:
            self.width = self.d
           
        # Initialize the cut matrix, fold vectors, and b
        self.output_layer = .1 * np.random.randn(self.num_classes, self.width + 1)
        self.fold_vectors = .1 * np.random.randn(self.layers, self.width)

        # If there is a validation set, save it
        if X_val_set is not None and y_val_set is not None:
            self.X_val_set = X_val_set
            self.y_val_set = y_val_set

        # Run the optimizer
        if self.optimizer == "sgd":
            raise ValueError("Stochastic Gradient Descent is not implemented yet")
            self.stochastic_gradient_descent()
        elif self.optimizer == "grad":
            self.gradient_descent()

        # Otherwise, raise an error
        else:
            raise ValueError("Optimizer must be 'sgd' or 'grad'")
        

    ############################## Prediction Functions #############################
    def predict(self, points:np.ndarray, show_probabilities=False):
        """
        Predict the labels of the data

        Parameters:
            points (n,d) ndarray - The data to predict the labels of
            show_probabilities (bool) - Whether to show the probabilities of the classes
        Returns:
            predictions (n,) ndarray - The predicted labels of the data
        """
        # Get the probabilities of the classes
        probabilities = self.forward_pass(points)[-1]
        
        # Get the predictions
        predictions = np.argmax(probabilities, axis=1)
        
        # Get the dictionary of the predictions
        predictions = np.array([self.classes[prediction] for prediction in predictions])
        
        # Return the predictions
        if show_probabilities:
            return probabilities
        else:
            return predictions