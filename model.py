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
    def __init__(self, layers=3, width=None, max_iter=1000, tol=1e-8, temp=0.5, learning_rate=0.01, reg=10, optimizer="grad", batch_size=32, epochs=100):
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
        self.output_layer = None # cut matrix
        self.input_layer = None # expand matrix
        
        # Check if the model has an input layer (expand matrix)
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
    def set_params(self, **kwargs):
        """
        Set the parameters of the model

        Parameters:
            **kwargs - The parameters to set
        Returns:
            None
        """
        # TODO: Test this function
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except Exception as e:
                print(f"Could not set {key} to {value}. Error: {e}")
    

    def get_params(self):
        """
        Get the parameters of the model

        Parameters:
            None
        Returns:
            params (dict) - The parameters of the model
        """
        # TODO: Test this function
        return self.__dict__
    


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
        raise NotImplementedError("This function is not implemented correctly yet")
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
        """
        This function folds the data Z along the normal vector n
        
        Parameters:
            Z (n,d) ndarray - The data to fold
            n (d,) ndarray - The normal vector to fold along
        Returns:
            Z_folded (n,d) ndarray - The folded and glued data
        """
        # Make the scalled inner product and the mask
        scales = (Z@n)/np.dot(n, n)
        mask = scales > 1
        
        # Make the projection and flip the points that are beyond the fold (mask)
        projected = 2 * np.outer(scales, n)
        adjustment = 2*n - projected
        return Z + mask[:,np.newaxis] * adjustment
    
    
    
    def derivative_fold(self, Z, n):
        """
        This function calculates the derivative of the fold operation
        
        Parameters:
            Z (n,d) ndarray - The data to fold
            n (d,) ndarray - The normal vector to fold along
        Returns:
            derivative (n,d,d) ndarray - The derivative of the fold operation
        """
        # Get the scaled inner product, mask, and make the identity stack
        n_normal = n / np.dot(n,n)
        scales = Z@n_normal
        mask = scales > 1
        identity_stack = np.stack([np.eye(self.width) for _ in range(len(Z))])
        
        # Calculate the first component and a helper term
        first_component = (1 - scales[:,np.newaxis, np.newaxis]) * identity_stack
        helper = 2*Z @ n_normal
        
        # Calculate the outer product of n and helper, then subtract the input
        outer_product = np.outer(helper, n_normal) - Z
        second_component = np.einsum('ij,k->ikj', outer_product, n_normal)
        
        # Return the derivative
        return 2 * mask[:,np.newaxis, np.newaxis] * (first_component + second_component)
    
    
    
    
    ############################## Training Calculations ##############################
    def forward_pass(self, D:np.ndarray):
        """
        Perform a forward pass of the data through the model

        Parameters:
            D (n,d) ndarray - The data to pass through the model
        Returns:
            output list - A list of the data at each layer of the model    
        """
        # Expand to a higher dimension if necessary
        if self.has_expand:
            Z = D @ self.input_layer.T
            output = [D, Z]
            hidden = Z
        
        # If there is no expand matrix, just use the data
        else:
            output = [D]
            hidden = D
        
        # Loop through the different layers and fold the data
        for i in range(self.layers):
            folded = self.fold(hidden, self.fold_vectors[i])
            output.append(folded)
            hidden = folded
        
        # make the final cut with the softmax
        cut = np.concatenate((hidden, np.ones((hidden.shape[0],1))), axis=1) @ self.cut_matrix.T
        # Normalize the cut and get the softmax
        cut = (cut / np.max(cut)) * self.temp * 200
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
        outer_layer = softmax - one_hot
        
        # Append ones to the forward pass and calculuate the W gradient, appending to the list
        second_stage_forward = np.concatenate((forward[-2], np.ones((forward[-2].shape[0],1))), axis=1)
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
        # show_iter = max(self.max_iter,100) // 100
        for i in range(self.max_iter):
            # Get the gradient
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
                    expand_reg = self.reg * self.expand_matrix

            # Update the weights of the cut matrix and the cut biases
            self.output_layer -= self.learning_rate * gradient[0]

            # Update the fold vectors
            for i in range(self.layers):
                self.fold_vectors[i] -= self.learning_rate * gradient[i+1]
                
            # Update the expand matrix if necessary
            if self.has_expand:
                self.expand_matrix -= self.learning_rate * gradient[-1]
            
            # Regularize the weights if it is not 0
            if self.reg != 0:
                self.cut_matrix += cut_reg
                for i in range(self.layers):
                    self.fold_vectors[i] += fold_reg[i]
                if self.has_expand:
                    self.expand_matrix += expand_reg



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
        self.b = np.random.rand(self.num_classes)

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

