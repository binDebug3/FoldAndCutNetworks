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
    def __init__(self, layers = 3, width = None, max_iter=1000, tol=1e-8, learning_rate=0.01, reg=10, optimizer="grad", batch_size=32, epochs=100):
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
        self.b = None
        
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
        scales = Z @ n_normal
        mask = scales > 1
        identity = np.eye(self.width)

        # Use broadcasting to apply scales along the first axis
        first_component = (1 - scales)[:, np.newaxis, np.newaxis] * identity
        
        # Calculate the outer product of n and helper, then subtract the input
        outer_product = np.outer(2 * scales, n_normal) - Z
        second_component = np.einsum('ij,k->ikj', outer_product, n_normal)
        
        # Return the derivative
        return 2 * mask[:,np.newaxis, np.newaxis] * (first_component + second_component)
    
    
    ############################## Training Calculations ##############################
    def forward_pass(self, D:np.ndarray):
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
        
        # make the final cut with the softmax
        cut = input @ self.output_layer.T + self.b[np.newaxis,:]
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
        
        # Make the b and W gradient and append them to the gradient
        dW = np.einsum('ik,id->kd', outer_layer, forward[-2])
        db = np.sum(outer_layer, axis=0)
        gradient.append(dW)
        gradient.append(db)
        
        # Calculate the gradients of each fold using the forward propogation
        fold_grads = [self.derivative_fold(forward[i], self.fold_vectors[i]) for i in range(self.layers)]
        
        # Perform the back propogation for the folds
        backprop_start = outer_layer @ self.output_layer
        # CHECK: backprop_start = outer_layer @ self.output_layer[:,:-1]
        for i in range(self.layers):
            backprop_start = np.einsum('ij,ijk->ik', backprop_start, fold_grads[-i-1])
            gradient.append(np.sum(backprop_start, axis=0))
            
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



    def stochastic_gradient_descent(self, re_randomize=True):
        """
        Perform stochastic gradient descent on the model

        Parameters:
            re_randomize (bool) - Whether to re-randomize the batches after each epoch
        Returns:
            None
        """
        
        # Raise an error if there are no epochs or batch size, or if batch size is greater than the number of points
        if self.batch_size is None or self.epochs is None:
            raise ValueError("Batch size or epochs must be specified")
        if self.batch_size > self.n:
            raise ValueError("Batch size must be less than the number of points")
        
        # Initialize the loop, get the batches, and go through the epochs
        batches = self.randomize_batches()
        loop = tqdm(total=self.epochs*len(batches), position=0)
        self.update_differences(self.X, batches)
        for epoch in range(self.epochs):

            # reset the batches if re_randomize is true
            if re_randomize and epoch > 0:
                batches = self.randomize_batches()
                self.update_differences(self.X, batches)
            
            # Loop through the different batches
            loss_list = []
            self.weights_history.append(self.weights.copy())
            for i, batch in enumerate(batches):

                # Get the gradient, update weights, and append the loss
                gradient = self.gradient(self.weights, subset = batch, subset_num = i)
                self.weights -= self.learning_rate * gradient
                loss_list.append(self.loss(self.weights, subset = batch, subset_num = i))

                # update our loop
                loop.set_description('epoch:{}, loss:{:.4f}'.format(epoch,loss_list[-1]))
                loop.update()

            # If there is a validation set, check the validation error
            if self.X_val_set is not None and self.y_val_set is not None:
                
                # Predict on the validation set and append the history
                val_predictions = self.predict(self.X_val_set)
                val_accuracy = accuracy_score(self.y_val_set, val_predictions)
                self.val_history.append(val_accuracy)

                # Predict on the training set and append the history
                train_predictions = self.predict(self.X)
                train_accuracy = accuracy_score(self.y, train_predictions)
                self.train_history.append(train_accuracy)
                
                # Show the progress
                # print(f"({epoch}) Val Accuracy: {np.round(val_accuracy,5)}.   Train Accuracy: {train_accuracy}")

            # Append the history of the weights
            self.weights_history.append(self.weights.copy())
        loop.close()


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