import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pickle
from matplotlib import pyplot as plt
import copy
import pdb

class OrigamiNetwork():
    def __init__(self, layers = 3, width = None, max_iter=1000, tol=1e-8, learning_rate=0.01, reg=10, 
                 optimizer="grad", batch_size=32, epochs=100, leak=0):
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
        self.leak = leak

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
        raise NotImplementedError("Stochastic Gradient Descent is not implemented yet")
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
    
    
    
    def fold(self, Z, n, leaky=None):
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
        scales = (Z@n)/np.dot(n, n)
        indicator = scales > 1
        indicator = indicator.astype(int)
        indicator = indicator + (1 - indicator) * leaky
        
        # Make the projection and flip the points that are beyond the fold (mask)
        projected = np.outer(scales, n)
        folded = Z + 2 * indicator[:,np.newaxis] * (n - projected)
        
        
        # # filter n-projected to only those whose indices are 1 in indicator
        # plt.scatter(Z[:,0], Z[:,1], c=indicator)
        # plt.plot([0, n[0]], [0, n[1]], color="black")
        # plt.show()
        # fold = 2 * indicator[:,np.newaxis] * (n - projected)
        # # plot the fold with arrows from original position to new position
        # plt.figure(figsize=(12,6))
        # plt.scatter(Z[:,0], Z[:,1], c=indicator)
        # for i in range(len(Z)):
        #     plt.arrow(Z[i,0], Z[i,1], fold[i,0], fold[i,1], color="black", head_width=0.01)
        # plt.plot([0, n[0]], [0, n[1]], color="black")
        # plt.show()
 
        return folded
    
    
    
    def derivative_fold(self, Z, n, leaky=0.):
        """
        This function calculates the derivative of the fold operation
        
        Parameters:
            Z (n,d) ndarray - The data to fold
            n (d,) ndarray - The normal vector of the hyperplane
            leaky (float) - The amount of leak in the fold
        Returns:
            derivative (n,d,d) ndarray - The derivative of the fold operation
        """
        # Get the scaled inner product, mask, and make the identity stack
        quad_normal = n / np.dot(n, n)
        scales = Z @ quad_normal
        indicator = scales > 1
        indicator = indicator.astype(int)
        indicator = indicator + (1 - indicator) * leaky
        identity = np.eye(self.width)

        # Use broadcasting to apply scales along the first axis
        first_component = (1 - scales)[:, np.newaxis, np.newaxis] * identity
        
        # Calculate the outer product of n and helper, then subtract the input
        outer_product = np.outer(2 * scales, quad_normal) - Z
        second_component = np.einsum('ij,k->ikj', outer_product, quad_normal)
        
        # Return the derivative
        derivative = 2 * indicator[:,np.newaxis, np.newaxis] * (first_component + second_component)
        return derivative
    
    
    
    ############################## Training Calculations ##############################
    def forward_pass(self, D:np.ndarray):
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
    
    
    
    def back_propagation(self, indices:np.ndarray, freeze_folds:bool=False):
        """
        Perform a back propagation of the data through the model

        Parameters:
            indices (ndarray) - The indices of the data to back propagate
            freeze_folds (bool) - Whether to freeze the folds during back propogation
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
        if not freeze_folds:
            fold_grads = [self.derivative_fold(forward[i], self.fold_vectors[i]) for i in range(self.layers)]
        
            # Perform the back propogation for the folds
            backprop_start = outer_layer @ self.output_layer
            # CHECK: backprop_start = outer_layer @ self.output_layer[:,:-1]
            for i in range(self.layers):
                backprop_start = np.einsum('ij,ijk->ik', backprop_start, fold_grads[-i-1])
                gradient.append(np.sum(backprop_start, axis=0))
            
        return gradient
        
        
    
    ########################## Optimization and Training Functions ############################
    def gradient_descent(self, freeze_folds:bool=False, maxiter=None, verbose=0):
        """
        Perform gradient descent on the model
        Parameters:
            freeze_folds (bool) - Whether to freeze the folds during back propogation
            verbose (int) - Whether to show the progress of the training (default is 0)
        Returns:
            fold_history (list) - A list of the fold vectors at each iteration
        """
        # show_iter = max(self.max_iter,100) // 100
        if maxiter is None:
            maxiter = self.max_iter
        progress = tqdm(total=maxiter, position=0, leave=True, desc="Training Progress", disable=verbose==0)
        fold_history = []
        for i in range(maxiter):
            # Get the gradient
            gradient = self.back_propagation(np.arange(self.n), freeze_folds=freeze_folds)
            
            # Clip any gradients that are too large
            max_norm = 5.0
            for g in gradient:
                np.clip(g, -max_norm, max_norm, out=g)

            # Update the weights of the cut matrix and the cut biases
            self.output_layer -= self.learning_rate * gradient[0]
            self.b -= self.learning_rate * gradient[1]

            # Update the fold vectors
            if not freeze_folds:
                for i in range(self.layers):
                    self.fold_vectors[i] -= self.learning_rate * gradient[i+2]
                fold_history.append(self.fold_vectors.copy())
            progress.update()
        progress.close()
        return fold_history



    def stochastic_gradient_descent(self, re_randomize=True):
        """
        Perform stochastic gradient descent on the model

        Parameters:
            re_randomize (bool) - Whether to re-randomize the batches after each epoch
        Returns:
            None
        """
        raise NotImplementedError("Stochastic Gradient Descent is not implemented yet")
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



    def fit(self, X:np.ndarray=None, y:np.ndarray=None, X_val_set=None, y_val_set=None, freeze_folds:bool=False, maxiter=None, verbose=1):
        """
        Fit the model to the data

        Parameters:
            X (n,d) ndarray - The data to fit the model on
            y (n,) ndarray - The labels of the data
            X_val_set (n_val,d) ndarray - The validation set for the data
            y_val_set (n_val,) ndarray - The validation labels for the data
            freeze_folds (bool) - Whether to freeze the folds during back propogation
            verbose (int) - Whether to show the progress of the training (default is 1)
        Returns:
            train_history (list) - A list of training accuracies
            val_history (list) - A list of validation accuracies
            fold_history (list) - A list of the fold vectors at each iteration
        """
        if maxiter is None:
            maxiter = self.max_iter
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

        # Initialize the expand matrix if necessary
        if self.has_expand:
            self.input_layer = self.he_init((self.d, self.width))
        else:
            self.width = self.d
            
        # Initialize the cut matrix, fold vectors, and biases
        if not freeze_folds:
            self.output_layer = self.he_init((self.num_classes, self.width))
            self.fold_vectors = self.he_init((self.layers, self.width))
            self.b = np.random.rand(self.num_classes)

        # If there is a validation set, save it
        if X_val_set is not None and y_val_set is not None:
            self.X_val_set = X_val_set
            self.y_val_set = y_val_set

        # Run the optimizer
        if self.optimizer == "sgd":
            raise ValueError("Stochastic Gradient Descent is not implemented yet")
        elif self.optimizer == "grad":
            fold_history = self.gradient_descent(freeze_folds=freeze_folds, maxiter=maxiter, verbose=verbose)
        # Otherwise, raise an error
        else:
            raise ValueError("Optimizer must be 'sgd' or 'grad'")
        return fold_history
        
        

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
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    # initialization method
    def he_init(self, shape):
        # Calculates the standard deviation
        stddev = np.sqrt(2 / shape[0])
        # Initializes weights from a normal distribution with mean 0 and calculated stddev
        return np.random.normal(0, stddev, size=shape)
        
        
        
        
        
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
    
    
    
    def score_landscape(self, score_layers:int=None, X:np.ndarray=None, y:np.ndarray=None, 
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
        X = X if X is not None else self.X
        y = y if y is not None else self.y
        density = [density]*self.d if density is not None else [10]*self.d
        feature_mins = feature_mins if feature_mins is not None else np.min(X, axis=0)
        feature_maxes = feature_maxes if feature_maxes is not None else np.max(X, axis=0)
        score_layers = score_layers if type(score_layers) == list else [score_layers] if type(score_layers) == int else [l for l in range(self.layers)]
        og_fold_vectors = copy.deepcopy(self.fold_vectors)
        og_output_layer = copy.deepcopy(self.output_layer)

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
                self.fit(maxiter=10, freeze_folds=True, verbose=0)
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
                f1_name = self.feature_names[f1id] if self.feature_names is not None else f"Feature {f1id}"
                f2_name = self.feature_names[f2id] if self.feature_names is not None else f"Feature {f2id}"
                
                # get just the scores where the two features are unique in feature_combinations
                mesh = np.zeros((density[f2id], density[f1id]))
                for i, f1_val in enumerate(f1_folds):
                    for j, f2_val in enumerate(f2_folds):
                        mesh[j,i] = scores[np.where((f1 == f1_val) & (f2 == f2_val))[0][0]]
                
                offset = 1 if self.has_expand else 0
                self.fold_vectors = og_fold_vectors.copy()
                paper = self.forward_pass(X)
                outx = paper[offset + score_layer][:,f1id]
                outy = paper[offset + score_layer][:,f2id]

                # plot input to layer
                plt.figure(figsize=(12,6))
                plt.subplot(121)
                plt.scatter(outx, outy, c=y, cmap="viridis")
                pred_fold = self.fold_vectors[score_layer]
                best_fold = max_features_list[-1]
                pred_view = (round(pred_fold[f1id], 2), round(pred_fold[f2id], 2))
                best_view = (round(best_fold[f1id], 2), round(best_fold[f2id], 2))
                self.draw_fold(self.fold_vectors[score_layer], outx, outy, color="red", name=f"Predicted Fold {pred_view}")
                self.draw_fold(max_features_list[-1], outx, outy, color="black", name=f"Maximum Score Fold {best_view}")
                plt.xlabel(f1_name)
                plt.ylabel(f2_name)
                plt.title(f"Input to Layer {score_layer}")
                plt.legend()
            
                # plot the heatmap
                plt.subplot(122)
                plt.pcolormesh(f1_folds, f2_folds, mesh, cmap=theme)
                # plt.scatter(np.where(f1_folds == f1[max_index])[0], np.where(f2_folds == f2[max_index])[0], color="red", label="maximum score")
                plt.xlabel(f1_name)
                plt.ylabel(f2_name) 
                plt.title(f"Score Landscape for Layer {score_layer}")
                plt.colorbar()
                if png_path is not None:
                    try:
                        plt.savefig(png_path)
                    except Exception as e:
                        print(e)
                        print(f"The path '{png_path}' is not valid")
                plt.tight_layout()
                plt.show()
            else:
                self.fold_vectors = og_fold_vectors
                self.output_layer = og_output_layer
        
            if learning:
                # update weights with the best fold for this layer
                self.fold_vectors[score_layer] = max_features_list[-1].copy()
                # update input and output layers
                self.fit(freeze_folds=True, verbose=0, maxiter=50)
        
        if len(max_scores) == 1:
            return max_scores[0], max_features_list[0]
        return max_scores, max_features_list
        


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

        # TODO: Test this function
        # Get the predictions and return the accuracy
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
    

    def cross_val_score(self, X:np.ndarray, y:np.ndarray, cv=5):
        """
        Get the cross validated accuracy of the model on the data
        
        Parameters:
            X (n,d) ndarray - The data to score the model on
            y (n,) ndarray - The labels of the data
            cv (int) - The number of cross validation splits
        Returns:
            scores (list) - The accuracy of the model on the data for each split
        """
        #TODO: Test this function
        # Split the data and initialize the scores
        scores = []
        for train_index, test_index in train_test_split(np.arange(X.shape[0]), test_size=1/cv):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Fit the model and get the score
            self.fit(X_train, y_train)
            scores.append(self.score(X_test, y_test))
        
        # Return the scores
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
        #TODO: Test this function
        # If the data is not provided, use the training data
        if X is None:
            X = self.X
            y = self.y

        # Get the predictions and return the confusion matrix
        predictions = self.predict(X)
        return confusion_matrix(y, predictions, labels=self.classes)


    ############################## Other Functions ###############################
    def copy(self, deep=False):
        """
        Create a copy of the model

        Parameters:
            None
        Returns:
            new_model (model3 class) - A copy of the model
        """
        # Initialize a new model
        new_model = OrigamiNetwork(max_iter=self.max_iter, tol=self.tol, 
                                   learning_rate=self.learning_rate, reg=self.reg, 
                                   optimizer=self.optimizer, batch_size=self.batch_size, 
                                   epochs=self.epochs)
        
        # Copy the other attributes
        if deep:
            for param in self.__dict__:
                value = self.__dict__[param]
                if type(value) in [np.ndarray, list]:
                    setattr(new_model, param, value.copy())
                else:
                    setattr(new_model, param, value)
        return new_model
    

    def save_weights(self, file_path:str="origami_weights", save_type="standard"):
        """
        Save the weights of the model to a file so that it can be loaded later

        Parameters:
            file_path (str) - The name of the file to save the weights to
            save_type (str) - How much of the model to save
                "full" - Save the full model and all of its attributes
                "standard" - Save the standard attributes of the model
                "weights" - Save only the weights of the model
        Returns:
            None
        """
        # TODO: Test this function
        if save_type not in ["full", "standard", "weights"]:
            raise ValueError("save_type must be 'full', 'standard', or 'weights'")
        
        preferences = {"fold_vectors": self.fold_vectors,
                       "input_layer": self.input_layer,
                       "output_layer": self.output_layer,
                       "b": self.b,
                       "save_type": save_type}
        if save_type == "standard":
            standard_preferences = {
                        "max_iter": self.max_iter, 
                        "tol": self.tol, 
                        "learning_rate": self.learning_rate,
                        "optimizer": self.optimizer,
                        "batch_size": self.batch_size,
                        "epochs": self.epochs,
                        "reg": self.reg,
                        "layers": self.layers,
                        "classes": self.classes,
                        "num_classes": self.num_classes,
                        "y_dict": self.y_dict,
                        "one_hot": self.one_hot,
                        "has_expand": self.has_expand,
                        }
            preferences.update(standard_preferences)
        
        if save_type == "full":
            remaining_attributes = {"X": self.X,
                                    "y": self.y,
                                    "n": self.n,
                                    "d": self.d,
                                    "X_val_set": self.X_val_set,
                                    "y_val_set": self.y_val_set,
                                    "class_index": self.class_index,
                                    "val_history": self.val_history,
                                    "train_history": self.train_history,
                                    "weights_history": self.weights_history,
                                    }
            preferences.update(remaining_attributes)

        try:
            if "." in file_path:
                file_path = file_path.split(".")[0]
            with open(f'{file_path}.pkl', 'wb') as f:
                pickle.dump(preferences, f)
        except Exception as e:
            print(e)
            raise ValueError(f"The file '{file_path}.pkl' could not be saved.")
    

    def load_weights(self, file_path):
        """
        Load the weights of the model from a file

        Parameters:
            file_path (str) - The name of the file to load the weights from
        Returns:
            None
        """
        # TODO: Test this function
        try:
            with open(f'{file_path}.pkl', 'rb') as f:
                data = pickle.load(f)
            save_type = data["save_type"]

            self.fold_vectors = data["fold_vectors"]
            self.input_layer = data["input_layer"]
            self.output_layer = data["output_layer"]
            self.b = data["b"]
            
            if save_type == "standard" or save_type == "full":
                self.max_iter = data["max_iter"]
                self.tol = data["tol"]
                self.learning_rate = data["learning_rate"]
                self.optimizer = data["optimizer"]
                self.batch_size = data["batch_size"]
                self.epochs = data["epochs"]
                self.reg = data["reg"]
                self.n = data["n"]
                self.d = data["d"]
                self.classes = data["classes"]
                self.num_classes = data["num_classes"]
                self.y_dict = data["y_dict"]
                self.one_hot = data["one_hot"]
                self.has_expand = data["has_expand"]
            if save_type == "full":
                self.X = data["X"]
                self.y = data["y"]
                self.X_val_set = data["X_val_set"]
                self.y_val_set = data["y_val_set"]
                self.class_index = data["class_index"]
                self.val_history = data["val_history"]
                self.train_history = data["train_history"]
                self.weights_history = data["weights_history"]

        except Exception as e:
            print(e)
            raise ValueError(f"The file '{file_path}.pkl' could not be loaded")