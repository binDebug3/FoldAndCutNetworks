import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import torch.nn.functional as F # type: ignore
import numpy as np # type: ignore
import plotly.graph_objects as go # type: ignore
import plotly.express as px # type: ignore
from plotly.subplots import make_subplots # type: ignore
import matplotlib.pyplot as plt # type: ignore
from tqdm import tqdm # type: ignore
import warnings
import os
import sys



class Fold(nn.Module):
    """
    A PyTorch module that performs a folding operation on input tensors along a specified direction.
    """
    def __init__(self, width: int, leak: float = 0, fold_in: bool = True, has_stretch: bool = False):
        """
        Args:
            width (int): The expected input dimension.
            crease (float, optional): The crease parameter. If None, it will be initialized as a learnable parameter.
        """
        super().__init__()
        # Hyperparameters
        self.width = width
        self.leak = leak
        self.fold_in = fold_in
        self.has_stretch = has_stretch
        
        # Parameters
        n = torch.randn(self.width) * (2 / self.width) ** 0.5
        min_norm = 1e-2
        while n.norm().item() < min_norm:
            n = torch.randn(self.width) * (2 / self.width) ** 0.5
        self.n = nn.Parameter(n)
            
        # Initialize stretch as a parameter if needed
        if self.has_stretch:
            self.stretch = nn.Parameter(torch.tensor(2.0))
        else:
            self.register_buffer('stretch', torch.tensor(2.0))
            
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): The input tensor of shape (batch_size, input_dim).
        Returns:
            folded (torch.Tensor): The transformed tensor after the folding operation.
        """
        # pad the input if the width is greater than the input width, raise error if input width is greater than fold width
        if self.width > input.shape[1]:
            input = F.pad(input, (0, self.width - input.shape[1]))
        elif self.width < input.shape[1]:
            raise ValueError(f"Input dimension ({input.shape[1]}) is greater than fold width ({self.width})")

        # Compute scales
        eps = 1e-8
        scales = (input @ self.n) / (self.n @ self.n + eps)
        
        # If it is a fold in, we want to fold in the values that are greater than 1
        if self.fold_in:
            indicator = (scales > 1).float()
        else:
            indicator = (scales < 1).float()
        indicator = indicator + (1 - indicator) * self.leak

        # Compute the projected and folded values
        projection = scales.unsqueeze(1) * self.n
        folded = input + self.stretch * indicator.unsqueeze(1) * (self.n - projection)
        return folded
       

class SoftFold(nn.Module):
    """
    Sigmoid Fold module.

    This module performs a soft fold of the input data along the hyperplane defined by the normal vector n.
    It uses a sigmoid function to smoothly transition the folding effect.

    Parameters:
        width (int): The dimensionality of the input data.
        crease (float or None): A scaling factor for the sigmoid function. If None, it is set as a learnable parameter.
        has_stretch (bool): Whether the module allows stretching.

    Attributes:
        n (nn.Parameter): The normal vector of the hyperplane (learnable parameter).
        crease (nn.Parameter or float): The sigmoid scaling factor (learnable or fixed).
        has_stretch (bool): Whether the module allows stretching.
    """
    def __init__(self, width: int, crease: float = None, has_stretch: bool = False):
        """
        Args:
            width (int): The expected input dimension.
            crease (float, optional): The crease parameter. If None, it will be initialized as a learnable parameter.
        """
        super().__init__()
        # Hyperparameters
        self.width = width
        self.has_stretch = has_stretch
        
        # Parameters
        n = torch.randn(self.width) * (2 / self.width) ** 0.5
        min_norm = 1e-2
        while n.norm().item() < min_norm:
            n = torch.randn(self.width) * (2 / self.width) ** 0.5
        self.n = nn.Parameter(n)

        # Initialize crease parameter
        if crease is None:
            self.crease = nn.Parameter(self.crease_dist())
        else:
            self.register_buffer('crease', torch.tensor(crease))
            
        # Initialize stretch as a parameter if needed
        if self.has_stretch:
            self.stretch = nn.Parameter(torch.tensor(2.0))
        else:
            self.register_buffer('stretch', torch.tensor(2.0))

            
    def crease_dist(self, n_samples=1, std=0.5):
        # Randomly choose which distribution to sample from (50% chance for each mode)
        mode_selector = torch.randint(0, 2, (n_samples,))
        left_mode = torch.randn(n_samples) * std - 1
        right_mode = torch.randn(n_samples) * std + 1
        return torch.where(mode_selector == 0, left_mode, right_mode)
    

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): The input tensor of shape (batch_size, input_dim).

        Returns:
            output (torch.Tensor): The transformed tensor after the soft folding operation.
        """
        # pad the input if the width is greater than the input width, raise error if input width is greater than fold width
        if self.width > input.shape[1]:
            input = F.pad(input, (0, self.width - input.shape[1]))
        elif self.width < input.shape[1]:
            raise ValueError(f"Input dimension ({input.shape[1]}) is greater than fold width ({self.width})")

        # Small epsilon for numerical stability
        eps = 1e-8  

        # Compute z_dot_x, n_dot_n, and get scales
        z_dot_x = input @ self.n  # shape: (batch_size,)
        n_dot_n = self.n @ self.n + eps  # shape: (1,)
        scales = z_dot_x / n_dot_n  # shape: (batch_size,)

        # Compute 'p' and sigmoid value (batch_size,)
        p = self.crease * (z_dot_x - n_dot_n)
        p = torch.clamp(p, min=-25.0, max=25.0)
        sigmoid = torch.sigmoid(p)  # shape: (batch_size,)

        # Get the orthogonal projection of the input onto the normal vector and compute the output
        ortho_proj = (1 - scales).unsqueeze(1) * self.n  # shape: (batch_size, width)
        output = input + self.stretch * sigmoid.unsqueeze(1) * ortho_proj  # shape: (batch_size, width)
        return output


class NoamScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, model_size, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.model_size = model_size
        super(NoamScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step_num = self.last_epoch + 1
        lr = self.optimizer.defaults['lr'] * (self.model_size ** (-0.5)) * min(step_num ** (-0.5), step_num * self.warmup_steps ** (-1.5))
        return [lr for _ in self.base_lrs]



class OrigamiNetwork(nn.Module):
    def __init__(self, n_layers:int=3, width:int=None, learning_rate:float=0.001, reg:int=10, optimizer_type:str="grad", lr_schedule:bool=False,
                 batch_size:int=32, epochs:int=100, leak:float=0, crease:float=0, verbose=1, regularization:float=0):
        
        super(OrigamiNetwork, self).__init__()

        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.lr_schedule = lr_schedule
        self.batch_size = batch_size
        self.epochs = epochs
        self.reg = reg
        self.layers = n_layers
        self.width = width
        self.leak = leak
        self.crease = crease
        self.verbose = verbose
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        # Model parameters (to be initialized later)
        self.input_layer = None
        self.fold_layers = None
        self.output_layer = None

        # Data placeholders
        self.X = None
        self.y = None
        self.classes = None
        self.num_classes = None
        self.one_hot = None
        self.val_history = []
        self.train_history = []
        self.fold_history = []
        self.learning_rates = []
        self.cut_history = []
        self.crease_history = []
        self.regularization = regularization
        
        if self.verbose > 1 and self.reg != 0:
            warnings.warn("Regularization is not implemented yet.")


    def initialize_layers(self) -> None:
        """
        Initializes the fold layers of the model.
        """
        if self.X is None and self.y is None:
            raise ValueError("Training data is needed before initialization")
        
        self.n, self.d = self.X.shape
        self.encode_y(self.y)

        if self.width is None:
            self.width = self.d
        
        # Initialize input layer (expand layer)
        if self.width != self.d:
            self.has_expand = True
            # self.input_layer = nn.Parameter(torch.randn(self.width, self.d) * (2 / self.d) ** 0.5)
            self.input_layer = nn.Linear(self.d, self.width)
        else:
            self.has_expand = False

        # Initialize fold vectors and cut layer
        if self.crease == 0:
            self.fold_layers = nn.ModuleList([Fold(self.width, self.leak) for _ in range(self.layers)])
        else:
            self.fold_layers = nn.ModuleList([SoftFold(self.width, self.crease) for _ in range(self.layers)])
            if self.verbose > 1 and self.leak != 0:
                warnings.warn("Leaky folds are ignored when crease is nonzero.")
        self.output_layer = nn.Linear(self.width, self.num_classes)
    
    
    def encode_y(self, y) -> None:
        """
        Encodes the labels into one-hot format.
        Parameters:
            y (np.ndarray) - The labels
        """
        # if type(y) == np.ndarray:
        #     y = torch.tensor(y)
        y = y.clone().detach().to(self.device, dtype=torch.long) if isinstance(y, torch.Tensor) \
            else torch.tensor(y, dtype=torch.long).to(self.device)
        self.classes = torch.unique(y)
        self.num_classes = len(self.classes)
        self.one_hot = F.one_hot(y, num_classes = self.num_classes).float()
    
    
    def compile_model(self) -> None:
        """
        Compiles the model by initializing the optimizer and loss function.
        """
        if self.optimizer_type == "grad":
            self.optimizer = optim.Adagrad(self.parameters(), lr = self.learning_rate)
        elif self.optimizer_type == "sgd":
            self.optimizer = optim.SGD(self.parameters(), lr = self.learning_rate)
        elif self.optimizer_type == "adam":
            if self.regularization != 0:
                self.optimizer = optim.Adam(self.parameters(), lr = self.learning_rate, weight_decay = self.regularization)
            else:
                self.optimizer = optim.Adam(self.parameters(), lr = self.learning_rate)
        else:
            raise ValueError("Optimizer must be 'sgd', 'grad', or 'adam'")
        
        if self.lr_schedule:
            self.schedule = NoamScheduler(self.optimizer, 200, self.width)
        self.loss_fn = nn.CrossEntropyLoss()


    def forward(self, D, return_intermediate:bool=False) -> torch.Tensor:
        """
        Performs a forward pass of the model.
        Parameters:
            D (torch.Tensor) - The input data
            return_intermediate (bool) - Whether to return intermediate outputs
        Returns:
            logits (torch.Tensor) - The output of the model
            outputs (list) - (optional) The intermediate outputs of the model
        """
        outputs = []
        if self.has_expand:
            Z = D @ self.input_layer.T
        else:
            Z = D
        outputs.append(Z)
        
        for fold_vector in self.fold_layers:
            Z = fold_vector.forward(Z)
            outputs.append(Z)
        
        logits = self.output_layer(Z)
        if return_intermediate:
            return logits, outputs  # Return logits and intermediate outputs
        else:
            return logits
    
    
    def load_data(self, X, y, freeze_folds:bool=False, freeze_cut:bool=False) -> None:
        """
        This function loads the data into the model and initializes the data loader.
        Parameters:
            X (np.ndarray) - The input data
            y (np.ndarray) - The labels
            freeze_folds (bool) - Whether to freeze the fold layers during training
            freeze_cut (bool) - Whether to freeze the cut layer during training
        """
        self.X = X.clone().detach().to(self.device) if isinstance(X, torch.Tensor) \
            else torch.tensor(X, dtype=torch.float32).to(self.device)
        self.y = y.clone().detach().to(self.device) if isinstance(y, torch.Tensor) \
            else torch.tensor(y, dtype=torch.long).to(self.device)
        
        self.initialize_layers()
        self.compile_model()
        
        # TEST
        if freeze_folds:
            for fold_layer in self.fold_layers:
                for param in fold_layer.parameters():
                    param.requires_grad = False
        if freeze_cut:
            for param in self.output_layer.parameters():
                param.requires_grad = False

        dataset = torch.utils.data.TensorDataset(self.X, self.y)
        self.data_loader = torch.utils.data.DataLoader(dataset, batch_size = self.batch_size, shuffle = True)
    
    
    def fit(self, X=None, y=None, X_val=None, y_val=None, validate:bool=True, history:bool=True) -> list:
        """
        Trains the model on the input data.
        Parameters:
            X (np.ndarray) - The input data
            y (np.ndarray) - The labels
            X_val (np.ndarray) - The validation input data
            y_val (np.ndarray) - The validation labels
            validate (bool) - Whether to validate the model during training
            history (bool) - Whether to save the performance and parameter history
            verbose (int) - The verbosity level of the training
        Returns:
            history (list) - The training history of the model
        """
        # check if data_loader is defined
        if not hasattr(self, "data_loader"):
            if X is not None and y is not None:
                self.load_data(X, y)
            else:
                raise ValueError("Data loader is not defined. Please load data first by calling 'load_data'.")

        val_update_wait = max(1, self.epochs // 50)
        progress = tqdm(total=self.epochs, desc="Training", disable=self.verbose==0)
        for epoch in range(self.epochs):
            history and self.update_history()
            for batch_X, batch_y in self.data_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device, dtype=torch.long)
                self.optimizer.zero_grad()
                y_hat = self.forward(batch_X)
                loss = self.loss_fn(y_hat, batch_y)
                loss.backward()
                self.optimizer.step()
                
                if self.lr_schedule:
                    self.schedule.step()
                lr = self.optimizer.param_groups[0]['lr']
                self.learning_rates.append(lr)
            if validate and epoch % val_update_wait == 0 and X_val is not None and y_val is not None:
                acc = self.evaluate(X_val, y_val)
                history and self.val_history.append(acc)
                progress.set_description(f"Val Accuracy: {round(acc, 4)}")                
            progress.update(1)
        progress.close()
        return self.get_history()

    
    def evaluate(self, X_val, y_val) -> float:
        """
        Evaluates the model on the validation data during training
        Parameters:
            X_val (np.ndarray) - The input data
            y_val (np.ndarray) - The labels
        Returns:
            accuracy (float) - The accuracy of the model on the validation data
        """
        X_val = torch.tensor(X_val, dtype = torch.float32).to(self.device)
        y_val = torch.tensor(y_val, dtype = torch.long).to(self.device)

        with torch.no_grad():
            y_hat = self.forward(X_val)
            _, predicted = torch.max(y_hat, 1)
            accuracy = (predicted == y_val).float().mean()
            return accuracy.item()

    
    def predict(self, X) -> np.ndarray:
        """
        Predict the labels of the input data
        Parameters:
            X (np.ndarray) - The input data
        Returns:
            predicted (np.ndarray) - The predicted labels
        """
        X = torch.tensor(X, dtype=torch.float32).to(self.device) if not isinstance(X, torch.Tensor) \
                else X.clone().detach().to(self.device)
        with torch.no_grad():
            y_hat = self.forward(X)
            _, predicted = torch.max(y_hat, 1)
        return predicted.numpy()
    
    
    def score(self, X, y) -> float:
        """
        Score the model on the input data
        Parameters:
            X (np.ndarray) - The input data
            y (np.ndarray) - The labels
        Returns:
            accuracy (float) - The accuracy of the model on the input data
        """
        y_pred = self.predict(X)
        # convert y_pred to numpy array
        y = y.numpy() if isinstance(y, torch.Tensor) else y
        accuracy = (y_pred == y).mean()
        return accuracy
    
    
    def get_loss(self, X, y) -> float:
        """
        Compute the loss of the model on the input data
        Parameters:
            X (np.ndarray or torch.Tensor) - The input data
            y (np.ndarray or torch.Tensor) - The labels
        Returns:
            loss (float) - The loss of the model on the input data
        """
        # Convert to tensor only if not already a tensor
        X = torch.tensor(X, dtype=torch.float32).to(self.device) if not isinstance(X, torch.Tensor) \
                else X.clone().detach().to(self.device)
        y = torch.tensor(y, dtype=torch.long).to(self.device) if not isinstance(y, torch.Tensor) \
                else y.clone().detach().long().to(self.device)
        
        with torch.no_grad():
            y_hat = self.forward(X)
            loss = self.loss_fn(y_hat, y)
        return loss
    
    
    def update_history(self) -> None:
        """
        This function updates the history of the model parameters over each epoch
        """
        self.fold_history.append([self.to_numpy(fv.n) for fv in self.fold_layers])
        self.cut_history.append(self.to_numpy(self.output_layer.weight))
        if self.crease != 0:
            self.crease_history.append([self.to_numpy(fv.crease) for fv in self.fold_layers])
        
    
    
    def get_history(self, history:str=None) -> list:
        """
        Get the history of the model
        Parameters:
            history (str) - The history to get
        Returns:
            history (list) - The history of the model
        """
        libary = ["fold", "cut", "crease", "train", "val"]
        if history is None:
            return self.fold_history, self.cut_history, self.crease_history, self.train_history, self.val_history
        elif history.lower() in libary:
            return getattr(self, f"{history}_history")
        
    
    def set_folds(self, fold_vectors:list) -> None:
        """
        Set the fold vectors of the model
        Parameters:
            fold_vectors (list(np.ndarray / torch.tensor)) - nxd (n_layer rows by dimension columns) The fold vectors to set
        """
        assert self.layers == len(fold_vectors), f"Number of fold vectors must match the number of layers ({len(fold_vectors)} != {self.layers})"
        # fix typing
        if type(fold_vectors[0]) == list:
            fold_vectors = [np.array(fv) for fv in fold_vectors]
        if type(fold_vectors[0]) == np.ndarray:
            fold_vectors = [torch.tensor(fv, dtype=torch.float32) for fv in fold_vectors]
        # set the fold vectors
        for fold_layer, fold_vector in zip(self.fold_layers, fold_vectors):
            fold_layer.n = nn.Parameter(fold_vector)
    
    
    def set_cut(self, cut_vector:np.ndarray) -> None:
        """
        Set the cut vector of the model
        Parameters:
            cut_vector (np.ndarray) - The cut vector to set
        """
        assert cut_vector.shape == (self.width, self.num_classes), f"Cut vector must be of shape ({self.width}, {self.num_classes})"
        # fix typing
        if type(cut_vector) == list:
            cut_vector = np.array(cut_vector)
        if type(cut_vector) == np.ndarray:
            cut_vector = torch.tensor(cut_vector, dtype=torch.float32)
        # set the cut vector
        self.output_layer.weight = nn.Parameter(cut_vector)


    def set_params(self, **kwargs) -> None:
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
    
    
    def to_numpy(self, tensor:torch.Tensor) -> np.ndarray:
        """
        Convert a tensor to a numpy array
        Parameters:
            tensor (torch.Tensor) - The tensor to convert
        Returns:
            np_array (np.ndarray) - The numpy array
        """
        return tensor.clone().detach().cpu().numpy()
    
    
    def get_fold_vectors(self, precision:int=3) -> dict:
        """
        Get the fold vectors of the model
        Parameters:
            precision (int) - The precision of the fold vector parameters
        Returns:
            fold_vectors (list) - The fold vectors of the model
        """
        output = {}
        for i, fv in enumerate(self.fold_layers):
            output[f"Layer {i}"] = {"hyperplane": [round(val, precision) for val in self.to_numpy(fv.n).tolist()]}
            if self.crease != 0:
                output[f"Layer {i}"]["crease"] = [round(val, precision) for val in self.to_numpy(fv.crease).tolist()]
        return output
    
    
    def get_cut_vector(self) -> np.ndarray:
        """
        Get the cut vector of the model
        Returns:
            cut_vector (np.ndarray) - The cut vector of the model
        """
        return self.to_numpy(self.output_layer.weight)
    

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
            if type(val) == np.ndarray or type(val) == list or type(val) == torch.Tensor:
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
    
    
    def copy(self, deep:bool=False) -> 'OrigamiNetwork':
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

        
    def save_weights(self, path:str="model_weights.pth") -> None:
        """
        Save the weights of the model
        Parameters:
            path (str) - The path to save the weights
        """
        assert type(path) == str, "Path must be a string"
        assert path[-4:] == ".pth", "Path must end in '.pth'"
        assert os.path.exists(os.path.dirname(path)), "Directory does not exist"
        torch.save(self.state_dict(), path)


    def load_weights(self, path:str="model_weights.pth") -> None:
        """
        Load the weights of the model
        Parameters:
            path (str) - The path to load the weights from
        """
        assert type(path) == str, "Path must be a string"
        assert os.path.exists(path), "File does not exist"
        self.load_state_dict(torch.load(path))

        