import torch                        # type: ignore
import torch.nn as nn               # type: ignore
import torch.nn.functional as F     # type: ignore
import numpy as np                  # type: ignore
from models.folds import Fold, SoftFold




#################################### Dynamic Origami Model ####################################
class DynamicOrigami(nn.Module):
    def __init__(self, architecture, num_classes, no_cut=False, no_relu=False, iknowaboutthecutlayer=False):
        """
        This function initializes the Dynamic Origami model
        Parameters:
            architecture: (list) - A list of dictionaries that define the architecture of the model
            num_classes: (int) - The number of classes in the dataset
        """
        super().__init__()
        # Define the architecture
        self.architecture_example = """
            [{'type': 'Fold', 'params': {'width': (int), 'leak': (float), 'fold_in':(bool), 'has_stretch': (bool)}},
            {'type': 'SoftFold', 'params': {'width': (int), 'crease': (float), 'has_stretch': (bool)}},
            {'type': 'Linear', 'params': {'in_features': (int), 'out_features': (int)}}]
            control: [(int)]
            """
        self.architecture = architecture
        self.num_classes = num_classes
        self.layers = nn.ModuleList()
        
        # Try to create the architecture by looping through the layers if its a dictionary
        try:
            if type(self.architecture[0]) == dict:
                for layer in self.architecture:
                    layer_type = layer['type']
                    params = layer['params']
                    if layer_type == 'Fold':
                        self.layers.append(Fold(**params))
                    elif layer_type == 'SoftFold':
                        self.layers.append(SoftFold(**params))
                    elif layer_type == 'Linear':
                        self.layers.append(nn.Linear(**params))
                        if not no_relu :
                            self.layers.append(nn.ReLU())
                
                # Get the width of the penultimate layer and add a linear layer to the output
                penultimate_layer = self.architecture[-1]
                if penultimate_layer['type'] == 'Linear':
                    in_features = penultimate_layer['params']['out_features']
                    if not iknowaboutthecutlayer:
                        print("Warning: A linear 'cut' layer is already automatically added to the forward pass")
                else:
                    in_features = penultimate_layer['params']['width']

            # Handle the control case
            elif type(self.architecture[0]) == int:
                in_features = self.architecture[0]
            else:
                raise KeyError(f"Control case must have type(self.architecture[0]) = int not {type(self.architecture[0])} ({self.architecture[0]})")
            
            # Define the cut layer and append it to the layers
            if not no_cut :
                cut = nn.Linear(in_features, self.num_classes)
                self.layers.append(cut)
            
        except:
            print(f"--KeyError--\nVariable 'architecture' must be in the form of:\n{self.architecture_example}\n not {self.architecture}")
            raise KeyError
        
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        This function defines the forward pass of the model
        Parameters:
            x: (torch.Tensor) - The input tensor to the model
        Returns:
            x: (torch.Tensor) - The output tensor from the model
        """
        # flatten the input if it is not already
        if x.dim() > 2:
            x = x.view(x.shape[0], -1)
            
        # Pass the input through the layers
        for layer in self.layers:
            x = layer(x)

        return x




#################################### Testing Networks ####################################
class OrigamiToy2(nn.Module):
    def __init__(self):
        super().__init__()
        self.f1 = Fold(2)
        self.f2 = Fold(2)
        self.f3 = Fold(2)
        self.f4 = Fold(2)
        self.cut = nn.Linear(2, 2)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        This function defines the forward pass of the model
        Parameters:
            x: (torch.Tensor) - The input tensor to the model
        Returns:
            x: (torch.Tensor) - The output tensor from the model
        """
        # flatten the input if it is not already
        if x.dim() > 2:
            x = x.view(x.shape[0], -1)
            
        # Pass the input through the layers
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = self.cut(x)
        return x


class OrigamiControl0(nn.Module):
    def __init__(self):
        super().__init__()
        self.cut = nn.Linear(784, 10)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        This function defines the forward pass of the model
        Parameters:
            x: (torch.Tensor) - The input tensor to the model
        Returns:
            x: (torch.Tensor) - The output tensor from the model
        """
        # flatten the input if it is not already
        if x.dim() > 2:
            x = x.view(x.shape[0], -1)
            
        # Pass the input through the layers
        x = self.cut(x)
        return x


class OrigamiFold4(nn.Module):
    def __init__(self, input_size:int):
        super().__init__()
        self.f1 = Fold(input_size, 0.1)
        self.f2 = Fold(int(input_size*1.05), 0.1, fold_in=False)
        self.f3 = Fold(int(input_size*1.1), 0.1)
        self.f4 = Fold(int(input_size*1.1), 0.1, fold_in=False)
        self.cut = nn.Linear(int(input_size*1.1), 10)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        This function defines the forward pass of the model
        Parameters:
            x: (torch.Tensor) - The input tensor to the model
        Returns:
            x: (torch.Tensor) - The output tensor from the model
        """
        # flatten the input if it is not already
        if x.dim() > 2:
            x = x.view(x.shape[0], -1)
            
        # Pass the input through the layers
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = self.cut(x)
        return x


class OrigamiSoft4(nn.Module):
    def __init__(self, input_size:int):
        super().__init__()
        self.f1 = SoftFold(input_size, has_stretch=True)
        self.f2 = SoftFold(int(input_size*1.05), has_stretch=True)
        self.f3 = SoftFold(int(input_size*1.1), has_stretch=True)
        self.f4 = SoftFold(int(input_size*1.1), has_stretch=True)
        self.cut = nn.Linear(int(input_size*1.1), 10)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        This function defines the forward pass of the model
        Parameters:
            x: (torch.Tensor) - The input tensor to the model
        Returns:
            x: (torch.Tensor) - The output tensor from the model
        """
        # flatten the input if it is not already
        if x.dim() > 2:
            x = x.view(x.shape[0], -1)
            
        # Pass the input through the layers
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = self.cut(x)
        return x
    
    
    
    
#################################### Softmax Control ####################################
class Softmax(nn.Module):
    def __init__(self, dim, classes):
        super().__init__()
        self.f1 = nn.Linear(dim, classes)

    def forward(self, x):
        # flatten the input if it is not already
        if x.dim() > 2:
            x = x.view(x.shape[0], -1)
            
        # Pass the input through the layers
        x = self.f1(x)
        
        # Return the final output
        return x