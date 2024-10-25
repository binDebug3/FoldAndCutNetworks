import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
import numpy as np # type: ignore
from models.folds import Fold, SoftFold




#################################### Dynamic Origami Model ####################################
class DynamicOrigami(nn.Module):
    def __init__(self, architecture, num_classes):
        super().__init__()
        # Define the architecture
        self.architecture_example = """
            [{'type': 'Fold', 'params': {'width': (int), 'leak': (float), 'fold_in':(bool), 'has_stretch': (bool)}},
            {'type': 'SoftFold', 'params': {'width': (int), 'crease': (float), 'has_stretch': (bool)}},
            {'type': 'Linear', 'params': {'in_features': (int), 'out_features': (int)}}]
            """
        self.architecture = architecture
        self.num_classes = num_classes
        self.layers = nn.ModuleList()
        
        # Try to create the architecture by looping through the layers
        try: 
            for layer in self.architecture:
                type = layer['type']
                params = layer['params']
                if type == 'Fold':
                    self.layers.append(Fold(**params))
                elif type == 'SoftFold':
                    self.layers.append(SoftFold(**params))
                elif type == 'Linear':
                    self.layers.append(nn.Linear(**params))
                    self.layers.append(nn.ReLU())
            
            # Get the width of the penultimate layer and add a linear layer to the output
            penultimate_layer = self.architecture[-1]
            if penultimate_layer['type'] == 'Linear':
                in_features = penultimate_layer['params']['out_features']
                print("Warning: A linear 'cut' layer is already automatically added to the forward pass")
            else:
                in_features = penultimate_layer['params']['width']
                
            # Define the cut layer and append it to the layers
            cut = nn.Linear(in_features, self.num_classes)
            self.layers.append(cut)
            
        except KeyError as e:
            print(f"--KeyError--\nMissing key: {e}\nVariable 'architecture' must be in the form of:\n{self.architecture_example}\n")
            raise e
        
        
    def forward(self, x):
        # flatten the input if it is not already
        if x.dim() > 2:
            x = x.view(x.shape[0], -1)
            
        # Pass the input through the layers
        for layer in self.layers:
            x = layer(x)
        
        # Return the final output
        return x




#################################### Simple Origami Net Example ####################################
class OrigamiNetExample(nn.Module):
    def __init__(self):
        super().__init__()
        self.f1 = Fold(784, 0.1)
        self.f2 = Fold(800, 0.1, fold_in=False)
        self.f3 = Fold(1000, 0.1)
        self.f4 = Fold(1000, 0.1, fold_in=False)
        self.cut = nn.Linear(1000, 10)
        
    def forward(self, x):
        # flatten the input if it is not already
        if x.dim() > 2:
            x = x.view(x.shape[0], -1)
            
        # Pass the input through the layers
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = self.cut(x)
        
        # Return the final output
        return x