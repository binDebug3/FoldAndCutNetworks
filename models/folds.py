import torch                    # type: ignore
import torch.nn as nn           # type: ignore
import torch.nn.functional as F # type: ignore




###################################### Fold Module ######################################


class Fold(nn.Module):
    """
    A PyTorch module that performs a folding operation on input tensors along a specified direction.
    """
    def __init__(self, width:int, leak:float=0, fold_in:bool=True, has_stretch:bool=False) -> None:
        """
        Thus function initializes the Fold module.
        Parameters:
            width (int): The expected input dimension.
            crease (float, optional): The crease parameter. If None, it will be initialized as a learnable parameter.
            has_stretch (bool): Whether the module allows stretching.
        """
        super().__init__()
        # Hyperparameters
        self.width = width
        self.leak = leak
        self.fold_in = fold_in
        self.has_stretch = has_stretch
        
        # Parameters
        min_norm = 1e-2
        n = torch.randn(self.width) * (2 / self.width) ** 0.5
        while n.norm().item() < min_norm:
            n = torch.randn(self.width) * (2 / self.width) ** 0.5
        self.n = nn.Parameter(n)
            
        # Initialize stretch as a parameter if needed
        no_stretch = torch.tensor(2.0)
        if self.has_stretch:
            self.stretch = nn.Parameter(no_stretch)
        else:
            self.register_buffer('stretch', no_stretch)
    
    
    def forward(self, input:torch.Tensor) -> torch.Tensor:
        """
        This function performs the folding operation on the input tensor.
        Parameters:
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
       



###################################### SoftFold Module ######################################


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
    def __init__(self, width:int, crease:float=None, has_stretch:bool=False) -> None:
        """
        This function initializes the SoftFold module.
        Parameters:
            width (int): The expected input dimension.
            crease (float, optional): The crease parameter. If None, it will be initialized as a learnable parameter.
            has_stretch (bool): Whether the module allows stretching.
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
        self.crease = nn.Parameter(self.crease_dist()) if crease is None else \
                        self.register_buffer('crease', torch.tensor(crease))
            
        # Initialize stretch as a parameter if needed
        self.stretch = nn.Parameter(torch.tensor(2.0)) if self.has_stretch else \
                        self.register_buffer('stretch', torch.tensor(2.0))

            
    def crease_dist(self, n_samples=1, std=0.5):
        """
        Create the crease parameter by sampling from two normal distributions
        centered at -1 and 1 with a standard deviation of 0.5.
        Parameters:
            n_samples (int): The number of samples to generate.
            std (float): The standard deviation of the normal distributions.
        Returns:
            crease (torch.Tensor): The crease parameter.
        """
        # Randomly choose which distribution to sample from (50% chance for each mode)
        mode_selector = torch.randint(0, 2, (n_samples,))
        left_mode = torch.randn(n_samples) * std - 1
        right_mode = torch.randn(n_samples) * std + 1
        return torch.where(mode_selector == 0, left_mode, right_mode)
    

    def forward(self, input_tensor:torch.Tensor) -> torch.Tensor:
        """
        This function performs the soft folding operation on the input tensor.
        Parameters:
            input_tensor (torch.Tensor): The input tensor of shape (batch_size, input_dim).
        Returns:
            output (torch.Tensor): The transformed tensor after the soft folding operation.
        """
        # pad the input if the width is greater than the input width, raise error if input width is greater than fold width
        if self.width > input_tensor.shape[1]:
            input_tensor = F.pad(input_tensor, (0, self.width - input_tensor.shape[1]))
        elif self.width < input_tensor.shape[1]:
            raise ValueError(f"Input dimension ({input_tensor.shape[1]}) is greater than fold width ({self.width})")

        # Compute z_dot_x, n_dot_n, and get scales
        eps = 1e-8  
        z_dot_x = input_tensor @ self.n     # shape: (batch_size,)
        n_dot_n = self.n @ self.n + eps     # shape: (1,)
        scales = z_dot_x / n_dot_n          # shape: (batch_size,)

        # Compute 'p' and sigmoid value (batch_size,)
        p = self.crease * (z_dot_x - n_dot_n)
        p = torch.clamp(p, min=-25.0, max=25.0)
        sigmoid = torch.sigmoid(p)          # shape: (batch_size,)

        # Get the orthogonal projection of the input onto the normal vector and compute the output
        ortho_proj = (1 - scales).unsqueeze(1) * self.n                          # shape: (batch_size, width)
        return input_tensor + self.stretch * sigmoid.unsqueeze(1) * ortho_proj   # shape: (batch_size, width)
