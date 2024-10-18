import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

class OrigamiNetwork(nn.Module):
    def __init__(self, n_layers = 3, width = None, learning_rate = 0.001, reg = 10, sigmoid = False, optimizer_type = "grad", 
                 batch_size = 32, epochs = 100, leak = 0, crease = 1):
        
        super(OrigamiNetwork, self).__init__()

        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.batch_size = batch_size
        self.epochs = epochs
        self.reg = reg
        self.layers = n_layers
        self.width = width
        self.leak = leak
        self.crease = crease
        self.sigmoid = sigmoid
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        if self.sigmoid:
            self.leak =0

        # Model parameters (to be initialized later)
        self.input_layer = None
        self.fold_vectors = None
        self.output_layer = None
        self.b = None

        # Data placeholders
        self.X = None
        self.y = None
        self.classes = None
        self.num_classes = None
        self.one_hot = None
        self.fold_vectors_history = []

    def initialize_layers(self):
        if self.X is None and self.y is None:
            raise ValueError("Training data is needed before initialization")
        
        self.n, self.d = self.X.shape
        self.encode_y(self.y)

        if self.width is None:
            self.width = self.d
        
        # Initialize input layer (expand layer)
        if self.width != self.d:
            self.has_expand = True
            self.input_layer = nn.Parameter(torch.randn(self.width, self.d) * (2 / self.d) ** 0.5)
        else:
            self.has_expand = False

        # Initialize fold vectors
        self.fold_vectors = nn.ParameterList([
            nn.Parameter(torch.randn(self.width) * (2 / self.width) ** 0.5)
            for _ in range(self.layers)
        ])

        # Initialize output layer and bias
        self.output_layer = nn.Linear(self.width, self.num_classes)
    
    def encode_y(self, y):
        y = torch.tensor(y)
        self.classes = torch.unique(y)
        self.num_classes = len(self.classes)
        self.one_hot = F.one_hot(y, num_classes = self.num_classes).float()

    def fold(self, Z, n):
        if n.norm() == 0:
            n = n + 1e-8
        scales = (Z@n)/(n@n) 
        indicator = (scales > 1).float()
        indicator = indicator + (1 - indicator) * self.leak

        projected = scales.unsqueeze(1) * n
        folded = Z + 2 * indicator.unsqueeze(1) * (n - projected)

        return folded
    
    def compile_model(self):
        if self.optimizer_type == "grad":
            self.optimizer = optim.Adagrad(self.parameters(), lr = self.learning_rate)
        elif self.optimizer_type == "sgd":
            self.optimizer = optim.SGD(self.parameters(), lr = self.learning_rate)
        elif self.optimizer_type == "adam":
            self.optimizer = optim.Adam(self.parameters(), lr = self.learning_rate)
        else:
            raise ValueError("Optimizer must be 'sgd', 'grad', or 'adam'")
        
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, D, return_intermediate = False):
        outputs = []
        if self.has_expand:
            Z = D @ self.input_layer.T
            outputs.append(Z)
        else:
            Z = D
            outputs.append(Z)
        
        for fold_vector in self.fold_vectors:
            Z = self.fold(Z, fold_vector)
            outputs.append(Z)
        
        logits = self.output_layer(Z)
        if return_intermediate:
            return logits, outputs  # Return logits and intermediate outputs
        else:
            return logits
    
    def fit(self, X, y, X_val = None, y_val = None):
        self.X = torch.tensor(X, dtype = torch.float32).to(self.device)
        self.y = torch.tensor(y, dtype = torch.long).to(self.device)

        self.initialize_layers()
        self.compile_model()

        dataset = torch.utils.data.TensorDataset(self.X, self.y)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size = self.batch_size, shuffle = True)

        for epoch in range(self.epochs):
            fold_vectors_epoch = [fv.clone().detach().cpu().numpy() for fv in self.fold_vectors]
            self.fold_vectors_history.append(fold_vectors_epoch)
            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                y_hat = self.forward(batch_X)
                loss = self.loss_fn(y_hat, batch_y)
                loss.backward()
                self.optimizer.step()
            
            if X_val is not None and y_val is not None:
                self.evaluate(X_val, y_val)

    
    def evaluate(self, X_val, y_val):
        X_val = torch.tensor(X_val, dtype = torch.float32).to(self.device)
        y_val = torch.tensor(y_val, dtype = torch.long).to(self.device)

        with torch.no_grad():
            y_hat = self.forward(X_val)
            _, predicted = torch.max(y_hat, 1)
            accuracy = (predicted == y_val).float().mean()
            print(f"Validation Accuracy: {accuracy.item():.4f}")

    
    def predict(self, X):
        X = torch.tensor(X, dtype = torch.float32).to(self.device)
        with torch.no_grad():
            y_hat = self.forward(X)
            _, predicted = torch.max(y_hat, 1)
        
        return predicted.numpy()

    def draw_fold(self, hyperplane, outx, outy, color='blue', name=None):
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
            plane_range = np.where((plane_range > np.min(outy)) & (plane_range < np.max(outy)), plane_range, np.nan)
            plt.plot(plane_domain, plane_range, color=color, lw=2, label=name)
    

    def idraw_fold(self, hyperplane, outx, outy, color='blue', name=None):
        """
        Draws a hyperplane on a Plotly plot.
        """
        plane_domain = np.linspace(np.min(outx), np.max(outx), 100)
        if hyperplane[1] == 0:
            return go.Scatter(
                x=[hyperplane[0], hyperplane[0]], y=[np.min(outy), np.max(outy)],
                mode="lines", line=dict(color=color, width=2), name=name
            )
        elif hyperplane[0] == 0:
            return go.Scatter(
                x=[np.min(outx), np.max(outx)], y=[hyperplane[1], hyperplane[1]],
                mode="lines", line=dict(color=color, width=2), name=name
            )
        else:
            a, b = hyperplane
            slope = -a / b
            intercept = b - slope * a
            plane_range = slope * plane_domain + intercept
            # Keep values inside y range
            plane_range = np.where((plane_range > np.min(outy)) & (plane_range < np.max(outy)), plane_range, np.nan)
            return go.Scatter(
                x=plane_domain, y=plane_range, mode="lines",
                line=dict(color=color, width=2), name=name
            )


    def plot_folds(self, X, y, layer_index=0, use_plotly=False):
        # Ensure X and y are tensors on the correct device
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.long).to(self.device)

        # Forward pass to get intermediate outputs
        with torch.no_grad():
            logits, outputs = self.forward(X, return_intermediate=True)

        # Get the data after the specified layer
        Z = outputs[layer_index]
        Z = Z.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        # Get the fold vector
        hyperplane = self.fold_vectors[layer_index].detach().cpu().numpy()

        # Extract the two dimensions to plot
        if Z.shape[1] >= 2:
            outx = Z[:, 0]
            outy = Z[:, 1]
        else:
            raise ValueError("Data has less than 2 dimensions after folding.")

        if use_plotly:
            # Create a Plotly figure
            fig = go.Figure()
            # Add data points
            fig.add_trace(go.Scatter(x=outx, y=outy, mode='markers', marker=dict(color=y), name='Data'))
            # Add the fold (hyperplane)
            fold_trace = self.idraw_fold(hyperplane, outx, outy, color='red', name='Fold')
            fig.add_trace(fold_trace)
            fig.update_layout(title=f'Layer {layer_index} Fold Visualization', xaxis_title='Feature 1', yaxis_title='Feature 2')
            fig.show()
        else:
            import matplotlib.pyplot as plt
            # Create a Matplotlib plot
            plt.figure(figsize=(8, 6))
            plt.scatter(outx, outy, c=y, cmap='viridis', label='Data')
            # Draw the fold (hyperplane)
            self.draw_fold(hyperplane, outx, outy, color='red', name='Fold')
            plt.title(f'Layer {layer_index} Fold Visualization')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.legend()
            plt.show()





        