import numpy as np
import copy
from tqdm import tqdm
import torch

from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.io as pio

from matplotlib import pyplot as plt




def draw_fold(hyperplane, outx, outy, color='blue', name=None):
    """
    This function draws a hyperplane on a Matplotlib plot.
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
        plane_range = np.where((plane_range > np.min(outy)) & (plane_range < np.max(outy)), plane_range, np.nan)
        plt.plot(plane_domain, plane_range, color=color, lw=2, label=name)


def idraw_fold(hyperplane, outx, outy, color='blue', name=None):
    """
    Draws a hyperplane on a Plotly plot.
    Parameters:
        hyperplane (list) - The hyperplane to draw
        outx (list) - The x values of the data
        outy (list) - The y values of the data
        color (str) - The color of the hyperplane
        name (str) - The name of the hyperplane
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


def plot_folds(model, layer_index=0, use_plotly=False):
    """
    This function plots the folds of a specific layer of the model.
    Parameters:
        X (np.ndarray) - The input data
        y (np.ndarray) - The labels
        layer_index (int) - The index of the layer to plot
        use_plotly (bool) - Whether to use Plotly for plotting
    """
    # Ensure X and y are tensors on the correct device
    # X = torch.tensor(X, dtype=torch.float32).to(self.device)
    # y = torch.tensor(y, dtype=torch.long).to(self.device)
    X = model.X
    y = model.y
    X = X.clone().detach().to(model.device) if isinstance(X, torch.Tensor) \
        else torch.tensor(X, dtype=torch.float32).to(model.device)
    y = y.clone().detach().to(model.device) if isinstance(y, torch.Tensor) \
        else torch.tensor(y, dtype=torch.long).to(model.device)


    # Forward pass to get intermediate outputs
    with torch.no_grad():
        logits, outputs = model.forward(X, return_intermediate=True)

    # Get the data after the specified layer
    Z = outputs[layer_index]
    Z = Z.detach().cpu().numpy()
    y = y.detach().cpu().numpy()

    # Get the fold vector
    hyperplane = model.fold_layers[layer_index].n.detach().cpu().numpy()

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
        fold_trace = idraw_fold(hyperplane, outx, outy, color='red', name='Fold')
        fig.add_trace(fold_trace)
        fig.update_layout(title=f'Layer {layer_index} Fold Visualization', xaxis_title='Feature 1', yaxis_title='Feature 2')
        fig.show()
    else:
        # Create a Matplotlib plot
        plt.figure(figsize=(8, 6))
        plt.scatter(outx, outy, c=y, cmap='viridis', label='Data')

        # Draw the fold (hyperplane)
        ph = ", ".join([str(round(h, 2)) for h in hyperplane])
        draw_fold(hyperplane, outx, outy, color='red', name='Fold')
        plt.title(f'Layer {layer_index} [{ph}] Fold Visualization')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.show()



def plot_wiggles(fold_histories, layer=0):
    """
    This function plots the first and second parameters of a fold layer over epochs
    Parameters:
        fold_histories (list) - The fold histories to plot
        layer (int) - The layer to plot
    """
    y_wiggles = [x[layer][1] for x in fold_histories]#[325:340]
    try:
        x_wiggles = [x[layer][0] for x in fold_histories]#[325:340]
    except:
        x_wiggles = None
    fig = make_subplots(rows=1, cols=2, subplot_titles=("X Wiggles", "Y Wiggles"))
    fig.add_trace(go.Scatter(y=y_wiggles, mode='lines', name='y_wiggles'), row=1, col=2)
    if x_wiggles is not None:
        fig.add_trace(go.Scatter(y=x_wiggles, mode='lines', name='x_wiggles'), row=1, col=1)
    fig.update_layout(width=1000, height=400, title_text=f'Fold layer {layer} over epochs')
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    pio.show(fig)





def plot_history(model, n_folds=50, include_cut=True, verbose=1):
    """
    This function plots the fold history of a model
    Parameters:
        model (OrigamiNetwork) - The model to plot
        n_folds (int) - The number of folds to plot
        include_cut (bool) - Whether to include the final decision boundary
        verbose (int) - Whether to display progress bars
    """
    resolution = 0.05
    X = model.X
    Y = model.y
    mod_number = max(1, model.epochs // n_folds)
    scalor = 1 / (30 * model.epochs / mod_number)

    # get pure colors for color scale
    length = 255
    cmap = [plt.get_cmap('spring')(i) for i in range(0, length, length//n_folds)]
    cmap = np.array([np.array(cmap[i][:-1])*length for i in range(n_folds)], dtype=int)
    colors = ['#%02x%02x%02x' % tuple(cmap[i]) for i in range(n_folds)]

    # set up grid
    if include_cut:
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        # og_output_layer = copy.deepcopy(model.output_layer)
        
    # loop over each fold
    logits, out = model.forward(X, return_intermediate=True)
    plt.figure(figsize=(8, 3.5*model.layers), dpi=150)
    n_cols = 2
    n_rows = model.layers // n_cols + 1
    
    for i in range(model.layers):
        outx = out[i][:,0]
        outy = out[i][:,1]
        # if outx is a tensor convert it to a numpy array
        if hasattr(outx, 'detach'):
            outx = outx.detach().numpy()
            outy = outy.detach().numpy()
        progress = tqdm(total=n_folds, desc="Plotting", disable=verbose==0)
        
        # plot every mod_number fold and decision boundary
        plt.subplot(n_rows, n_cols, i+1)
        for j in range(0, len(model.fold_history), mod_number):
            idx = j//mod_number
            draw_fold(model.fold_history[j][i], outx, outy, color=colors[idx], name=None)
            # if include_cut:
            #     model.output_layer = model.cut_history[j][0]
            #     # if model is a tensor convert it to a numpy array
            #     if hasattr(model.output_layer, 'detach'):
            #         model.output_layer = model.output_layer.detach().numpy()
            #     Z = model.predict(grid_points)
            #     # if Z is a tensor convert it to a numpy array
            #     if hasattr(Z, 'detach'):
            #         Z = Z.detach().numpy
            #     Z = Z.reshape(xx.shape)
            #     plt.contourf(xx, yy, Z, alpha=scalor*idx, cmap=plt.cm.YlGnBu)   
            progress.update(1)
        
        hyperplane = np.round(model.fold_history[-1], 2)
        plt.ylim(np.min(outy), np.max(outy))
        plt.xlim(np.min(outx), np.max(outx)) 
        plt.tick_params(axis='both', which='major', labelsize=6)
        plt.title(f"Layer {i+1}: {hyperplane}", fontsize=8)
        if i % n_cols == 0:
            plt.ylabel("Feature 2", fontsize=6)
        if i >= n_cols * (n_rows - 1):
            plt.xlabel("Feature 1", fontsize=6)
        progress.close()

    # reset the output layer and b
    # model.output_layer = og_output_layer.copy()

    # plot the final decision boundary
    plt.subplot(n_rows, n_cols, n_rows*n_cols)
    if include_cut:
        Z = model.predict(grid_points)
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.6, cmap=plt.cm.Greens)
    
    # plot the points
    outx = out[-2][:,0]
    outy = out[-2][:,1]
    rescale = 1.1
    miny = torch.min(outy) * rescale if torch.min(outy) > 0 else torch.min(outy) / rescale
    minx = torch.min(outx) * rescale if torch.min(outx) > 0 else torch.min(outx) / rescale
    maxy = torch.max(outy) * rescale if torch.max(outy) > 0 else torch.max(outy) / rescale
    maxx = torch.max(outx) * rescale if torch.max(outx) > 0 else torch.max(outx) / rescale
    plt.scatter(outx, outy, c=Y)
    plt.ylim(miny, maxy)
    plt.xlim(minx, maxx)
    plt.xlabel("Feature 1", fontsize=6)
    plt.title("Final Decision Boundary", fontsize=8)
    plt.tick_params(axis='both', which='major', labelsize=6)
    plt.suptitle("Pink is early, yellow is late", fontsize=9)  
    plt.tight_layout()  
    plt.show()
