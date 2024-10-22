import numpy as np
import copy
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn

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
    # X = torch.tensor(X, dtype=torch.float32).to(model.device)
    # y = torch.tensor(y, dtype=torch.long).to(model.device)
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
    if layer_index != model.layers:
        hyperplane = model.fold_layers[layer_index].n.detach().cpu().numpy()

    # Extract the two dimensions to plot
    if Z.shape[1] >= 2:
        outx = Z[:, 0]
        outy = Z[:, 1]
    else:
        raise ValueError("Data has less than 2 dimensions after folding.")

    if use_plotly:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=outx, y=outy, mode='markers', marker=dict(color=y), name='Data'))
        if layer_index != model.layers:
            fold_trace = idraw_fold(hyperplane, outx, outy, color='red', name='Fold')
        fig.add_trace(fold_trace)
        fig.update_layout(title=f'Layer {layer_index} Fold Visualization', xaxis_title='Feature 1', yaxis_title='Feature 2')
        fig.show()
    else:
        # Create a Matplotlib plot
        plt.figure(figsize=(8, 6))
        plt.scatter(outx, outy, c=y, cmap='viridis', label='Data')

        # Draw the fold (hyperplane)
        if layer_index != model.layers:
            ph = ", ".join([str(round(h, 2)) for h in hyperplane])
            draw_fold(hyperplane, outx, outy, color='red', name='Fold')
            plt.title(f'Layer {layer_index} [{ph}] Fold Visualization')
        else:
            plt.title("Output Before Softmax")
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.show()



def plot_wiggles(fold_histories, crease_histories=[], layer=0):
    """
    This function plots the first and second parameters of a fold layer over epochs
    Parameters:
        fold_histories (list) - The fold histories to plot
        layer (int) - The layer to plot
    """
    x_wiggles = [x[layer][0] for x in fold_histories]
    y_wiggles = [x[layer][1] for x in fold_histories]
    if len(crease_histories) > 0:
        crease_wiggles = [x[layer][0] for x in crease_histories]
        cc = 3
    else:
        cc = 2
    fig = make_subplots(rows=1, cols=cc, subplot_titles=("X Wiggles", "Y Wiggles"))
    fig.add_trace(go.Scatter(y=x_wiggles, mode='lines', name='x_wiggles'), row=1, col=1)
    fig.add_trace(go.Scatter(y=y_wiggles, mode='lines', name='y_wiggles'), row=1, col=2)
    if cc == 3:
        fig.add_trace(go.Scatter(y=crease_wiggles, mode='lines', name='crease_wiggles'), row=1, col=3)
    fig.update_layout(width=1000, height=400, title_text=f'Fold layer {layer} over epochs')
    for i in range(1, cc+1):
        fig.update_xaxes(title_text="Epoch", row=1, col=i)
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
        
        hyperplane = np.round(model.fold_history[-1][i], 2)
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
    outx = out[-2][:,0].detach().numpy()
    outy = out[-2][:,1].detach().numpy()
    rescale = 1.1
    miny = np.min(outy) * rescale if np.min(outy) > 0 else np.min(outy) / rescale
    minx = np.min(outx) * rescale if np.min(outx) > 0 else np.min(outx) / rescale
    maxy = np.max(outy) * rescale if np.max(outy) > 0 else np.max(outy) / rescale
    maxx = np.max(outx) * rescale if np.max(outx) > 0 else np.max(outx) / rescale
    # plt.scatter(outx, outy, c=Y)
    plt.scatter(X[:,0].detach().numpy(), X[:,1].detach().numpy(), c=Y.detach().numpy(), s=0.4)
    plt.ylim(miny, maxy)
    plt.xlim(minx, maxx)
    plt.xlabel("Feature 1", fontsize=6)
    plt.title("Final Decision Boundary", fontsize=8)
    plt.tick_params(axis='both', which='major', labelsize=6)
    plt.suptitle("Pink is early, yellow is late", fontsize=9)  
    plt.tight_layout()  
    plt.show()





def iscore_landscape(model, score_layers:int=None, feature_mins: torch.Tensor = None, feature_maxes: torch.Tensor = None, density: int = 10, 
                     f1id: int = 0, f2id: int = 1, create_plot: bool = False, png_path: str = None, theme: str = "viridis",
                     verbose: int = 0):
    """
    Visualizes the score landscape of the model for a given layer and two features.
    
    Parameters:
        model (OrigamiNetwork) - The model to visualize
        score_layers (int or list) - The layer(s) to calculate the score for
        feature_mins (torch.Tensor) - The minimum values for each feature
        feature_maxes (torch.Tensor) - The maximum values for each feature
        density (int) - The number of points to calculate the score for
        f1id (int) - The id of the first feature to calculate the score for
        f2id (int) - The id of the second feature to calculate the score for
        create_plot (bool) - Whether to create a plot of the score landscape
        png_path (str) - The path to save the plot to
        theme (str) - The theme of the plot
        verbose (int) - Whether to show the progress of the training (default is 1)
    Returns:
        max_score (float) - The maximum score of the model
        max_features (list) - The features that produced the maximum score
    """
    # Set default values
    og_model = copy.deepcopy(model)
    X = model.X
    y = model.y
    density = [density] * X.shape[1] if density is not None else [10] * X.shape[1]
    feature_mins = feature_mins if feature_mins is not None else torch.min(X, dim=0).values
    feature_maxes = feature_maxes if feature_maxes is not None else torch.max(X, dim=0).values
    score_layers = score_layers if isinstance(score_layers, list) else [score_layers] if isinstance(score_layers, int) else [l for l in range(model.layers)]

    # Input error handling
    assert isinstance(X, torch.Tensor) and X.ndim == 2, f"X must be a 2D PyTorch tensor. Instead got {type(X)}"
    assert isinstance(y, torch.Tensor), f"y must be a PyTorch tensor. Instead got {type(y)}"
    assert isinstance(score_layers, list) and len(score_layers) > 0 and isinstance(score_layers[0], int), f"score_layers must be a list of integers. Instead got {score_layers}"
    
    # Create a grid of features (use torch.linspace instead of np.linspace)
    feature_folds = []
    for mins, maxes, d in zip(feature_mins, feature_maxes, density):
        feature_folds.append(torch.linspace(mins.item(), maxes.item(), d))

    # Use torch.meshgrid to get feature combinations
    feature_combinations = torch.cartesian_prod(*feature_folds)

    # Compute scores for each feature combination and each layer
    max_scores = []
    max_features_list = []

    for score_layer in score_layers:
        scores = []
        for features in tqdm(feature_combinations, position=0, leave=True, disable=verbose==0, desc=f"score Layer {score_layer}"):
            model.fold_layers[score_layer].n = nn.Parameter(features.clone().detach().to(model.device))
            model.output_layer.load_state_dict(og_model.output_layer.state_dict())
            scores.append(model.score(X, y).item())
        
        # Find the maximum score and the features that produced it
        scores = torch.tensor(scores)
        max_score = torch.max(scores).item()
        max_index = torch.argmax(scores).item()
        max_scores.append(max_score)
        max_features_list.append(feature_combinations[max_index])

        # Create a heatmap of the score landscape for features f1id and f2id
        if create_plot:
            f1 = feature_combinations[:, f1id].cpu().numpy()
            f2 = feature_combinations[:, f2id].cpu().numpy()
            f1_folds = feature_folds[f1id].cpu().numpy()
            f2_folds = feature_folds[f2id].cpu().numpy()

            # Get the heatmap data
            mesh = np.zeros((len(f2_folds), len(f1_folds)))
            for i, f1_val in enumerate(f1_folds):
                for j, f2_val in enumerate(f2_folds):
                    mesh[j, i] = scores[(f1 == f1_val) & (f2 == f2_val)].item()

            offset = 1 if model.has_expand else 0
            model.fold_layers = og_model.fold_layers
            _, paper = model.forward(X, return_intermediate=True)
            outx = paper[offset + score_layer][:, f1id].detach().numpy()
            outy = paper[offset + score_layer][:, f2id].detach().numpy()

            # Create subplots
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Input Data", "Score Landscape"), 
                                specs=[[{"type": "scatter"}, {"type": "heatmap"}]])

            # Scatter plot with colors
            fig.add_trace(go.Scatter(x=outx, y=outy, mode='markers', 
                                    marker=dict(color=y.cpu().numpy().astype(int)*0.5 + 0.2, colorscale=theme, size=8), 
                                    name="Data", showlegend=False), row=1, col=1)

            # Add predicted and maximum folds
            pred_fold = og_model.fold_layers[score_layer].n.detach().numpy()
            best_fold = max_features_list[-1].cpu().numpy()
            fig.add_trace(idraw_fold(pred_fold, outx, outy, color="red", 
                                    name=f"Predicted Fold ({pred_fold[f1id]:.2f}, {pred_fold[f2id]:.2f})"), row=1, col=1)
            fig.add_trace(idraw_fold(best_fold, outx, outy, color="black", 
                                    name=f"Maximum Score Fold ({best_fold[f1id]:.2f}, {best_fold[f2id]:.2f})"), row=1, col=1)

            # Heatmap
            fig.add_trace(go.Heatmap(z=mesh, x=f1_folds, y=f2_folds, colorscale=theme, zmin=np.min(mesh)*0.99, zmax=np.max(mesh)*1.01), row=1, col=2)

            # Point on the max score
            max_index = np.unravel_index(np.argmax(mesh), mesh.shape)
            max_x = f1_folds[max_index[1]]
            max_y = f2_folds[max_index[0]]
            fig.add_trace(go.Scatter(x=[max_x], y=[max_y], mode='markers', 
                            marker=dict(color='black', size=8), name=f"Max={round(max_score, 2)}", showlegend=False), row=1, col=2)
            # Point on the predicted score
            fig.add_trace(go.Scatter(x=[pred_fold[f1id]], y=[pred_fold[f2id]], mode='markers',
                                     marker=dict(color='red', size=8), name=f"Predicted=NI", showlegend=False), row=1, col=2)

            # Update layout
            fig.update_xaxes(title_text=f"Feature {f1id}", row=1, col=1)
            fig.update_yaxes(title_text=f"Feature {f2id}", row=1, col=1)
            fig.update_xaxes(title_text=f"Feature {f1id}", row=1, col=2)
            fig.update_yaxes(title_text=f"Feature {f2id}", row=1, col=2)

            fig.update_layout(height=500, width=1000, 
                              title_text=f"Layer {score_layer} Visualization", 
                              showlegend=True, 
                              legend=dict(x=0.5, y=-0.2, xanchor="center", yanchor="bottom"))

            # Save plot if png_path is provided
            if png_path:
                fig.write_image(png_path)
            fig.show()

        # Restore original state
        model.fold_layers = og_model.fold_layers
        model.output_layer = og_model.output_layer

    if len(max_scores) == 1:
        return max_scores[0], max_features_list[0]
    return max_scores, max_features_list
   
