import numpy as np
import copy
from tqdm import tqdm

from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.io as pio

from matplotlib import pyplot as plt




def plot_wiggles(layer, histories):
    y_wiggles = [x[layer][0] for x in histories[2]]#[325:340]
    try:
        x_wiggles = [x[layer][1] for x in histories[2]]#[325:340]
    except:
        x_wiggles = None
    lrs = histories[-1]
    name = "Learning Rates" if x_wiggles is None else "X Wiggles"
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Y Wiggles", name))
    fig.add_trace(go.Scatter(y=y_wiggles, mode='lines', name='y_wiggles'), row=1, col=1)
    if x_wiggles is not None:
        fig.add_trace(go.Scatter(y=x_wiggles, mode='lines', name='x_wiggles'), row=1, col=2)
    else:
        fig.add_trace(go.Scatter(y=lrs, mode='lines', name='Learning Rates'), row=1, col=2)
    fig.update_layout(width=1000, height=400, title_text=f'Fold layer {layer} over epochs')
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    pio.show(fig)
    
    
    


def draw_fold( hyperplane, outx, outy, color, name):
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
        plane_range = np.where((plane_range > np.min(outy)) & (plane_range < np.max(outy)), plane_range, np.nan)
        plt.plot(plane_domain, plane_range, color=color, lw=2, label=name)


def plot_history(model, n_folds=50, include_cut=True, verbose=1):
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
        og_output_layer = copy.deepcopy(model.output_layer)
        og_b_thing = copy.deepcopy(model.b)
        
    # loop over each fold
    out = model.forward_pass(X)
    plt.figure(figsize=(8, 3.5*model.layers), dpi=150)
    n_cols = 2
    n_rows = model.layers // n_cols + 1
    
    for i in range(model.layers):
        outx = out[i][:,0]
        outy = out[i][:,1]
        progress = tqdm(total=n_folds, desc="Plotting", disable=verbose==0)
        
        # plot every mod_number fold and decision boundary
        plt.subplot(n_rows, n_cols, i+1)
        for j in range(0, len(model.fold_history), mod_number):
            idx = j//mod_number
            draw_fold(model.fold_history[j][i], outx, outy, color=colors[idx], name=None)
            if include_cut:
                model.output_layer = model.cut_history[j][0]
                model.b = model.cut_history[j][1]
                Z = model.predict(grid_points)
                Z = Z.reshape(xx.shape)
                plt.contourf(xx, yy, Z, alpha=scalor*idx, cmap=plt.cm.YlGnBu)   
            progress.update(1)
        
        hyperplane = np.round(model.fold_vectors[i], 2)
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
    model.output_layer = og_output_layer.copy()
    model.b = og_b_thing.copy()

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
    miny = np.min(outy) * rescale if np.min(outy) > 0 else np.min(outy) / rescale
    minx = np.min(outx) * rescale if np.min(outx) > 0 else np.min(outx) / rescale
    maxy = np.max(outy) * rescale if np.max(outy) > 0 else np.max(outy) / rescale
    maxx = np.max(outx) * rescale if np.max(outx) > 0 else np.max(outx) / rescale
    plt.scatter(outx, outy, c=Y)
    plt.ylim(miny, maxy)
    plt.xlim(minx, maxx)
    plt.xlabel("Feature 1", fontsize=6)
    plt.title("Final Decision Boundary", fontsize=8)
    plt.tick_params(axis='both', which='major', labelsize=6)
    plt.suptitle("Pink is early, yellow is late", fontsize=9)  
    plt.tight_layout()  
    plt.show()
    
    if verbose > 1:
        print(model.output_layer)
        print(model.b)
        print("Fold vectors")
        print(model.fold_vectors)