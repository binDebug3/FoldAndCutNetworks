import numpy as np
import matplotlib.pyplot as plt
import os

def evaluate_model(dataset_folder, top_n = 5):
    """
    Determine which Model configuration performed best on the dataset

    Parameters:
            dataset_folder (str): path to the dataset folder
            top_n (int): number of top models to display

    Returns:
            dict: models and overall statistics for that dataset
    """

    model_results = []

    for model in os.listdir(dataset_folder):
        model_path = os.path.join(dataset_folder, model)
        if os.path.isdir(model_path):
            npy_path = os.path.join(model_path, 'npy_files')
            if not os.path.exists(npy_path):
                print(f"Model {model} does not have npy files")
                continue
            
            try:
                train_acc_file = [f for f in os.listdir(npy_path) if f.startswith('train_acc')]
                train_loss_file = [f for f in os.listdir(npy_path) if f.startswith('train_loss')]
                val_loss_file = [f for f in os.listdir(npy_path) if f.startswith('val_loss')]
                params_file = [f for f in os.listdir(npy_path) if f.startswith('params')]
                time_file = [f for f in os.listdir(npy_path) if f.startswith('train_time')]

                train_accuracy = np.mean([np.load(os.path.join(npy_path, f)) for f in train_acc_file])
                train_loss = np.mean([np.load(os.path.join(npy_path, f)) for f in train_loss_file])
                val_loss = np.mean([np.load(os.path.join(npy_path, f)) for f in val_loss_file])
                params = np.mean([np.load(os.path.join(npy_path, f)) for f in params_file]) if params_file else 0
                train_time = np.mean([np.load(os.path.join(npy_path, f)) for f in time_file]) if time_file else 0


                score = train_accuracy - train_loss - val_loss - 0.01 * params - 0.01 * train_time

                model_results.append({
                    'model': model,
                    'train_accuracy': train_accuracy,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'params': params,
                    'train_time': train_time,
                    'score': score
                })

            except Exception as e:
                print(f"Error processing model {model}: {e}")
                continue
    
    model_results = sorted(model_results, key=lambda x: x['score'], reverse=True)

    return model_results[:top_n]


def plot_loss(dataset_name, models):
    """
    Plot the loss for the top models

    Parameters:
            dataset_name (str): name of the dataset
            models (list): list of top models

    Returns:
            None
    """
    model_names = [model['model'] for model in models]
    train_loss = [model['train_loss'] for model in models]
    val_loss = [model['val_loss'] for model in models]

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Models')
    plt.ylabel('Loss')
    plt.title(f'Top Models for {dataset_name}')
    plt.legend()
    plt.show()


def plot_parameter_and_time(dataset_name, models):
    """
    Plot the parameters and time for the top models

    Parameters:
            dataset_name (str): name of the dataset
            models (list): list of top models

    Returns:
            None
    """
    
    model_names = [model['model'] for model in models]
    params = [model['params'] for model in models]
    train_time = [model['train_time'] for model in models]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Parameters', color=color)
    ax1.bar(model_names, params, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Train Time', color=color)
    ax2.plot(model_names, train_time, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title(f'Top Models for {dataset_name}')
    plt.show()

if __name__ == '__main__':
    dataset_folder = '../results/HIGGS/'
    top_n = 10
    results = evaluate_model(dataset_folder, top_n)
    #print(results)
    for result in results:
        print(f"Model: {result['model']})")

    plot_loss('HIGGS', results)
    plot_parameter_and_time('HIGGS', results)




