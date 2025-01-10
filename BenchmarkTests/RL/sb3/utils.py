from stable_baselines3.common.callbacks import BaseCallback
import tensorflow as tf
import torch
import numpy as np

def count_parameters(model):
    return sum(p.numel() for name, p in model.get_parameters()['policy'].items() if isinstance(p, torch.Tensor))

class NumParamsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.model = None

    def _on_training_start(self) -> None:
        num_params = count_parameters(self.model)
        self.logger.record("num_params", num_params)

    def _on_step(self) -> bool:
        return True



def get_scalar_run_tensorboard(file_path):
    rollout_ep_rew_mean = []
    # Read the TensorBoard file
    for event in tf.compat.v1.train.summary_iterator(file_path):
        for value in event.summary.value:
            if value.tag == 'rollout/ep_rew_mean':
                rollout_ep_rew_mean.append(value.simple_value)

    # Convert the list to a NumPy array
    return np.array(rollout_ep_rew_mean)