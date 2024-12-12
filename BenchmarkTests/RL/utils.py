from stable_baselines3.common.callbacks import BaseCallback
import torch

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