from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from gymnasium import spaces
import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import Actor, ContinuousCritic
# from stable_baselines3.common.policies import 
from BenchmarkTests.experimenter import get_model
import json

class FoldAndCutRLNetwork(nn.Module):
    """
    Custom network for policy and value function.
    Receives as input the observation, or the output of the feature extractor if there is one

    :param feature_dim: dimension of the features extracted with the features_extractor, or directly from the obs space
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        model_name: str,
        feature_dim: int,
        last_layer_dim_pi: int,
        last_layer_dim_vf: int,
        no_relu: bool
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net, lr = get_model(model_name, input_size=feature_dim,
                                        output_size=last_layer_dim_pi, 
                                        architecture_path_local='BenchmarkTests/RL/rl_architectures.json',
                                        no_cut=True, no_relu=no_relu)
        # Value network
        self.value_net, lr = get_model(model_name, input_size=feature_dim,
                                       output_size=last_layer_dim_vf, 
                                       architecture_path_local='BenchmarkTests/RL/rl_architectures.json',
                                       no_cut=True, no_relu=no_relu)
        self.lr = lr

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)
    
    def get_lr(self) :
        return self.lr


class CustomPPOPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        model_name: str = None,
        no_relu: bool = None,
        *args,
        **kwargs,
    ):
        self.model_name = model_name
        self.no_relu = no_relu

        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        with open('BenchmarkTests/RL/rl_architectures.json') as f :
            architectures = json.load(f)
        model_dict = architectures.get(self.model_name, {})
        output_dim = model_dict['structure'][-1]['params']['width'] if \
                        'Fold' in model_dict['structure'][-1]['type'] else \
                        model_dict['structure'][-1]['params']['out_features']
        self.mlp_extractor = FoldAndCutRLNetwork(self.model_name, 
                                                 self.features_dim,
                                                 self.features_dim*output_dim,
                                                 self.features_dim*output_dim,
                                                 self.no_relu)


class CustomSACPolicy(SACPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        model_name: str = None,
        **kwargs,
    ):
        self.model_name = model_name
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs,
        )

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        actor_net, lr = get_model(self.model_name, input_size=self.observation_space.shape[0], 
                                  output_size=self.action_space.shape[0],
                                  architecture_path_local='BenchmarkTests/RL/rl_architectures.json')
        return Actor(
            observation_space=self.observation_space,
            action_space=self.action_space,
            net_arch=[],  # Empty, as we're using our custom network
            features_extractor=actor_net,
            features_dim=self.action_space.shape[0]
        )

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return CustomContinuousCritic(
            self.model_name,
            self.observation_space,
            self.action_space,
            net_arch=[],  # Empty, as we're using our custom network
            features_extractor=nn.Flatten(),
            features_dim=self.observation_space.shape[0]
        )

class CustomContinuousCritic(ContinuousCritic) :
    def __init__(
        self,
        model_name: str,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: list[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int
    ): 
        super().__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim
        )
        
        # remove the networks that the default ContinuousCritic made
        del self.q_networks
        del self.qf0
        del self.qf1

        # now make them ourselves, but with FoldAndCutNetworks instead of MLPs
        self.q_networks: list[nn.Module] = []
        for idx in range(self.n_critics):
            q_net, lr = get_model(model_name, self.observation_space.shape[0] + self.action_space.shape[0], output_size=1, 
                                  architecture_path_local='BenchmarkTests/RL/rl_architectures.json')
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)
