import fire
from stable_baselines3 import SAC, PPO
from BenchmarkTests.RL.custom_policy import CustomPPOPolicy, CustomSACPolicy
from stable_baselines3.common.env_util import make_vec_env
from BenchmarkTests.RL.utils import count_parameters, NumParamsCallback
import numpy as np
import json
import os

def main(env:str, model_index:int, mlp:bool, no_relu:bool) :
    # mlp = mlp == 'true' # easier for slurm to pass it as a string

    # load the benchmark model names
    with open('BenchmarkTests/RL/rl_architectures.json') as f:
        architectures = json.load(f)
    benchmark_models = list(architectures.keys())
    model_name = benchmark_models[model_index]

    log_path = 'BenchmarkTests/RL/logs/' + env.split('-')[0] + '/'
    if env in ['CartPole-v1', 'LunarLander-v3'] :
        vec_env = make_vec_env(env, n_envs=1)
        custom_policy_kwargs=dict(model_name=model_name, no_relu=no_relu)
        model = PPO(CustomPPOPolicy, vec_env, n_steps=256, policy_kwargs=custom_policy_kwargs, 
                    verbose=1, tensorboard_log=log_path, device='cpu')
        if mlp :
            num_params = count_parameters(model)
            action_dim = 2 if env == 'CartPole-v1' else 4
            # this solves for the root of the polynomial that maps hidden dimension size
            # to the number of parameters in the PPO model whose networks have 2 hidden layers 
            # to get the hidden size to make the MlpPolicy match the number of parameters in the custom policy
            a = 2
            b = 2*vec_env.observation_space.shape[0] + action_dim + 5
            c = action_dim + 1 - num_params
            mlp_size = int(np.round((-b + np.sqrt(b**2 - 4*a*c))/(2*a)))
            policy_kwargs=dict(net_arch=dict(pi=[mlp_size, mlp_size], vf=[mlp_size, mlp_size]))
            model = PPO("MlpPolicy", vec_env, n_steps=256, policy_kwargs=policy_kwargs, 
                        verbose=1, tensorboard_log=log_path, device='cpu')
    
    elif env == "HalfCheetah-v4" :
        if model_index == -1 :
            model = SAC('MlpPolicy', "HalfCheetah-v4", verbose=1, tensorboard_log=log_path)
        else : 
            custom_policy_kwargs=dict(
                model_name=benchmark_models[model_index],
                share_features_extractor=False    
            )
            model = SAC(CustomSACPolicy, "HalfCheetah-v4", 
                        policy_kwargs=custom_policy_kwargs, verbose=1, tensorboard_log=log_path)
    
    model.learn(total_timesteps=60000, tb_log_name=get_exp_name(log_path, benchmark_models[model_index], mlp, no_relu),
                callback=NumParamsCallback())
    


def get_exp_name(log_path:str, model_name:str, mlp:bool, no_relu:bool) :
    no_relu_str = '_no_relu' if no_relu else ''
    tb_log_name = model_name + '/mlp' if mlp else model_name + '/fold' + no_relu_str
    i = 1
    while True : 
        if not os.path.isdir(log_path + tb_log_name + f'_{i}') :
            tb_log_name += f'_{i}'
            break
        i += 1
    return tb_log_name

if __name__ == "__main__":
    fire.Fire(main)