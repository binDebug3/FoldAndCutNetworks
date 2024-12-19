import fire
from stable_baselines3 import SAC, PPO
from BenchmarkTests.RL.custom_policy import CustomPPOPolicy, CustomSACPolicy
from stable_baselines3.common.env_util import make_vec_env
from BenchmarkTests.RL.utils import count_parameters, NumParamsCallback
import numpy as np
import json
import os

def main(env:str, model_index:int, mlp:bool, no_relu:bool, run_num:int) :
    # load the benchmark model names
    with open('BenchmarkTests/RL/rl_architectures.json') as f:
        architectures = json.load(f)
    benchmark_models = list(architectures.keys())
    model_name = benchmark_models[model_index]

    log_path = 'BenchmarkTests/RL/logs/' + env.split('-')[0] + '/'

    if env in ['CartPole-v1', 'LunarLander-v3'] :
        custom_policy_kwargs=dict(model_name=model_name, no_relu=no_relu)
        if env == 'CartPole-v1' : 
            vec_env = make_vec_env(env, n_envs=8)
            kwargs=dict(policy_kwargs=custom_policy_kwargs, 
                        n_steps=32, batch_size=256, gae_lambda=0.8,
                        gamma=0.98, n_epochs=20, ent_coef=0.0, learning_rate=0.001, 
                        clip_range=0.2, verbose=1, tensorboard_log=log_path, 
                        device='cpu')
        if env == 'LunarLander-v3' :
            vec_env = make_vec_env(env, n_envs=16)
            kwargs=dict(policy_kwargs=custom_policy_kwargs, 
                        n_steps=1024, batch_size=64, gae_lambda=0.98,
                        gamma=0.999, n_epochs=4, ent_coef=0.01, verbose=1, 
                        tensorboard_log=log_path, device='cpu')
        model = PPO(CustomPPOPolicy, vec_env, **kwargs)
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
            kwargs["policy_kwargs"] = policy_kwargs
            model = PPO("MlpPolicy", vec_env, **kwargs)
    
    if env == "HalfCheetah-v5" :
        custom_policy_kwargs=dict(
            model_name=benchmark_models[model_index],
            share_features_extractor=False    
        )
        model = SAC(CustomSACPolicy, "HalfCheetah-v5", policy_kwargs=custom_policy_kwargs,
                    learning_starts=10000, verbose=1, tensorboard_log=log_path)
        if mlp :
            num_params = count_parameters(model)
            action_dim = 6
            obs_dim = 17
            a = 5
            b = 4*(action_dim+obs_dim) + 2*action_dim + obs_dim + 14
            c = 4 + 2*action_dim - num_params
            mlp_size = int(np.round((-b + np.sqrt(b**2 - 4*a*c))/(2*a)))
            policy_kwargs=dict(net_arch=dict(pi=[mlp_size, mlp_size], qf=[mlp_size, mlp_size]))
            model = SAC('MlpPolicy', "HalfCheetah-v5", policy_kwargs=policy_kwargs, 
                        verbose=1, tensorboard_log=log_path)

    
    training_steps = 100000 if env == 'CartPole-v1' else 1000000
    model.learn(total_timesteps=training_steps, 
                tb_log_name=get_exp_name(benchmark_models[model_index], 
                                         mlp, no_relu, run_num),
                callback=NumParamsCallback())
    


def get_exp_name(model_name:str, mlp:bool, no_relu:bool, run_num:int) :
    no_relu_str = '_no_relu' if no_relu else ''
    tb_log_name = model_name + '/mlp' if mlp else model_name + '/fold' + no_relu_str
    tb_log_name += f'_{run_num}'
    return tb_log_name

if __name__ == "__main__":
    fire.Fire(main)



# rerun with half cheetah v5 see if all still works
# delete the stuff in the logs from the experiments 
# run all experiments for half cheetah with zero 1 2 
# run all experiments with lunar lander for zero 1 
# run do full mlp with half cheetah v5
# plot and compare