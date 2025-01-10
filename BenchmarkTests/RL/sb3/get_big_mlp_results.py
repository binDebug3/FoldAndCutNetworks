import fire
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from BenchmarkTests.RL.utils import count_parameters, NumParamsCallback
import numpy as np
import os

def main(env:str, run_num:int) :
    log_path = 'BenchmarkTests/RL/logs/' + env.split('-')[0] + '/'
    set_random_seed(run_num)
    if env == 'CartPole-v1' :
        vec_env = make_vec_env(env, n_envs=8)
        model = PPO("MlpPolicy", vec_env, n_steps=32, batch_size=256, gae_lambda=0.8,
                    gamma=0.98, n_epochs=20, ent_coef=0.0, learning_rate=0.001, 
                    clip_range=0.2, verbose=1, tensorboard_log=log_path, device='cpu')
        training_steps = 100000

    elif env == 'LunarLander-v3':
        vec_env = make_vec_env(env, n_envs=16)
        model = PPO("MlpPolicy", vec_env, n_steps=1024, batch_size=64, gae_lambda=0.98,
                    gamma=0.999, n_epochs=4, ent_coef=0.01, verbose=1, 
                    tensorboard_log=log_path, device='cpu')
        training_steps = 1000000
    elif env == "HalfCheetah-v5" :
        model = SAC('MlpPolicy', "HalfCheetah-v5", learning_starts=10000, verbose=1, tensorboard_log=log_path)
        training_steps = 1000000
    else:
        raise ValueError(f"Unsupported environment: {env}")
    
    print(count_parameters(model), 'parameters')
    model.learn(total_timesteps=training_steps, tb_log_name=f'new_full_mlp/run_{run_num}',
                callback=NumParamsCallback())
    model.save(log_path + f'new_full_mlp/model_{run_num}')

if __name__ == "__main__":
    fire.Fire(main)

