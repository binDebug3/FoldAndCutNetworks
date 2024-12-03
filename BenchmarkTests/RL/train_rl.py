import fire
from stable_baselines3 import SAC, PPO
from BenchmarkTests.RL.custom_policy import CustomPPOPolicy, CustomSACPolicy
from stable_baselines3.common.env_util import make_vec_env
from BenchmarkTests.experimenter import get_model
import json

def main(env:str, model_index:int) :
    # load the benchmark model names
    with open('BenchmarkTests/architectures.json') as f:
        architectures = json.load(f)
    benchmark_models = list(architectures.keys())

    log_path = 'BenchmarkTests/RL/logs'
    if env == "CartPole-v1" :
        vec_env = make_vec_env("CartPole-v1", n_envs=1)
        if model_index == -1 :
            model = PPO("MlpPolicy", vec_env, n_steps=256, verbose=1, tensorboard_log=log_path)
        else :
            custom_policy_kwargs=dict(model_name=benchmark_models[model_index])
            model = PPO(CustomPPOPolicy, vec_env, n_steps=256, policy_kwargs=custom_policy_kwargs, 
                        verbose=1, tensorboard_log=log_path)
    
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
    model.learn(total_timesteps=30000, tb_log_name=get_exp_name(env, model_index, benchmark_models))

def get_exp_name(env:str, model_index:int, benchmark_models:list) :
    if model_index == -1 :
        return f'{env.split('-')[0]}_mlp'
    else :
        return f'{env.split('-')[0]}_samelr_{benchmark_models[model_index]}'

if __name__ == "__main__":
    fire.Fire(main)