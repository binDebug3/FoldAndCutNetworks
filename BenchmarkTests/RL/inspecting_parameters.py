import torch
from BenchmarkTests.RL.MEow.meow_continuous_action import FlowPolicy
import os
import numpy as np

good_runs_num_pos = []
bad_runs_num_pos = []
good_runs_sum = []
bad_runs_sum = []
good_runs_crease = []
bad_runs_crease = []
for run in os.listdir("runs/HalfCheetah-v4/fold_ln/"):
    flow_policy = torch.load(os.path.join("runs/HalfCheetah-v4/fold_ln", run, "init.pt"), map_location="cpu")
    for name, param in flow_policy.named_parameters() :
        if name[-1] == 'n' :
            if ('__2__' in run) or ('__4__' in run) :
                good_runs_num_pos.append((sum(param > 0) / len(param)).item())
                good_runs_sum.append(sum(param).item())
            else :
                bad_runs_num_pos.append((sum(param > 0) / len(param)).item())
                bad_runs_sum.append(sum(param).item())
        if 'crease' in name:
            if ('__2__' in run) or ('__4__' in run) :
                good_runs_crease.append(param.item())
            else :
                bad_runs_crease.append(param.item())

np.save("BenchmarkTests/RL/MEow/good_runs_num_pos_init.npy", np.array(good_runs_num_pos))
np.save("BenchmarkTests/RL/MEow/bad_runs_num_pos_init.npy", np.array(bad_runs_num_pos))
np.save("BenchmarkTests/RL/MEow/good_runs_sum_init.npy", np.array(good_runs_sum))
np.save("BenchmarkTests/RL/MEow/bad_runs_sum_init.npy", np.array(bad_runs_sum))
np.save("BenchmarkTests/RL/MEow/good_runs_crease_init.npy", np.array(good_runs_crease))
np.save("BenchmarkTests/RL/MEow/bad_runs_crease_init.npy", np.array(bad_runs_crease))


