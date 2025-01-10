from stable_baselines3 import PPO
import torch 
import matplotlib.pyplot as plt
import numpy as np

crease_policy_first_layer = []
crease_policy_second_layer = []
crease_value_first_layer = []
crease_value_second_layer = []
for i in range(1, 4):
    model = PPO.load(f"BenchmarkTests/RL/logs/LunarLander/two_hidden_layer_8x_softfold_has_stretch_crease_sign1_{i}", device="cpu")
    crease_policy_first_layer.append(torch.svd(model.get_parameters()['policy']['mlp_extractor.policy_net.layers.0.weight'])[1])
    crease_policy_second_layer.append(torch.svd(model.get_parameters()['policy']['mlp_extractor.policy_net.layers.3.weight'])[1])
    crease_value_first_layer.append(torch.svd(model.get_parameters()['policy']['mlp_extractor.value_net.layers.0.weight'])[1])
    crease_value_second_layer.append(torch.svd(model.get_parameters()['policy']['mlp_extractor.value_net.layers.3.weight'])[1])

mlp_policy_first_layer = []
mlp_policy_second_layer = []
mlp_value_first_layer = []
mlp_value_second_layer = []
for i in range(1, 4):
    model = PPO.load(f"BenchmarkTests/RL/logs/LunarLander/new_full_mlp/model_{i}", device="cpu")
    mlp_policy_first_layer.append(torch.svd(model.get_parameters()['policy']['mlp_extractor.policy_net.0.weight'])[1])
    mlp_policy_second_layer.append(torch.svd(model.get_parameters()['policy']['mlp_extractor.policy_net.2.weight'])[1])
    mlp_value_first_layer.append(torch.svd(model.get_parameters()['policy']['mlp_extractor.value_net.0.weight'])[1])
    mlp_value_second_layer.append(torch.svd(model.get_parameters()['policy']['mlp_extractor.value_net.2.weight'])[1])

    # print("MODEL NUMBER", i)
    # print("Policy Net")
    # print("first linear layer singular values")
    # print(torch.svd(model.get_parameters()['policy']['mlp_extractor.policy_net.layers.0.weight'])[1])
    # print("first fold n", sum(model.get_parameters()['policy']['mlp_extractor.policy_net.layers.2.n'] > 0), "positive out of", len(model.get_parameters()['policy']['mlp_extractor.policy_net.layers.2.n']))
    # print("first fold crease", model.get_parameters()['policy']['mlp_extractor.policy_net.layers.2.crease'])
    # print("first fold stretch", model.get_parameters()['policy']['mlp_extractor.policy_net.layers.2.stretch'])
    # print("second linear layer singular values")
    # sing_vals = torch.svd(model.get_parameters()['policy']['mlp_extractor.policy_net.layers.3.weight'])[1]
    # print("first ten", sing_vals[:10])
    # print("last ten", sing_vals[-10:])
    # print("second fold n", sum(model.get_parameters()['policy']['mlp_extractor.policy_net.layers.5.n'] > 0), "positive out of", len(model.get_parameters()['policy']['mlp_extractor.policy_net.layers.5.n']))
    # print("second fold crease", model.get_parameters()['policy']['mlp_extractor.policy_net.layers.5.crease'])
    # print("second fold stretch", model.get_parameters()['policy']['mlp_extractor.policy_net.layers.5.stretch'])
    # print("Value Net")
    # print("first linear layer singular values")
    # print(torch.svd(model.get_parameters()['policy']['mlp_extractor.value_net.layers.0.weight'])[1])
    # print("first fold n", sum(model.get_parameters()['policy']['mlp_extractor.value_net.layers.2.n'] > 0), "positive out of", len(model.get_parameters()['policy']['mlp_extractor.value_net.layers.2.n']))
    # print("first fold crease", model.get_parameters()['policy']['mlp_extractor.value_net.layers.2.crease'])
    # print("first fold stretch", model.get_parameters()['policy']['mlp_extractor.value_net.layers.2.stretch'])
    # print("second linear layer singular values")
    # sing_vals = torch.svd(model.get_parameters()['policy']['mlp_extractor.value_net.layers.3.weight'])[1]
    # print("first ten", sing_vals[:10])
    # print("last ten", sing_vals[-10:])
    # print("second fold n", sum(model.get_parameters()['policy']['mlp_extractor.value_net.layers.5.n'] > 0), "positive out of", len(model.get_parameters()['policy']['mlp_extractor.value_net.layers.5.n']))
    # print("second fold crease", model.get_parameters()['policy']['mlp_extractor.value_net.layers.5.crease'])
    # print("second fold stretch", model.get_parameters()['policy']['mlp_extractor.value_net.layers.5.stretch'])

    # print("MODEL NUMBER", i)
    # print("Policy Net")
    # print("first linear layer singular values")
    # print(torch.svd(model.get_parameters()['policy']['mlp_extractor.policy_net.0.weight'])[1])
    # print("second linear layer singular values")
    # sing_vals = torch.svd(model.get_parameters()['policy']['mlp_extractor.policy_net.2.weight'])[1]
    # print("first ten", sing_vals[:10])
    # print("last ten", sing_vals[-10:])
    # print("Value Net")
    # print("first linear layer singular values")
    # print(torch.svd(model.get_parameters()['policy']['mlp_extractor.value_net.0.weight'])[1])
    # print("second linear layer singular values")
    # sing_vals = torch.svd(model.get_parameters()['policy']['mlp_extractor.value_net.2.weight'])[1]
    # print("first ten", sing_vals[:10])
    # print("last ten", sing_vals[-10:])

# save a figure with four subplots for each layers of each network
fig, axs = plt.subplots(2, 2)
fig.suptitle('Singular Values of Linear Layers of Policy and Value Networks')
for i in range(3):
    axs[0, 0].plot(crease_policy_first_layer[i].numpy(), color='red', label='with folds' if i == 0 else '')
    axs[0, 0].plot(mlp_policy_first_layer[i].numpy(), color='blue', label='MLP' if i == 0 else '')
    axs[0, 0].set_title('First Layer Policy Network')
    axs[0, 1].plot(crease_policy_second_layer[i].numpy(), color='red', label='with folds' if i == 0 else '')
    axs[0, 1].plot(mlp_policy_second_layer[i].numpy(), color='blue', label='MLP' if i == 0 else '')
    axs[0, 1].set_title('Second Layer Policy Network')
    axs[1, 0].plot(crease_value_first_layer[i].numpy(), color='red', label='with folds' if i == 0 else '')
    axs[1, 0].plot(mlp_value_first_layer[i].numpy(), color='blue', label='MLP' if i == 0 else '')
    axs[1, 0].set_title('First Layer Value Network')
    axs[1, 1].plot(crease_value_second_layer[i].numpy(), color='red', label='with folds' if i == 0 else '')
    axs[1, 1].plot(mlp_value_second_layer[i].numpy(), color='blue', label='MLP' if i == 0 else '')
    axs[1, 1].set_title('Second Layer Value Network')
plt.legend()
fig.tight_layout(h_pad=2) 
plt.show()
plt.savefig("images/singular_values.png")

