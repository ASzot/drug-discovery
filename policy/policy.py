import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, get_base, hidden_out):
        super().__init__()

        self.actor_critic = get_base(obs_shape, hidden_out)

        num_outputs = np.prod(action_space.shape)

        # How we will define our normal distribution to sample action from
        self.action_mean = nn.Linear(hidden_out, num_outputs)
        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))

    def __get_dist(self, actor_features):
        action_mean = self.action_mean(actor_features)
        action_log_std = self.action_log_std.expand_as(action_mean)

        return torch.distributions.Normal(action_mean, action_log_std.exp())


    def act(self, inputs, deterministic=False):
        actor_features, value = self.actor_critic(inputs)
        dist = self.__get_dist(actor_features)

        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()

        action_log_probs = dist.log_prob(action).sum(-1, keepdim=True)
        dist_entropy = dist.entropy().sum(-1).mean()

        return value, action, action_log_probs

    def get_value(self, inputs):
        _, value = self.actor_critic(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        actor_features, value = self.actor_critic(inputs)
        dist = self.__get_dist(actor_features)

        action_log_probs = dist.log_prob(action).sum(-1, keepdim=True)
        dist_entropy = dist.entropy().sum(-1).mean()

        return value, action_log_probs, dist_entropy
