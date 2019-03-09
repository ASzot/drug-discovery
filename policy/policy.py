import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from policy.dist import Categorical, Bernoulli

class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, get_base):
        super().__init__()

        # Number of hidden layers to use in the "m" networks from Equation 3.
        m_hidden = obs_shape['node'].shape[1]

        self.actor_critic = get_base(obs_shape, m_hidden)

        self.dists = nn.ModuleList([
            Categorical(m_hidden, action_dim) for action_dim in action_space.nvec
            ])

    def calculate_log_probs(self, actions, dists):
        #action_log_probs = torch.cat([
        #        dist.log_probs(action)
        #        for action, dist in zip(actions, dists)
        #        ], dim=-1)
        action_log_probs = []
        for action, dist in zip(actions, dists):
            action_log_probs.append(dist.log_probs(action))
        action_log_probs = torch.stack(action_log_probs, dim=1)

        action_log_probs = torch.sum(action_log_probs, dim=1)
        return action_log_probs

    def calculate_entropies(self, dists):
        dist_entropies = torch.stack([
                dist.entropy()
                for dist in dists
                ], dim=-1)
        dist_entropies = torch.mean(dist_entropies, dim=-1)
        return dist_entropies

    def get_dists(self, actor_features):
        dists = [
            dist(actor_feature)
            for actor_feature, dist in zip(actor_features, self.dists)
            ]
        return dists

    def act(self, inputs, deterministic=False):
        actor_features, value = self.actor_critic(inputs)
        dists = self.get_dists(actor_features)

        if deterministic:
            actions = [dist.mode() for dist in dists]
        else:
            actions = [dist.sample() for dist in dists]

        action_log_probs = self.calculate_log_probs(actions, dists)
        actions = torch.stack(actions, dim=-1)

        return value, actions, action_log_probs

    def get_value(self, inputs):
        _, value = self.actor_critic(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        actor_features, value = self.actor_critic(inputs)
        dists = self.get_dists(actor_features)

        action = action.reshape((4, -1, 1))
        action_log_probs = self.calculate_log_probs(action, dists)
        dist_entropy = self.calculate_entropies(dists)

        return value, action_log_probs, dist_entropy
