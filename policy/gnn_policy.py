import torch.nn as nn
import torch.nn.functional as F
import torch

class GnnPolicy(nn.Module):
    def __init__(self, obs_shape, num_outputs):
        super().__init__()

        #########################################
        # JUST A BASIC MLP FOR NOW.
        # THIS IS WHAT NEEDS TO BE CHANGED
        #########################################
        num_inputs = obs_shape[0]
        self.actor_hidden = nn.Sequential(
                nn.Linear(num_inputs, num_outputs),
                nn.Tanh(),
                nn.Linear(num_outputs, num_outputs),
                nn.Tanh(),
            )

        self.critic = nn.Sequential(
                nn.Linear(num_inputs, num_outputs),
                nn.Tanh(),
                nn.Linear(num_outputs, num_outputs),
                nn.Tanh(),
                nn.Linear(num_outputs, 1),
            )

    def forward(self, inputs):
        return self.actor_hidden(inputs), self.critic(inputs)
