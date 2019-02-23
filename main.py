from gym_molecule.envs.molecule import GraphEnv
import numpy as np
from ppo.args import get_default_args
from ppo.ppo import train
from policy.gnn_policy import GnnPolicy
from policy.policy import Policy


# Out action space is outputting 4 numbers
# - Node 1
# - Node 2
# - Edge type
#   - For the GraphEnv this is always 0.
# - Is done? (0,1)

def get_env():
    env = GraphEnv()
    env.init(reward_step_total=0.5, is_normalize=0, dataset='zinc')
    return env

def get_gnn_policy(obs_space, action_space):
    return Policy(obs_space, action_space, GnnPolicy, 64)

params = get_default_args()
params['n_envs'] = 2
params['n_mini_batch'] = 5
train(get_env, get_gnn_policy, params)

