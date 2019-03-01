import gym

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim

import numpy as np
from tqdm import tqdm

from ppo.rollout_storage import RolloutStorage
from ppo.multiprocessing_env import SubprocVecEnv, VecNormalize, DummyVecEnv

from tensorboardX import SummaryWriter

import os
import shutil
import copy


def update_params(rollouts, policy, optimizer, params):
    advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
    # Normalize advantages. (0 mean. 1 std)
    advantages = (advantages - advantages.mean()) / (advantages.std() + params['eps'])

    for epoch_i in range(params['n_epoch']):
        samples = rollouts.sample(advantages, params['n_mini_batch'])

        value_losses = []
        action_losses = []
        entropy_losses = []
        losses = []
        for obs, actions, returns, masks, old_action_log_probs, adv_targ in samples:
            values, action_log_probs, dist_entropy = policy.evaluate_actions(obs, actions)

            # This is where we apply the PPO equation.
            ratio = torch.exp(action_log_probs - old_action_log_probs)

            surr1 = ratio * adv_targ
            surr2 = torch.clamp(ratio, 1.0 - params['clip_param'], 1.0 + params['clip_param']) * adv_targ

            action_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(returns, values)
            optimizer.zero_grad()

            loss = (value_loss * params['value_coeff'] + action_loss - dist_entropy * params['entropy_coeff'])
            loss.backward()

            nn.utils.clip_grad_norm_(policy.parameters(), params['max_grad_norm'])
            optimizer.step()

            value_losses.append(value_loss.item())
            action_losses.append(action_loss.item())
            entropy_losses.append(dist_entropy.item())
            losses.append(loss.item())

    return np.mean(value_losses), np.mean(action_losses), np.mean(entropy_losses), np.mean(losses)

def train(get_env_fn, get_policy, params):
    if not os.path.exists(params['model_dir']):
        os.makedirs(params['model_dir'])
    else:
        shutil.rmtree(params['model_dir'])
        os.makedirs(params['model_dir'])

    # Create logging directory
    if os.path.exists('runs'):
        shutil.rmtree('runs')
    writer = SummaryWriter()

    # Parallelize environments
    if params['n_envs'] == 1:
        envs = DummyVecEnv([get_env_fn])
    else:
        envs = [get_env_fn for i in range(params['n_envs'])]
        envs = SubprocVecEnv(envs)

    obs_shape = envs.observation_space['adj'].shape
    action_shape = (np.prod(envs.action_space.shape),)

    policy = get_policy(envs.observation_space, envs.action_space)

    optimizer = optim.Adam(policy.parameters(), lr=params['lr'], eps=params['eps'])

    # Intialize the tensor we will use everytime for the observation. See the note
    # in update_current_obs for more
    current_obs = torch.zeros(params['n_envs'], *obs_shape)
    obs = envs.reset()

    def update_current_obs(obs):
        obs_parts = []
        for i in range(len(obs)):
            nodes = obs[i]['node']
            adj = obs[i]['adj']
            #obs_parts.append(np.concatenate([nodes, adj], axis=0))
            obs_parts.append(adj)

        obs = np.array(obs_parts)
        # we want to use the same tensor every time so just copy it over.
        obs = torch.from_numpy(obs).float()
        current_obs[:, :] = obs

    update_current_obs(obs)

    # Intialize our rollouts
    rollouts = RolloutStorage(params['n_steps'], params['n_envs'], obs_shape,
            action_shape, current_obs)

    if params['cuda']:
        # Put on the GPU
        policy.cuda()
        rollouts.cuda()
        current_obs.cuda()

    episode_rewards = torch.zeros([params['n_envs'], 1])
    final_rewards = torch.zeros([params['n_envs'], 1])

    n_updates = int(params['n_frames'] // params['n_steps'] // params['n_envs'])
    for update_i in tqdm(range(n_updates)):
        # Generate samples
        for step in range(params['n_steps']):
            # Generate and take an action
            with torch.no_grad():
                value, action, action_log_prob = policy.act(rollouts.observations[step])

            take_actions = action.squeeze(1).cpu().numpy()

            if len(take_actions.shape) == 1:
                take_actions = np.expand_dims(take_actions, axis=-1)

            obs, reward, done, info = envs.step(take_actions)

            # convert to pytorch tensor
            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            masks = torch.FloatTensor([[0.0] if d else [1.0] for d in done])

            # update reward info for logging
            episode_rewards += reward
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            # Update our current observation tensor
            current_obs *= masks
            update_current_obs(obs)

            rollouts.insert(current_obs, action, action_log_prob, value, reward, masks)

        with torch.no_grad():
            next_value = policy.get_value(rollouts.observations[-1]).detach()

        rollouts.compute_returns(next_value, params['gamma'])

        value_loss, action_loss, entropy_loss, overall_loss = update_params(rollouts, policy,
                optimizer, params)

        rollouts.after_update()

        # Log to tensorboard
        writer.add_scalar('data/action_loss', action_loss, update_i)
        writer.add_scalar('data/value_loss', value_loss, update_i)
        writer.add_scalar('data/entropy_loss', entropy_loss, update_i)
        writer.add_scalar('data/overall_loss', overall_loss, update_i)
        writer.add_scalar('data/avg_reward', final_rewards.mean(), update_i)

        if update_i % params['log_interval'] == 0:
            print('Reward: %.3f' % (final_rewards.mean()))

        if update_i % params['save_interval'] == 0:
            save_model = policy
            if params['cuda']:
                save_model = copy.deepcopy(policy).cpu()

            torch.save(save_model, os.path.join(params['model_dir'], 'model_%i.pt' % update_i))

    writer.close()
