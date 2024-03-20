import gymnasium as gym
from time import sleep
import torch
import torch.nn as nn
import sys
import numpy as np
from typing import Tuple
from torch.distributions.categorical import Categorical
from torch.optim import Adam, Optimizer

def get_policy(model: nn.Module, observation: np.ndarray) -> Categorical:
	
	observation_tensor = torch.as_tensor(observation, dtype=torch.float32)

	#logits function ln(x/1-x)

	logits = model(observation_tensor)

	return Categorical(logits = logits)

def get_action(policy: Categorical) -> Tuple[int, torch.Tensor]:
	action = policy.sample()

	action_int = int(action.item())

	log_probability_action = policy.log_prob(action)

	return action_int, log_probability_action

def calculate_loss(epoch_log_probability_actions: torch.Tensor,
									 epoch_action_rewards: torch.Tensor) -> torch.Tensor:
	return -(epoch_log_probability_actions * epoch_action_rewards).mean()


def policy_gradient(env: gym.Env,
										model: nn.Module,
										optimizer: Optimizer,
										max_rollouts = 20) -> float:
	#discount factor
	discount = 0.99

	epoch_total_timesteps = 0
	epoch_returns: list[float] = []

	epoch_log_probability_actions = []
	epoch_action_rewards = []

	for i in range(max_rollouts):

		episode_reward_history = []
		
		observation, _ = env.reset()

		temp_epoch_log_prob = []

		finished_flag = False

		terminated, truncated = False, False

		while not terminated and not truncated:

			epoch_total_timesteps += 1

			policy = get_policy(model, observation)

			action, log_probability_action = get_action(policy)

			observation, reward, terminated, truncated , _ = env.step(action)

			episode_reward_history.append(reward)

			temp_epoch_log_prob.append(log_probability_action)

		epoch_log_probability_actions += temp_epoch_log_prob
		epoch_returns.append(sum(episode_reward_history))

		for t in range(len(episode_reward_history)-2, -1, -1):
			episode_reward_history[t] += discount * episode_reward_history[t+1]

		epoch_action_rewards += episode_reward_history
		
	epoch_loss = calculate_loss(torch.stack(epoch_log_probability_actions),
															torch.as_tensor(epoch_action_rewards, dtype=torch.float32))

	epoch_loss.backward()
	optimizer.step()
	optimizer.zero_grad()

	return float(np.mean(epoch_returns))





def reinforce(env: gym.Env,
							model: nn.Module,
							optimizer: Optimizer,
							max_timesteps = 5000,
							episode_timesteps = 200) -> float:
	epoch_total_timesteps = 0
	epoch_returns: list[float] = []

	epoch_log_probability_actions = []
	epoch_action_rewards = []

	while True:
		if epoch_total_timesteps > max_timesteps:
			break

		episode_reward: float = 0
		
		observation, _ = env.reset()

		temp_epoch_log_prob = []

		for timestep in range(episode_timesteps):

			epoch_total_timesteps += 1

			policy = get_policy(model, observation)

			action, log_probability_action = get_action(policy)

			observation, reward, done, _ , _ = env.step(action)

			episode_reward += reward

			temp_epoch_log_prob.append(log_probability_action)

			if done is True:
				epoch_log_probability_actions += temp_epoch_log_prob
				for _ in range(timestep + 1):
					epoch_action_rewards.append(episode_reward)
				break

				epoch_returns.append(episode_reward)
	
	if epoch_log_probability_actions:
		epoch_loss = calculate_loss(torch.stack(epoch_log_probability_actions),
															torch.as_tensor(epoch_action_rewards, dtype=torch.float32))

		epoch_loss.backward()
		optimizer.step()
		optimizer.zero_grad()

	return float(np.mean(epoch_returns))


