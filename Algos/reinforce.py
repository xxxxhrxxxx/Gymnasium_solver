import gymnasium as gym
from time import sleep
import torch
import math
import scipy
import torch.nn as nn
import sys
import numpy as np
from typing import Tuple
from torch.distributions.categorical import Categorical
from torch.optim import Adam, Optimizer

def vpg_model(num_obs, num_actions) -> nn.Module:
	#create the vg MLP model
	#output logits

	hidden_layer_features = 32

	return nn.Sequential(nn.Linear(in_features = num_obs,
																 out_features = hidden_layer_features),
																 nn.ReLU(),
											 nn.Linear(in_features = hidden_layer_features,
											 					 out_features = num_actions)
											)

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

def policy_gradient_center(env: gym.Env,
													 model: nn.Module,
													 optimizer: Optimizer,
										       max_rollouts = 200) -> float:
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
		
	epoch_mean = np.mean(epoch_returns)
	epoch_var = math.sqrt(np.var(epoch_returns))

	def zero_center(value):
		return (value - epoch_mean) / epoch_var

	epoch_action_rewards = list(map(zero_center, epoch_action_rewards))

	epoch_loss = calculate_loss(torch.stack(epoch_log_probability_actions),
															torch.as_tensor(epoch_action_rewards, dtype=torch.float32))

	epoch_loss.backward()
	optimizer.step()
	optimizer.zero_grad()

	return float(np.mean(epoch_returns))



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

def vpg_train(epochs = 1000, load_model = False, algo = 'reinforce', gym_environment = 'CartPole-v1') -> None:
	env = gym.make(gym_environment)

	#torch.manual_seed(0)
	#env.seed(0)

	number_observation_features = env.observation_space.shape[0]
	number_actions = env.action_space.n
	model = vpg_model(number_observation_features, number_actions)

	if load_model:
		if algo == 'policy_gradient':
			model.load_state_dict(torch.load('saved_MODELS/best_model_policy_gradient.pth'))
		elif algo == 'policy_gradient_center':
			model.load_state_dict(torch.load('saved_MODELS/best_model_policy_gradient_center.pth'))
		elif algo == 'reinforce':
			model.load_state_dict(torch.load('saved_MODELS/best_model_reinforce.pth'))

	print(number_observation_features, number_actions)

	optimizer = Adam(model.parameters(), 1e-2)

	best_return = -sys.maxsize

	for epoch in range(epochs):
		if algo == 'policy_gradient':
			average_return = policy_gradient(env, model, optimizer)
		elif algo == 'policy_gradient_center':
			average_return = policy_gradient_center(env, model, optimizer)
		elif algo == 'reinforce':
			average_return = reinforce(env, model, optimizer)

		if average_return > best_return:
			best_return = average_return
			if algo == 'policy_gradient':
				torch.save(model.state_dict(), 'saved_MODELS/best_model_policy_gradient.pth')
			elif algo == 'policy_gradient_center': 
				torch.save(model.state_dict(), 'saved_MODELS/best_model_policy_gradient_center.pth')
			elif algo == 'reinforce':
				torch.save(model.state_dict(), 'saved_MODELS/best_model_reinforce.pth')
			print('Best model updated!')
		print('epoch: %3d \t return: %.3f' % (epoch, average_return))

def vpg_evaluate(load_model = False, algo = 'reinforce', gym_environment = 'CartPole-v1'):

	env = gym.make(gym_environment, render_mode = 'human')

	if load_model == True:

		number_observation_features = env.observation_space.shape[0]
		number_actions = env.action_space.n
		model = vpg_model(number_observation_features, number_actions)

		if algo == 'policy_gradient':
			model.load_state_dict(torch.load('saved_MODELS/best_model_policy_gradient.pth'))
		elif algo == 'policy_gradient_center':
			model.load_state_dict(torch.load('saved_MODELS/best_model_policy_gradient_center.pth'))
		elif algo == 'reinforce':
			model.load_state_dict(torch.load('saved_MODELS/best_model_reinforce.pth'))

		model.eval()

		observation, _ = env.reset()

		total_reward = 0.0

		terminated, truncated = False, False

		while not terminated and not truncated:
			env.render()

			policy = get_policy(model, observation)

			action, log_probability_action = get_action(policy)

			observation, reward, terminated, truncated , _ = env.step(action)

			total_reward += reward

		print('Total reward is: %.3f' % (total_reward))
	else:
		env.reset()

		total_reward = 0.0

		terminated, truncated = False, False

		while not terminated and not truncated:
			env.render()

			observation, reward, terminated, truncated, _ = env.step(env.action_space.sample())

			total_reward += reward

		print('Total reward is: %.3f' % (total_reward))

	env.close()


