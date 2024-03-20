import os
import gymnasium as gym
from time import sleep
import torch
import torch.nn as nn
import sys
import numpy as np
from typing import Tuple
from torch.distributions.categorical import Categorical
from torch.optim import Adam, Optimizer
import argparse

current_dir = os.getcwd()
sys.path.append(current_dir + '/../')

from Algos import reinforce, policy_gradient
from Algos import get_policy, get_action

def create_model(num_obs, num_actions) -> nn.Module:
	#create the MLP model
	#output logits

	hidden_layer_features = 32

	return nn.Sequential(nn.Linear(in_features = num_obs,
																 out_features = hidden_layer_features),
											 nn.ReLU(),
											 nn.Linear(in_features = hidden_layer_features,
											 					 out_features = num_actions)
											)

def train(epochs = 1000, load_model = False, algo = 'reinforce') -> None:
	env = gym.make('CartPole-v1')

	#torch.manual_seed(0)
	#env.seed(0)

	number_observation_features = env.observation_space.shape[0]
	number_actions = env.action_space.n
	model = create_model(number_observation_features, number_actions)

	if load_model:
		if algo == 'policy_gradient':
			model.load_state_dict(torch.load('saved_MODELS/best_model_policy_gradient.pth'))
		elif algo == 'reinforce':
			model.load_state_dict(torch.load('saved_MODELS/best_model_reinforce.pth'))

	print(number_observation_features, number_actions)

	optimizer = Adam(model.parameters(), 1e-2)

	best_return = -sys.maxsize

	for epoch in range(epochs):
		if algo == 'policy_gradient':
			average_return = policy_gradient(env, model, optimizer)
		elif algo == 'reinforce':
			average_return = reinforce(env, model, optimizer)

		if average_return > best_return:
			best_return = average_return
			if algo == 'policy_gradient':
				torch.save(model.state_dict(), 'saved_MODELS/best_model_policy_gradient.pth')
			elif algo == 'reinforce':
				torch.save(model.state_dict(), 'saved_MODELS/best_model_reinforce.pth')
			print('Best model updated!')
		print('epoch: %3d \t return: %.3f' % (epoch, average_return))

def evaluate(load_model = False, algo = 'reinforce'):

	env = gym.make('CartPole-v1', render_mode = 'human')

	if load_model == True:

		number_observation_features = env.observation_space.shape[0]
		number_actions = env.action_space.n
		model = create_model(number_observation_features, number_actions)

		if algo == 'policy_gradient':
			model.load_state_dict(torch.load('saved_MODELS/best_model_policy_gradient.pth'))
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

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description = 'Running Commands')
	parser.add_argument('--mode')
	parser.add_argument('--model')
	parser.add_argument('--algo')

	args = parser.parse_args()

	if args.mode == 'train':
		if args.model == 'random':
			train(load_model = False, algo = args.algo)
		else:
			train(load_model = True, algo = args.algo)
	else:
		if args.model == 'random':
			evaluate(load_model = False, algo = args.algo)
		else:
			evaluate(load_model = True, algo = args.algo)

