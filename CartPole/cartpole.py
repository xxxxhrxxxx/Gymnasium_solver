import gymnasium as gym
from time import sleep
import torch
import torch.nn as nn
import sys
import numpy as np
from typing import Tuple
from torch.distributions.categorical import Categorical
from torch.optim import Adam, Optimizer

def create_model(num_obs, num_actions) -> nn.Module:
	#create the MLP model

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

def train_one_epoch(env: gym.Env,
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
	
	epoch_loss = calculate_loss(torch.stack(epoch_log_probability_actions),
															torch.as_tensor(epoch_action_rewards, dtype=torch.float32))

	epoch_loss.backward()
	optimizer.step()
	optimizer.zero_grad()

	return float(np.mean(epoch_returns))


def train(epochs = 1000, load_model = False) -> None:
	env = gym.make('CartPole-v1')

	#torch.manual_seed(0)
	#env.seed(0)

	number_observation_features = env.observation_space.shape[0]
	number_actions = env.action_space.n
	model = create_model(number_observation_features, number_actions)

	if load_model:
		model.load_state_dict(torch.load('saved_MODELS/best_model.pth'))

	print(number_observation_features, number_actions)

	optimizer = Adam(model.parameters(), 1e-3)

	best_return = -sys.maxsize

	for epoch in range(epochs):
		average_return = train_one_epoch(env, model, optimizer)

		if average_return > best_return:
			best_return = average_return
			torch.save(model.state_dict(), 'saved_MODELS/best_model.pth')
			print('Best model updated!')
		print('epoch: %3d \t return: %.3f' % (epoch, average_return))

def evaluate(random = False):

	env = gym.make('CartPole-v1', render_mode = 'human')

	if random == False:

		number_observation_features = env.observation_space.shape[0]
		number_actions = env.action_space.n
		model = create_model(number_observation_features, number_actions)

		model.load_state_dict(torch.load('saved_MODELS/best_model.pth'))

		model.eval()

		observation, _ = env.reset()

		while True:
			env.render()

			policy = get_policy(model, observation)

			action, log_probability_action = get_action(policy)

			observation, reward, done, _ , _ = env.step(action)

			if done: break
	else:
		env.reset()

		while True:
			env.render()
			observation, reward, done, _, _ = env.step(env.action_space.sample())

			if done: break
	env.close()

if __name__ == '__main__':

	train_flag = sys.argv[1]
	random_flag = sys.argv[2]

	if train_flag == 'train':
		train(load_model = True)
	else:
		if random_flag == 'random':
			evaluate(random = True)
		else:
			evaluate(random = False)

sys.exit(1)


env = gym.make("CartPole-v1", render_mode='human')
env.reset()
#observation, info = env.reset(seed=42)
for _ in range(5000):
	env.render()
	action = env.action_space.sample()
	observation, reward, terminated, truncated, info = env.step(action)

	if terminated or truncated:
		observation, info = env.reset()
env.close()
