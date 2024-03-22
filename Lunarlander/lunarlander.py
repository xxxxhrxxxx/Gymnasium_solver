import os
import sys
import gymnasium as gym
from time import sleep
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from torch.distributions.categorical import Categorical
from torch.optim import Adam, Optimizer
import argparse

current_dir = os.getcwd()
sys.path.append(current_dir + '/../')

from Algos import reinforce, policy_gradient, policy_gradient_center
from Algos import get_policy, get_action, vpg_model
from Algos import vpg_train, vpg_evaluate

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Running Commands')
	parser.add_argument('--mode')
	parser.add_argument('--model')
	parser.add_argument('--algo')

	args = parser.parse_args()

	if args.mode == 'train':
		if args.model == 'random':
			vpg_train(load_model = False, algo = args.algo, gym_environment = 'LunarLander-v2')
		else:
			vpg_train(load_model = True, algo = args.algo, gym_environment = 'LunarLander-v2')
	else:
		if args.model == 'random':
			vpg_evaluate(load_model = False, algo = args.algo, gym_environment = 'LunarLander-v2')
		else:
			vpg_evaluate(load_model = True, algo = args.algo, gym_environment = 'LunarLander-v2')

