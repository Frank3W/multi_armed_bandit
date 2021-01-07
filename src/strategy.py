""" Module for multi-armed bandit strategy """

import random

import numpy as np

def random_strategy(history, config):
    return random.randrange(config['num_machines'])


class EpsilonGreedy:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def strategy(self, history, config):
        if len(history['reward']) == 0:
            return random.randrange(config['num_machines'])
        else:
            rand_val = random.uniform(0, 1)
            if rand_val < self.epsilon:
                return random.randrange(config['num_machines'])
            else:
                reward_dict = {}
                for i in range(len(history['reward'])):
                    reward = history['reward'][i]
                    selection = history['selection'][i]

                    if selection not in reward_dict:
                        reward_dict[selection] = []

                    reward_dict[selection].append(reward)

                reward_avg_dict = {}

                for selection, reward_list in reward_dict.items():
                    reward_avg_dict[selection] = np.mean(reward_list)

                reward_avg_pair = sorted(reward_avg_dict.items(), key=lambda x: -x[1])

                best_selection = reward_avg_pair[0][0]
                return best_selection
