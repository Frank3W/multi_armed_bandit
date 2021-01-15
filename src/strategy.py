""" Module for multi-armed bandit strategy """

import random

import numpy as np
from scipy.stats import beta

def random_strategy(history, config):
    return random.randrange(config['num_machines'])

class BaysianUCB:
    """Baysian Upper Confidence Bound multi-armed bandit strategy.

    The prior is a beta distribution. The class only supports binary reward 0 or 1.
    """

    def __init__(self, n_std=3, init_alpha=1, init_beta=1):
        if n_std <= 0:
            raise ValueError('n_std must be positive.')

        self.n_std = n_std
        self.init_alpha = init_alpha
        self.init_beta = init_beta

    def strategy(self, history, config):
        n_arms = config['num_machines']

        if len(history['reward']) == 0:
            return random.randrange(n_arms)
        else:
            alpha_list = [self.init_alpha] * n_arms
            beta_list = [self.init_beta] * n_arms

            for selection, reward in zip(history['selection'], history['reward']):
                if reward not in [0, 1]:
                    raise ValueError('reward must be 0 or 1: {}'.format(reward))

                alpha_list[selection] += reward
                beta_list[selection] += 1 - reward

            utility_list = [(alpha_list[i]/(alpha_list[i] + beta_list[i]) +
                            self.n_std * beta.std(alpha_list[i], beta_list[i]))
                            for i in range(n_arms)]

            return int(np.argmax(utility_list))


class UCB:
    """Upper Confidence Bound multi-armed bandit strategy.
    """
    def __init__(self, coef, init_action_val = np.inf, init_den_offset=1):
        if coef < 0:
            raise ValueError('Coef must be ')
        self.coef = coef
        self.init_action_val = init_action_val
        self.init_den_offset = init_den_offset

    def strategy(self, history, config):
        if len(history['reward']) == 0:
            return random.randrange(config['num_machines'])
        else:
            pre_cnt = len(history['reward'])
            reward_dict = {}

            for i in range(pre_cnt):
                reward = history['reward'][i]
                selection = history['selection'][i]

                if selection not in reward_dict:
                    reward_dict[selection] = []

                reward_dict[selection].append(reward)

            utility_list = [self.init_action_val + self.coef * np.sqrt(np.log(pre_cnt)/self.init_den_offset)] * config['num_machines']

            for selection, reward_list in reward_dict.items():
                utility_list[selection] = (np.mean(reward_list) +
                                           self.coef * np.sqrt(np.log(pre_cnt)/(len(reward_list) + self.init_den_offset)))

            return int(np.argmax(utility_list))


class EpsilonGreedy:
    """Epsilong-greedy multi-armed bandit strategy.
    """
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
