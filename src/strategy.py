""" Module for multi-armed bandit strategy """

import random

import numpy as np

def random_strategy(history, config):
    return random.randrange(config['num_machines'])

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
            
            return np.argmax(utility_list)


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
