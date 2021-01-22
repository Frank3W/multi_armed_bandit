"""Module for multi-armed bandit"""

import random


class Bandit:
    """Abstract class for bandit"""
    def pull(self, k):
        raise NotImplementedError('Not implemented.')

    def best_reward(self):
        raise NotImplementedError('Not implemented.')

    def num_machines(self):
        raise NotImplementedError('Not implemented.')


class BernoulliBandit(Bandit):
    """Bernouli Multi-armed bandit"""

    def __init__(self, n, means):
        if len(means) != n:
            raise ValueError('Number of means must be same as n.')

        if n <= 0:
            raise ValueError('n must be a positive integer.')

        self.n = n
        self.means = [mu for mu in means]

    def pull(self, k):
        """Pulls kth machine and gets rewards.

        Parameters
        ----------
            k: int
                gives which machine to pull

        Return
        ------
            int:
                reward 0 or 1
        """
        if k != int(k) or k < 0 or k >= self.n:
            raise ValueError('k must be a positive integer smaller than {}'.format(self.n))

        rand_val = random.random()
        if rand_val <= self.means[k]:
            return 1
        else:
            return 0

    def best_reward(self):
        """Returns the best expected reward.

        Return
        ------
            float
        """
        return max(self.means)

    def num_machines(self):
        """Gets number of the machines"""
        return self.n