""" Module for multi-armed bandit strategy """

import random


def gen_random(history, config):
    return random.randrange(config['num_machines'])

