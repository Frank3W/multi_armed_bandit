""" Module for running environment"""

from .bandit import Bandit

class IndEnv:
    """Independent Running Environment for Strategies.
    """

    def __init__(self, rounds, bandit):
        if rounds <= 0 or rounds != int(rounds):
            raise ValueError('rounds must be a positive integer.')
        self.rounds = rounds

        if not isinstance(bandit, Bandit):
            raise ValueError('bandit must be an object of class Bandit.')
        self.bandit = bandit
        self.num_machines = self.bandit.num_machines()
        self.strategy = {}
        self.result = {}

    def strategy_register(self, strategy, strategy_name):
        if strategy_name not in self.strategy:
            self.strategy[strategy_name] = strategy
        else:
            raise ValueError('{} already exists.'.format(strategy_name))


    def clear_strategy(self):
        self.strategy = {}
        self.result = {}


    def run(self):
        if len(self.strategy) == 0:
            return

        config = {}
        config['rounds'] = self.rounds
        config['num_machines'] = self.num_machines

        history = {}

        for i in range(self.rounds):
            if i == 0:
                for name in self.strategy:
                    history[name] = {}
                    history[name]['reward'] = []
                    history[name]['selection'] = []

            curr_selection_dict = {}

            for name, strategy in self.strategy.items():
                curr_selection_dict[name] = strategy(history[name], config)

            # ensure same pull would return same reward
            selection2reward_dict = {}
            for name in curr_selection_dict:
                curr_selection = curr_selection_dict[name]

                if curr_selection not in selection2reward_dict:
                    selection2reward_dict[curr_selection] = self.bandit.pull(curr_selection)

            for name in self.strategy:
                history[name]['reward'].append(selection2reward_dict[curr_selection_dict[name]])
                history[name]['selection'].append(curr_selection_dict[name])

        self.result = history
