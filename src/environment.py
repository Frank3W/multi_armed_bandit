""" Module for running environment"""

class Env:
    def __init__(self, rounds, bandit):
        self.rounds = rounds
        self.bandit = bandit
        self.strategy = {}
        self.result = {}

    def strategy_register(self, strategy, strategy_name):
        if strategy_name not in self.strategy:
            self.strategy[strategy_name] = strategy
        else:
            raise ValueError('{} already exists.'.format(strategy_name))


    def run(self):
        config = {}
        config['rounds'] = self.rounds
        config['num_machines'] = self.bandit.num_machines()

        for name, strategy in self.strategy.items():
            rewards = []
            selections = []

            for i in range(self.rounds):
                history = {}
                history['rewards'] = [ _ for _ in rewards]
                history['selections'] = [ _ for _ in selections]
                curr_selection = strategy(history, config)
                curr_reward = self.bandit.pull(curr_selection)

                rewards.append(curr_reward)
                selections.append(curr_selection)

            self.result[name] = {}
            self.result[name]['rewards'] = rewards
            self.result[name]['selections'] = selections
