import numpy as np


class Experiment:
    def __init__(self, run_env):
        self.run_env = run_env

    def get_mean(self, num_runs):
        avg_reward_dict = {}
        for i in range(num_runs):
            self.run_env.run()
            for strategy_name in self.run_env.result:
                avg_reward = np.mean(self.run_env.result[strategy_name]['reward'])
                if i == 0:
                    avg_reward_dict[strategy_name] = [avg_reward]
                else:
                    avg_reward_dict[strategy_name].append(avg_reward)

        return avg_reward_dict