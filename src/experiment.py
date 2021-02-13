
class Experiment:
    def __init__(self, run_env):
        self.run_env = run_env


    def get_reward_selection(self, num_runs):
        result_dict = {}

        for i in range(num_runs):
            self.run_env.run()
            for strategy_name in self.run_env.result:
                if i == 0:
                    result_dict[strategy_name] = {}
                    result_dict[strategy_name]['reward'] = []
                    result_dict[strategy_name]['selection'] = []

                result_dict[strategy_name]['reward'].append(self.run_env.result[strategy_name]['reward'])
                result_dict[strategy_name]['selection'].append(self.run_env.result[strategy_name]['selection'])

        return result_dict