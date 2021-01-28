""" Module for running environment"""

import pandas as pd
import numpy as np
import altair as alt

from .bandit import Bandit

alt.renderers.set_embed_options(actions=False)

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
                    history[name]['best_reward'] = []

            curr_selection_dict = {}

            for name, strategy in self.strategy.items():
                curr_selection_dict[name] = strategy(history[name], config)

            # ensure same pull would return same reward
            selection2reward_dict = {}
            for name in curr_selection_dict:
                curr_selection = curr_selection_dict[name]

                if curr_selection not in selection2reward_dict:
                    selection2reward_dict[curr_selection] = self.bandit.pull(curr_selection)

            best_reward = self.bandit.best_reward()

            for name in self.strategy:
                history[name]['reward'].append(selection2reward_dict[curr_selection_dict[name]])
                history[name]['selection'].append(curr_selection_dict[name])
                history[name]['best_reward'].append(best_reward)

        self.result = history


    def altair_chart_reward(self, names=None):
        """Gets altair chart of reward.

        Parameters
        ----------
            names: None or list of strings
                strategies kept for chart.

        Return
        ------
            altair chart
        """
        reward_dict = {}
        for strategy_name in self.result.keys():
            if names is None or strategy_name in names:
                reward_dict[strategy_name] = np.cumsum(self.result[strategy_name]['reward'])

        reward_df = pd.DataFrame(reward_dict)

        # only around 100 points for chart
        selected_indices = np.array(np.arange(0, reward_df.shape[0], reward_df.shape[0]//100))
        if reward_df.shape[0] - 1 not in selected_indices:
            selected_indices = np.append(selected_indices, reward_df.shape[0]-1)
        reward_df = reward_df.iloc[selected_indices]

        reward_df['step'] = reward_df.index
        reward_df_melt = reward_df.melt(id_vars=['step'], var_name='strategy', value_name='reward')
        chart_plot = alt.Chart(reward_df_melt).mark_line().encode(x='step:Q', y='reward:Q', color="strategy:N")
        return chart_plot


    def altair_chart_regret(self, names=None):
        """Gets altair chart of regret.

        Parameters
        ----------
            names: None or list of strings
                strategies kept for chart.

        Return
        ------
            altair chart
        """
        regret_dict = {}
        for strategy_name in self.result.keys():
            if names is None or strategy_name in names:
                total_reward = np.cumsum(self.result[strategy_name]['reward'])
                total_best_reward = np.cumsum(self.result[strategy_name]['best_reward'])
                regret_dict[strategy_name] = total_best_reward  - total_reward


        regret_df = pd.DataFrame(regret_dict)

        # only around 100 points for chart
        selected_indices = np.array(np.arange(0, regret_df.shape[0], regret_df.shape[0]//100))

        # include last step
        if regret_df.shape[0] - 1 not in selected_indices:
            selected_indices = np.append(selected_indices, regret_df.shape[0]-1)
        regret_df = regret_df.iloc[selected_indices]

        regret_df['step'] = regret_df.index

        regret_df_melt = regret_df.melt(id_vars=['step'], var_name='strategy', value_name='regret')
        chart_plot = alt.Chart(regret_df_melt).mark_line().encode(x='step:Q', y='regret:Q', color="strategy:N")
        return chart_plot

    def altair_action_history(self, name):
        """Gets altair chart of action history.

        Parameters
        ----------
            name: string
                strategy name

        Return
        ------
            altair chart
        """

        if name not in self.result.keys():
            raise Exception('Name must be one of {}'.format(list(self.result.keys())))

        selection_list = self.result[name]['selection']

        selection_matrix = np.zeros((len(selection_list), self.num_machines), dtype='int')

        for idx, selection in enumerate(selection_list):
            selection_matrix[idx, selection] = 1

        selection_prop_dict = {}

        for idx in range(selection_matrix.shape[1]):
            selection_prop_dict['action {}'.format(idx)] = np.cumsum(selection_matrix[:, idx])/np.arange(1, selection_matrix.shape[0] + 1)


        selection_prop_df = pd.DataFrame(selection_prop_dict)

        # only around 100 points for chart
        selected_indices = np.array(np.arange(0, selection_prop_df.shape[0], selection_prop_df.shape[0]//100))

        # include last step
        if selection_prop_df.shape[0] - 1 not in selected_indices:
            selected_indices = np.append(selected_indices, selection_prop_df.shape[0]-1)
        selection_prop_df = selection_prop_df.iloc[selected_indices]

        selection_prop_df['step'] = selection_prop_df.index
        selection_prop_df_melt = selection_prop_df.melt(id_vars=['step'], var_name='selection', value_name='proportion')

        chart_plot = alt.Chart(selection_prop_df_melt).mark_line().encode(x='step:Q', y='proportion:Q', color='selection')
        return chart_plot
