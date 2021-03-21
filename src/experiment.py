import joblib
from joblib import Parallel, delayed

class Experiment:
    def __init__(self, run_env):
        self.run_env = run_env

    def one_run(self):
        self.run_env.run()
        return self.run_env.result


    def multiple_run(self, num_runs, n_jobs=1):

        result = Parallel(n_jobs=n_jobs)(delayed(self.one_run)() for i in range(num_runs))
        return result
