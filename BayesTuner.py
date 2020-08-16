import numpy as np
import copy
import timeit
import os
from itertools import chain

from ..tuner import Tuner
from tvm import autotvm

from BayesOptimizer import BayesianOptimizer
from MetaBO import MetaBO

import multiprocessing

__USE_METABO__ = False

### The following macros are used to determine the structure of the system

## measure time is used for performance analysis

__measure_time__ = True
global __USE_CPP_BACKEND__
__USE_CPP_BACKEND__ = False

### when feature copression is enabled, we require, features space needs less memory.
__compress_features__ = False

if __USE_CPP_BACKEND__:
    try:
        import BayesOptimizer_wrapper

        print("Using C++ backend...")
    except:
        print("Loading C++ backend failed")
        __USE_CPP_BACKEND__ = False


class BayesWrapper(object):
    """
    The upper level class for Bayesian optimization tunner.
    Recieves a set of tasks with a set of tunning options, optimizes those tasks, stores the best results in a log file.


    Parameters
    ----------
    n_trial: number of trials for each tasks, this number refers to
             how many different configurations are tested on real hardware.

    log_filename: the file in which we store best results

    early_stopping: threshold for stopping the search earlier than N-trials

    use_transfer_learning: True/False, determines whether we will use TL or not.

    Attributes
    -----------
    tunner: A bayesian tunner, see the class inside BayesianOptimizer.py

    best_results: best observation for each task, used in determining the variance of
                  performance between tasks which is usefull in setting a variable number
                  for #traials in each task.

    sampling_points: keep track of tested points for every task

    stop_threshold: determines the maximum allowed variance among tasks performance
                    for stopping the algorithm.

    measure_option, callbacks: used inside the tuner class


    Fields
    ----------------------------------------------
    tasks

    """

    def __init__(self,
                 tasks,
                 measure_option,
                 n_trial=1000,
                 early_stopping=False,
                 log_filename='tuning.log',
                 use_transfer_learning=False,
                 stop_threshold=5e11,
                 call_back=None,
                 tuner=None,
                 metric=None
                 ):

        self.tasks = tasks
        self.n_trials = n_trial
        self.log_filename = log_filename
        self.early_stopping = early_stopping
        self.use_transfer_laerning = use_transfer_learning

        self.tunner = BayesianTuner(metric=metric)
        self.best_results = np.zeros(len(tasks))
        self.sampling_points = []
        self.stop_threshold = stop_threshold
        self.measure_option = measure_option
        self.call_back = call_back

        self.metric = metric


    def tune(self):
        """
        iterates over tasks, tunes each one, store the result.

        :return:

        """

        ### init the environment...
        tmp_log_file = self.log_filename + ".tmp"
        if os.path.exists(tmp_log_file):
            os.remove(tmp_log_file)

        ### set the tunning budget
        budget = self.n_trials * len(self.tasks)
        for i, task in enumerate(reversed(self.tasks)):
            self.tunner.reset(task, use_transfer_learning=self.use_transfer_laerning)
            prefix = "[Task %2d/%2d] " % (i + 1, len(self.tasks))
            self.best_results[i], samples, trials = self.tunner.tune(n_trial=self.n_trials * 1,
                                                                     measure_option=self.measure_option,
                                                                     callbacks=[
                                                                         autotvm.callback.progress_bar(
                                                                             self.n_trials * 1,
                                                                             prefix=prefix),
                                                                         autotvm.callback.log_to_file(tmp_log_file)]

                                                                     )
            self.sampling_points.append(samples)
            budget -= trials

        print(np.std(self.best_results))

        while np.std(self.best_results) > 0 and budget > 0:
            next_point = np.argmin(self.best_results)
            task = self.tasks[len(self.tasks) - np.argmin(self.best_results) - 1]
            self.tunner.reset(task, use_transfer_learning=self.use_transfer_laerning,
                              data=self.sampling_points[next_point])
            prefix = "[Task %2d/%2d] " % (next_point + 1, len(self.tasks))
            result, samples, trials = self.tunner.tune(n_trial=self.n_trials,
                                                       measure_option=self.measure_option,
                                                       callbacks=[
                                                           autotvm.callback.progress_bar(self.n_trials,
                                                                                         prefix=prefix),
                                                           autotvm.callback.log_to_file(tmp_log_file)]
                                                       )
            self.sampling_points[next_point] = samples
            budget -= trials
            self.best_results[next_point] = max(self.best_results[next_point], result)

        autotvm.record.pick_best(tmp_log_file, self.log_filename)
        os.remove(tmp_log_file)


class BayesianTuner(Tuner):
    """
    Subclass of Tuner implemented by TVM. It receives a task,
    select next samples, evaluate the results, and update the
    underlying model.

    :parameter
    -------------
    use_transfer_learning: Boolean, determines whether to use TL or not.

    training_intervals: determines after how many rounds, the underlying GP is updated

    log_interval: logging interval

    target: target platform, cuda, vta, cpu, etc.

    config_space: configuration space inside the task

    config_space_len: len of each axis of the configuration space

    trials: points which will be tested on bare hardware

    visisted: a set of visited opints

    xs: tested configurations so far
    ys: the result of xs so far

    flop_max: best result achieved so far

    training_epoch:


    Attributes
    --------------
    bayesian_optimizer: underlying optimizer class, it contains... #TODO

    feature_map: mapping from indices in configuration space to actual configuration



    """

    def __init__(self,
                 task=None,
                 training_interval=16,
                 log_interval=50,
                 use_transfer_learning=False,
                 metric=None
                 ):
        super(BayesianTuner, self).__init__(task)
        self.use_transfer_learning = use_transfer_learning
        self.training_intervals = training_interval
        self.log_intervals = log_interval
        self.traget = None

        self.config_space = None
        self.config_space_len = None

        self.trials = []
        self.trial_pt = 0
        self.visited = set()

        self.xs = []
        self.ys = []
        self.flops_max = 0.0
        self.training_epoch = 0

        if __USE_METABO__:
            self.bayesian_optimizer = None
        else:
            self.bayesian_optimizer = BayesianOptimizer(metric=metric)
        self.feature_map = []

        self.num_threads = 24
        self.pool = None
        self.metric = metric

    def _close_pool(self):
        if self.pool:
            self.pool.terminate()
            self.pool.join()
            self.pool = None

    def _reset_pool(self):
        self._close_pool()
        global _config_space, _config_space_dims, _config_space_maximums
        _config_space = self.config_space
        _config_space_dims = []
        _config_space_maximums = []
        for key, tile in self.config_space.space_map.items():
            _config_space_dims.append(tile.num_output)
            if _config_space_dims[-1] == 0:
                _config_space_dims[-1] = 1
            if hasattr(tile, 'product'):
                _config_space_maximums.append([tile.product] * len(tile.entities[-1].size))
            else:
                _config_space_maximums.append([max([tile.entities[-1].val])])
        _config_space_maximums = list(chain.from_iterable(_config_space_maximums))
        self.pool = multiprocessing.Pool(self.num_threads)

    def reset(self, task, data=None, use_transfer_learning=False):
        super(BayesianTuner, self).__init__(task)
        self.task = task
        self.use_transfer_learning = use_transfer_learning
        self.target = task.target
        self.config_space = task.config_space
        self.config_space_len = [len(x) for x in self.config_space.space_map.values()]
        self.trials = []
        self.trial_pt = 0
        self.visited = set()
        self.xs = []
        self.ys = []
        self.flops_max = 0.0
        self.training_epoch = 0

        if data is not None:
            self.xs = list(data[0])
            self.ys = list(data[1])
            self.flops_max = max(self.ys)
            print(len(self.xs))

        self.feature_map = None

        if __measure_time__:
            start = timeit.default_timer()
        self._reset_pool()
        ### Minimize index map size
        self.feature_map = []
        indices = np.arange(np.prod(self.config_space_len))

        if __USE_METABO__:
            # if self.bayesian_optimizer is None:
            self.bayesian_optimizer = MetaBO(config=self.task.config_space.space_map)
            # else:
            #     self.bayesian_optimizer = MetaBO(config=self.task.config_space.space_map,
            #                                      PPOModel=self.bayesian_optimizer.get_base_model())
        elif not __USE_CPP_BACKEND__:
            self.feature_map = self.pool.map(_get_config, indices)

        else:
            self.bayesian_optimizer.set_map(self.config_space.space_map)

        if __measure_time__:
            stop = timeit.default_timer()
            print('Time: ', stop - start)

    def next_batch(self, batch_size):
        ret = []
        counter = 0
        while counter < batch_size:
            if len(self.visited) >= len(self.config_space):
                break

            while self.trial_pt < len(self.trials):
                index = self.trials[self.trial_pt]
                if index not in self.visited:
                    break
                self.trial_pt += 1

            if self.trial_pt >= len(self.trials) - int(0.05 * self.training_intervals):
                # if the trial list is empty or
                # the tuner is doing the last 5% trials (e-greedy), choose randomly
                index = np.random.randint(len(self.config_space))
                while index in self.visited:
                    index = np.random.randint(len(self.config_space))

            ret.append(self.config_space.get(index))
            self.visited.add(index)

            counter += 1
        return ret

    def update(self, inputs, results):
        for inp, res in zip(inputs, results):
            index = inp.config.index
            if res.error_no == 0:
                self.xs.append(index)
                flops = inp.task.flop / np.mean(res.costs)
                self.flops_max = max(self.flops_max, flops)
                self.ys.append(flops)
            else:
                # self.xs.append(index)
                # self.ys.append(0.0)
                pass
                ## TODO

        xs_configurations = []
        update_size = len(self.xs)
        if __USE_CPP_BACKEND__:
            xs_configurations = self.xs
        else:
            for x in self.xs[-min(len(self.xs), update_size):]:
                if len(self.feature_map) == np.prod(self.config_space_len):
                    xs_configurations.append(self.feature_map[int(x)])
                else:
                    xs_configurations.append(np.asarray(self.config_space.get(x).get_flatten_feature()))

        if len(self.xs) >= self.training_intervals * (self.training_epoch + 1) \
                and self.flops_max > 1e-6:

            self.bayesian_optimizer.fit(xs_configurations, self.ys[-min(len(self.xs), update_size):])
            if not __USE_METABO__:
                visited_map = [self.feature_map[index] for index in self.visited]
                maxes = self.bayesian_optimizer.next_batch(visited=visited_map, visited_indexes=self.visited,
                                                           batch_size=self.training_intervals,
                                                           test_points=self.feature_map,
                                                           visited_results=self.ys)
            else:
                maxes = self.bayesian_optimizer.next_batch(batch_size=self.training_intervals,
                                                           config_space=self.config_space,
                                                           visited=self.visited)
            self.training_epoch += 1
            self.trials = maxes
            self.trial_pt = 0

        ## delete shallow copied configs
        del xs_configurations


    def has_next(self):
        return len(self.visited) < len(self.config_space)

    def tune(self, *args, **kwargs):
        super(BayesianTuner, self).tune(*args, **kwargs)
        return (self.flops_max / self.config_space.flop), [self.xs, self.ys], self.n_trial


_config_space = None
_config_space_dims = None
_config_space_maximums = None


### get the fallten features of configuration space from the globaled congi space
### seperated to be maped in a pool of threads

def _get_config(index):
    def dtype(num):
        bits = len(bin(int(np.max(np.abs(num))))[2:])
        if bits == 1:
            return 'bool'
        elif bits < 8:
            return 'int8'
        elif bits < 16:
            return 'int16'
        return 'int32'

    config = _config_space.get(index).get_flatten_feature()
    if __compress_features__:
        return np.asarray(compress_config(config))
    # config = config / _config_space_maximums
    config = [x for x in config if x >= 0]
    return np.asarray(config)


pow2 = [1, 1024, 1024 * 1024, 1024 * 1024 * 1024, 1024 ** 4]


def compress_config(config):
    _config = []
    g_index = 0
    for index in _config_space_dims:
        _config.append(np.sum([config[i] * pow2[i - g_index] for i in range(g_index, g_index + index)]))
        g_index += index
    return _config

### Input : tasks.config_space.space_map
### output: a sorted dict from axises to possible values for each axis - purpose is passing to cython code
