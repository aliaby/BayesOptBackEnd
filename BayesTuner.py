import numpy as np
from ..tuner import Tuner
import os
import multiprocessing
from scipy.stats import norm
from tvm import autotvm
from warnings import catch_warnings
from warnings import simplefilter
from ..DebugLogger import DebugLogger
import sys
from memory_profiler import profile
from sklearn.gaussian_process import GaussianProcessRegressor
import copy
import timeit

__measure_time__ = True

class BayesWrapper(object):

    def __init__(self,
                 tasks,
                  measure_option,
                  n_trial=1000,
                  early_stopping=False,
                  log_filename='tuning.log',
                  use_transfer_learning=False,
                  stop_threshold=5e11,
                  call_back = None,
                  tuner=None
                 ):
        
        self.tasks = tasks
        self.n_trials = n_trial
        self.log_filename = log_filename
        self.early_stopping = early_stopping
        self.use_transfer_laerning = use_transfer_learning
        self.tunner = BayesianTuner()
        self.best_results = np.zeros(len(tasks))
        self.sampling_points = []
        self.stop_threshold = stop_threshold
        self.measure_option = measure_option
        self.call_back = call_back

    def tune(self):
        ### init the environment...
        tmp_log_file = self.log_filename + ".tmp"
        if os.path.exists(tmp_log_file):
            os.remove(tmp_log_file)
        ### set the tunning budget
        budget = self.n_trials*len(self.tasks)
        for i, task in enumerate(reversed(self.tasks)):
            self.tunner.reset(task, use_transfer_learning=self.use_transfer_laerning)
            prefix = "[Task %2d/%2d] " % (i + 1, len(self.tasks))
            self.best_results[i], samples, trials = self.tunner.tune(n_trial=self.n_trials,
                                                                     measure_option=self.measure_option,
                                                                     callbacks=[autotvm.callback.progress_bar(self.n_trials, prefix=prefix),
                                                                                autotvm.callback.log_to_file(tmp_log_file)]
                                                                    )
            self.sampling_points.append(samples)
            budget -= trials

        ### second runs
        # while np.max(self.best_results) - np.min(self.best_results) > self.stop_threshold \
        #     and budget > 0:
        #
        #     task, index = self.tasks[np.argmin(self.best_results)], np.argmin(self.best_results)
        #     prefix = "[Task %2d/%2d] " % (index + 1, len(self.tasks))
        #     self.tunner.reset(task, use_transfer_learning=self.use_transfer_laerning, data=self.sampling_points[index])
        #     new_result, samples, trials = self.tunner.tune(n_trial=self.n_trials,
        #                                                              measure_option=self.measure_option,
        #                                                              callbacks=[autotvm.callback.progress_bar(self.n_trials, prefix=prefix),
        #                                                                         autotvm.callback.log_to_file(tmp_log_file)]
        #                                                             )
        #     self.best_results[index] = max(self.best_results[index], new_result)
        #     self.sampling_points[index].append(samples)
        #
        #     budget -= trials

        autotvm.record.pick_best(tmp_log_file, self.log_filename)
        os.remove(tmp_log_file)





class BayesianTuner(Tuner):
    def __init__(self,
                 task=None,
                 training_interval=16,
                 log_interval=50,
                 use_transfer_learning=False,
                 ):
        super(BayesianTuner, self).__init__(task)
        self.use_transfer_learning=False
        self.training_intervals = training_interval
        self.log_intervals=log_interval
        self.traget=None
        self.space=None
        self.space_len = None

        self.trials = []
        self.trial_pt = 0
        self.visited = set()

        self.xs = []
        self.ys = []
        self.flops_max = 0.0
        self.training_epoch = 0

        self.bayesian_optimizer = BayesianOptimizer()
        self.index_map = []

        self.num_threads = 24
        self.pool = None

    def _close_pool(self):
        if self.pool:
            self.pool.terminate()
            self.pool.join()
            self.pool = None

    def _reset_pool(self):
        self._close_pool()
        global _config_space
        _config_space = self.space
        self.pool = multiprocessing.Pool(self.num_threads)

    def reset(self, task, data=None, use_transfer_learning=False):
        super(BayesianTuner, self).__init__(task)
        self.training_intervals = 16
        self.task = task
        self.use_transfer_learning = use_transfer_learning
        self.target = task.target
        self.space = task.config_space
        self.space_len = [len(x) for x in self.space.space_map.values()]
        self.trials = []
        self.trial_pt = 0
        self.visited = set()
        self.xs = []
        self.ys = []
        self.flops_max = 0.0
        self.training_epoch = 0
        if data is not None:
            self.xs = data[0]
            self.ys = data[1]
        self.index_map = None
        if __measure_time__:
            start = timeit.default_timer()
        self._reset_pool()
        ### Minimize index map size
        self.index_map = []
        # if np.prod(self.space_len) > 2**24:
        #     indices = np.random.choice(np.prod(self.space_len), 2**24, replace=False)
        # else:
        indices = np.arange(np.prod(self.space_len))

        self.index_map = self.pool.map(_get_config, indices)
        # self.index_map = np.asarray([np.asarray(self.space.get(index).get_flatten_feature(), dtype='int16') for index in np.arange(np.prod(self.space_len))])
        if __measure_time__:
            stop = timeit.default_timer()
            print('Time: ', stop - start)

        # self.bayesian_optimizer.reset()

    def next_batch(self, batch_size):
        ret = []
        counter = 0
        while counter < batch_size:
            if len(self.visited) >= len(self.space):
                break

            while self.trial_pt < len(self.trials):
                index = self.trials[self.trial_pt]
                if index not in self.visited:
                    break
                self.trial_pt += 1

            if self.trial_pt >= len(self.trials) - int(0.05 * self.training_intervals):
                # if the trial list is empty or
                # the tuner is doing the last 5% trials (e-greedy), choose randomly
                index = np.random.randint(len(self.space))
                while index in self.visited:
                    index = np.random.randint(len(self.space))

            ret.append(self.space.get(index))
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
                self.xs.append(index)
                self.ys.append(0.0)

        xs_configurations = []
        update_size = len(self.xs)
        for x in self.xs[-min(len(self.xs),update_size):]:
            if len(self.index_map) == np.prod(self.space_len):
                xs_configurations.append(copy.copy(self.index_map[x]))
            else:
                xs_configurations.append(np.asarray(self.space.get(x).get_flatten_feature()))

        if len(self.xs) >= self.training_intervals * (self.training_epoch + 1) \
                and self.flops_max > 1e-6:

            self.bayesian_optimizer.fit(xs_configurations, self.ys[-min(len(self.xs),update_size):])
            visited_map = [np.asarray(self.space.get(index).get_flatten_feature(), dtype='int16') for index in self.visited]
            maxes = self.bayesian_optimizer.next_batch(visited=visited_map, visited_indexes=self.visited, batch_size = self.training_intervals, test_points=self.index_map)
            self.training_epoch += 1
            self.trials = maxes
            self.trial_pt = 0

        ## delete shallow copied configs
        del xs_configurations
        if len(self.xs) % 128 == 0:
            self.training_intervals *= 2


    def has_next(self):
        return len(self.visited) < len(self.space)



    def tune(self, *args, **kwargs):
        super(BayesianTuner, self).tune(*args, **kwargs)

        ##TODO
        ## Return stufs
        return self.flops_max, [self.xs,self.ys], self.n_trial


class BayesianOptimizer(object):
    def __init__(self, run_multi_threaded=False):
        self.regressor = GaussianProcessRegressor(copy_X_train=False,normalize_y=True)
        if run_multi_threaded:
            self.num_threads = multiprocessing.cpu_count()
            self.pool = multiprocessing.Pool(self.num_threads)
        else:
            self.num_threads = 1
            self.pool = None

        self.max_test_points = 2**20 * self.num_threads
        self.random_points = 0.3
        self.decay = 0.95

    def reset(self):
        self.regressor = None
        self.regressor = GaussianProcessRegressor(copy_X_train=False,normalize_y=True)

    def _close_pool(self):
        if self.pool:
            self.pool.terminate()
            self.pool.join()
            self.pool = None

    def _reset_pool(self, visited):
        self._close_pool()
        global cost_model, evaluated_points
        cost_model = self.regressor
        evaluated_points = visited
        if self.num_threads > 1:
            self.pool = multiprocessing.Pool(self.num_threads)

    def fit(self, xs, ys):
        ys = np.reshape(ys, (len(ys), 1))
        self.regressor.fit(xs, ys)


    def predict(self, xs,  return_std=False):
        predicates = self.regressor.predict(xs, return_std=return_std)
        return predicates

    def next_batch(self, visited, visited_indexes, batch_size, test_points):
        self._reset_pool(visited)

        if __measure_time__:
            start = timeit.default_timer()

        local_maximums = []
        for i in range(len(test_points) // (self.max_test_points) + 1):

            if self.num_threads > 1:
                step = min(self.max_test_points, len(test_points[i*self.max_test_points:]))//self.num_threads
                sliced_test_points = [test_points[i*self.max_test_points+step*j:i*self.max_test_points+step*(j+1)] for j in range(self.num_threads)]
                scores = self.pool.map(acquisition, sliced_test_points)
                scores = np.reshape(scores, [-1,])

            else:
                end_point = min(len(test_points), (i+1)*self.max_test_points)
                sliced_test_points = test_points[i * (self.max_test_points):end_point]
                scores = acquisition(sliced_test_points)

            accepted = 0

            while accepted < batch_size:
                arg_max = np.argmax(scores)
                if arg_max + (i * (self.max_test_points)) not in visited_indexes:
                    local_maximums.append([scores[arg_max], arg_max + (i * (self.max_test_points))])
                    accepted += 1
                scores[arg_max] = 0
            # print(i)
            del scores

        maximums = []
        for _ in range(batch_size):
            arg_max = np.argmax(local_maximums,axis=0)
            maximums.append(local_maximums[arg_max[0]][1])
            local_maximums[arg_max[0]][0] = 0

        self._close_pool()

        if __measure_time__:
            stop = timeit.default_timer()
            print('Time-find maximas: ', stop - start)
            print("Visited points so far {}, points to be tested {}".format(len(visited_indexes), len(test_points)))
        for i in range(int(self.random_points*len(maximums))):
            x = np.random.randint(0,len(test_points))
            while x in visited_indexes or x in maximums:
                x = np.random.randint(0, len(test_points))
            maximums[-i] = x
        self.random_points *= self.decay

        return maximums




cost_model = None
evaluated_points = None



def acquisition(test_points):
    best = 0
    if evaluated_points is not None:
        yhat, _ = surrogate(evaluated_points)
        best = max(yhat)
    mu, std = surrogate(test_points)
    mu = mu[:, 0]
    probs = norm.cdf((mu - best) / (std + 1E-9))
    return probs

def surrogate(x):
    with catch_warnings():
        # ignore generated warnings
        simplefilter("ignore")
        return cost_model.predict(x, return_std=True)


_config_space = None

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
    return np.asarray(config, dtype=dtype(config))

### Input : tasks.config_space.space_map
### output: a sorted dict from axises to possible values for each axis - purpose is passing to cython code

import collections
def extract_map_from_config(space_map):
    ret = {}
    i = 100
    for key, value in space_map.items():
        if hasattr(value.entities[-1],'size'):
            ret[bytes("a"+str(i), encoding='utf8')] = [x.size for x in value.entities]
        else:
            ret[bytes("a"+str(i), encoding='utf8')] = [[x.val] for x in value.entities]
        i += 1
    return ret