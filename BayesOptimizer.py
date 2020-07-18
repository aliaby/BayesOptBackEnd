import numpy as np

import multiprocessing
from scipy.stats import norm
from warnings import catch_warnings
from warnings import simplefilter
from sklearn.gaussian_process import GaussianProcessRegressor
import timeit
from math import log2
import threading
import copy
from multiprocessing import Queue, Process, Pipe, Manager

import os
mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
maximum_search_points = 2**int(log2(mem_bytes/512 - 1))

__measure_time__ = True
__USE_CPP_BACKEND__ = None
__REGRESSOR_LIB__ = "SKLearn"

if __REGRESSOR_LIB__ == "GPY":
    try:
        import GPy
    except:
        print("importing GPY failed...")

if __USE_CPP_BACKEND__:
    try:
        import BayesOptimizer_wrapper

        print("Using C++ backend...")
    except:
        print("Loading C++ backend failed")
        __USE_CPP_BACKEND__ = False


class BayesianOptimizer(object):

    def __init__(self, run_multi_threaded=False):

        if run_multi_threaded:
            self.num_threads = multiprocessing.cpu_count()
        else:
            self.num_threads = 1
            self.pool = None

        self.evaluation_depth = maximum_search_points * self.num_threads
        self.random_points = 0.3
        self.decay = 0.95

        if __USE_CPP_BACKEND__:
            self.regressor = None
        else:
            self.regressor = GaussianProcessRegressor(copy_X_train=False, normalize_y=True)

    def set_map(self, space_map):
        self.regressor = BayesOptimizer_wrapper.PyBayesOptimizer(extract_map_from_config(space_map))

    def reset(self):
        del self.regressor
        self.regressor = None
        self.regressor = GaussianProcessRegressor(copy_X_train=False, normalize_y=True)

    def fit(self, xs, ys):
        if __USE_CPP_BACKEND__:
            norm = np.linalg.norm(ys)
            ys = ys / norm
        else:
            ys = np.reshape(ys, (len(ys), 1))
        self.regressor.fit(xs, ys)

    def create_calc_score_threads(self, sliced_test_points, batch_size):
        start = timeit.default_timer()
        with Manager() as manager:
            threads = []
            maxes = []
            for i in range(self.num_threads):
                maximums = manager.dict()
                thread = RegressorThread(regressor=copy.copy(self.regressor), search_space=sliced_test_points[i], id=i, maximums=maximums, batch_size=batch_size)
                thread.start()
                threads.append(thread)
                maxes.append(maximums)

            for thread in threads:
                thread.join()
                thread.close()
                del thread
            local_maximums = {}

            minimum= 1e9
            for i in range(len(maxes)):
                for key, value in maxes[i].items():
                    if len(local_maximums) < batch_size:
                        local_maximums[key+i*len(sliced_test_points[-1])] = value
                        minimum= min(minimum, value)
                    elif value > minimum:
                        local_maximums[key+i*len(sliced_test_points[-1])] = value
                        for key_2, value_2 in local_maximums.items():
                            if value_2 == minimum:
                                local_maximums[key_2]
                        minimum= min(local_maximums.keys(), key=(lambda k: local_maximums[k]))
            print(timeit.default_timer()-start)
            return local_maximums

    def predict(self, xs, return_std=False):
        predicates = self.regressor.predict(xs, return_std=return_std)
        return predicates

    def next_batch(self, visited, visited_indexes, batch_size, test_points):
        if __measure_time__:
            start = timeit.default_timer()

        if __USE_CPP_BACKEND__:
            maximums = self.regressor.next_batch(batch_size, visited_indexes)
            if __measure_time__:
                stop = timeit.default_timer()
                print('Time-find maximas: ', stop - start)
                print("Visited points so far {}, points to be tested {}".format(len(visited_indexes), len(test_points)))
            return maximums

        global cost_model, evaluated_points
        cost_model = self.regressor
        evaluated_points = visited

        local_maximums = []
        for i in range(len(test_points) // (self.evaluation_depth) + 1):
            multi_threaded = False
            if self.num_threads > 1:
                if min(self.num_threads, 2**21/len(test_points)) > 1:
                    multi_threaded = True
                    self.num_threads = int(min(self.num_threads, maximum_search_points/len(test_points)))
            if multi_threaded:
                step = min(self.evaluation_depth, len(test_points[i * self.evaluation_depth:])) // self.num_threads
                sliced_test_points = [
                    test_points[i * self.evaluation_depth + step * j:i * self.evaluation_depth + step * (j + 1)] for j
                    in range(self.num_threads)]
                scores = self.create_calc_score_threads(sliced_test_points, batch_size)

                for key, value in scores.items():
                        local_maximums.append([value, key + i*self.evaluation_depth])

            else:
                end_point = min(len(test_points), (i + 1) * self.evaluation_depth)
                sliced_test_points = test_points[i * (self.evaluation_depth):end_point]
                scores = acquisition(sliced_test_points)
                accepted = 0
                while accepted < batch_size:
                    arg_max = np.argmax(scores)
                    if arg_max + (i * (self.evaluation_depth)) not in visited_indexes:
                        local_maximums.append([scores[arg_max], arg_max + (i * (self.evaluation_depth))])
                        accepted += 1
                    scores[arg_max] = 0
            del scores

        maximums = []
        for _ in range(batch_size):
            arg_max = np.argmax(local_maximums, axis=0)
            maximums.append(local_maximums[arg_max[0]][1])
            local_maximums[arg_max[0]][0] = 0

        if __measure_time__:
            stop = timeit.default_timer()
            print('Time-find maximas: ', stop - start)
            print("Visited points so far {}, points to be tested {}".format(len(visited_indexes), len(test_points)))
        for i in range(int(self.random_points * len(maximums))):
            x = np.random.randint(0, len(test_points))
            while x in visited_indexes or x in maximums:
                x = np.random.randint(0, len(test_points))
            maximums[-i] = x
        self.random_points *= self.decay

        return maximums


cost_model = None
evaluated_points = None


class RegressorThread(Process):
    def __init__(self, group=None, target=None, name=None, regressor=None, search_space=None, maximums=None, id=0, batch_size = 24):
        super(RegressorThread, self).__init__(group=group, target=target, name=name)
        self.regressor = regressor
        self.search_space = search_space
        self.id = id
        self.scores = []
        self.maximums = maximums
        self.batch_size = batch_size

    def _surrogate(self, points):
        with catch_warnings():
            simplefilter("ignore")
            return self.regressor.predict(points, return_std=True)

    def _acquisition(self):
        best = 0
        if evaluated_points is not None:
            yhat, _ = self._surrogate(evaluated_points)
            best = max(yhat)
        mu, std = self._surrogate(self.search_space)
        mu = mu[:, 0]
        return norm.cdf((mu - best) / (std + 1E-9))

    def run(self):
        # print(self.scores)
        scores = self._acquisition()
        minimum= 1e9
        for i in range(len(scores)):
            if len(self.maximums) < self.batch_size:
                self.maximums[i] = scores[i]
                minimum= min(minimum, scores[i])
            elif scores[i] > minimum:
                self.maximums[i] = scores[i]
                for key, value in self.maximums.items():
                    if value == minimum:
                        del self.maximums[key]
                minimum= min(self.maximums.values())

        return


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


def extract_map_from_config(space_map):
    ret = {}
    i = 100
    for key, value in space_map.items():
        if hasattr(value.entities[-1], 'size'):
            ret[bytes("a" + str(i), encoding='utf8')] = [x.size for x in value.entities]
        else:
            ret[bytes("a" + str(i), encoding='utf8')] = [[x.val] for x in value.entities]
        i += 1
    return ret
