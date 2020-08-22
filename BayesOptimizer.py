import numpy as np

import multiprocessing
from scipy.stats import norm
from warnings import catch_warnings
from warnings import simplefilter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, _check_length_scale
import timeit
import copy
from multiprocessing import Queue, Process, Manager

import os


from MetaBayesOpt.AquisitionFunctions import MLPAF
from stable_baselines3.common.policies import register_policy


MMetric = None

__ACQUISITION__ = 'PI'

register_policy('MLPAF', MLPAF)

mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
maximum_search_points = 2**20#2**int(log2(mem_bytes/512 - 1))

__measure_time__ = False
__USE_CPP_BACKEND__ = None
__REGRESSOR_LIB__ = "SKLearn"

if __REGRESSOR_LIB__ == "GPY":
    try:
        import GPy
    except:
        print("importing GPY failed...")
        __REGRESSOR_LIB__= "SKLEARN"

if __USE_CPP_BACKEND__:
    try:
        import BayesOptimizer_wrapper

        print("Using C++ backend...")
    except:
        print("Loading C++ backend failed")
        __USE_CPP_BACKEND__ = False


class BayesianOptimizer(object):

    def __init__(self, run_multi_threaded=False, metric=None):

        if metric is not None:
            global MMetric
            MMetric = metric

        if run_multi_threaded:
            self.num_threads = multiprocessing.cpu_count()
        else:
            self.num_threads = 1
            self.pool = None

        self.evaluation_depth = maximum_search_points * self.num_threads
        self.random_points = 0.0
        self.decay = 0.95

        if __USE_CPP_BACKEND__:
            self.regressor = None
        else:
            if __REGRESSOR_LIB__=="GPy":
                self.regressor = None
            else:
                kernel = (ConstantKernel(1.0, constant_value_bounds="fixed") *
                          sRBF(1.0, length_scale_bounds="fixed"))
                self.regressor = GaussianProcessRegressor(copy_X_train=False, normalize_y=True, kernel=kernel)

    def set_map(self, space_map):
        self.regressor = BayesOptimizer_wrapper.PyBayesOptimizer(extract_map_from_config(space_map))

    def reset(self):
        del self.regressor
        self.regressor = None
        self.regressor = GaussianProcessRegressor(copy_X_train=False, normalize_y=True)

    def fit(self, xs, ys):
        norm = np.linalg.norm(ys)
        ys = ys / norm
        if __USE_CPP_BACKEND__:
            self.regressor.fit(xs, ys)
        else:
            ys = np.reshape(ys, (len(ys), 1))
            if __REGRESSOR_LIB__ == "GPy":
                kern = GPy.kern.RBF(input_dim=1, ARD=True)
                M = 15
                RR = np.linspace(-1, 1, M)[:, None]
                α = 0.0001
                self.regressor = GPy.models.SparseGPRegression(xs, ys, kern.copy(), Z=RR.copy())
                self.regressor.inference_method = GPy.inference.latent_function_inference.PEP(α)
                self.regressor.optimize(messages=False)
            else:
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
        if isinstance(self.regressor, "GPy.models.SparseGPRegression"):
            mean, var = self.regressor.predict(xs)
            if not return_std:
                return mean
            return mean, var
        predicates = self.regressor.predict(xs, return_std=return_std)
        return predicates

    def next_batch(self, visited, visited_indexes, batch_size, test_points, visited_results):
        if __measure_time__:
            start = timeit.default_timer()

        if __USE_CPP_BACKEND__:
            maximums = self.regressor.next_batch(batch_size, visited_indexes)
            if __measure_time__:
                stop = timeit.default_timer()
                print('Time-find maximums: ', stop - start)
                print("Visited points so far {}, points to be tested {}".format(len(visited_indexes), len(test_points)))
            return maximums



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
                scores = self.acquisition(sliced_test_points,
                                          visited_results / np.linalg.norm(visited_results))
                accepted = 0
                while accepted < batch_size:
                    arg_max = np.argmax(scores)
                    if arg_max + (i * (self.evaluation_depth)) not in visited_indexes:
                        local_maximums.append([scores[arg_max], arg_max + (i * (self.evaluation_depth))])
                        accepted += 1
                    scores[arg_max] = -1

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

    def acquisition(self, test_points, evaluation_results):
        if __ACQUISITION__ == 'PI':
            return self.expected_improvement(test_points, evaluation_results)

    def PI(self, test_points, evaluation_results):
        minimum = min(evaluation_results)
        mu, std = self.surrogate(test_points)
        mu = mu[:, 0]
        probs = norm.cdf(minimum, loc=mu, scale=std)
        return probs

    def expected_improvement(self, test_points, evaluation_results, xi=0.02):
        best = min(evaluation_results)
        mu, std = self.surrogate(test_points)
        mu = mu[:, 0]
        # std = std
        with catch_warnings():
            # ignore generated warnings
            simplefilter("ignore")
            imp = mu - best - xi
            Z = imp / std
            ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
            ei[std == 0.0] = 0.0

        return ei

    def surrogate(self, x):
        with catch_warnings():
            # ignore generated warnings
            simplefilter("ignore")
            return self.regressor.predict(x, return_std=True)

cost_model = None
evaluated_points = None
evaluation_results = None

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
            if isinstance(self.regressor,"GaussianProcessRegressor"):
                return self.regressor.predict(points, return_std=True)
            return self.regressor.predict(points)

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

from scipy.spatial.distance import pdist, cdist, squareform

class sRBF(RBF):
    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)
        if Y is None:
            dists = pdist(X / length_scale, metric=MMetric)
            K = np.exp(-.5 * dists)
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            dists = cdist(X / length_scale, Y / length_scale,
                          metric=MMetric)
            K = np.exp(-.5 * dists)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                K_gradient = \
                    (K * squareform(dists))[:, :, np.newaxis]
                return K, K_gradient
            elif self.anisotropic:
                # We need to recompute the pairwise dimension-wise distances
                K_gradient = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 \
                    / (length_scale ** 2)
                K_gradient *= K[..., np.newaxis]
                return K, K_gradient
        else:
            return K