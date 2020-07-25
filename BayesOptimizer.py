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
from multiprocessing import Queue, Process, Manager

import os

import gym

from gym_MetaBo.envs.MetaBO_env import MetaBoEnv

from MetaBayesOpt.AquisitionFunctions import MLPAF
from stable_baselines3 import PPO
from stable_baselines3.common.policies import register_policy
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.buffers import RolloutBuffer

from typing import Type, Union, Callable, Optional, Dict, Any

import torch as th


register_policy('MLPAF', MLPAF)

mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
maximum_search_points = 2**20#2**int(log2(mem_bytes/512 - 1))

__measure_time__ = True
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

class PPOTL(PPO):
    def __init__(self, policy: Union[str, Type[ActorCriticPolicy]],
                 env: Union[GymEnv, str],
                 learning_rate: Union[float, Callable] = 3e-4,
                 n_steps: int = 64,
                 batch_size: Optional[int] = 64,
                 n_epochs: int = 10,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_range: float = 0.2,
                 clip_range_vf: Optional[float] = None,
                 ent_coef: float = 0.0,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 use_sde: bool = False,
                 sde_sample_freq: int = -1,
                 target_kl: Optional[float] = None,
                 tensorboard_log: Optional[str] = None,
                 create_eval_env: bool = False,
                 policy_kwargs: Optional[Dict[str, Any]] = None,
                 verbose: int = 0,
                 seed: Optional[int] = None,
                 device: Union[th.device, str] = "auto",
                 _init_setup_model: bool = True):

        super(PPOTL, self).__init__(policy=policy, env=env, learning_rate=learning_rate,
                                    n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs,
                                    gamma=gamma, gae_lambda=gae_lambda, clip_range=clip_range,
                                    clip_range_vf=clip_range_vf, ent_coef=ent_coef, vf_coef=vf_coef,
                                    max_grad_norm=max_grad_norm, use_sde=use_sde, sde_sample_freq=sde_sample_freq,
                                    target_kl=target_kl, tensorboard_log=tensorboard_log, create_eval_env=create_eval_env,
                                    policy_kwargs=policy_kwargs, verbose=verbose, seed=seed, device=device,
                                    _init_setup_model=_init_setup_model)


    def set_policy(self, policy):
        self.policy = policy
        self.setup_PPO_model()

    def setup_PPO_model(self) -> None:

        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.rollout_buffer = RolloutBuffer(self.n_steps, self.observation_space,
                                            self.action_space, self.device,
                                            gamma=self.gamma, gae_lambda=self.gae_lambda,
                                            n_envs=self.n_envs)
        self.policy = self.policy.to(self.device)

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, ("`clip_range_vf` must be positive, "
                                                "pass `None` to deactivate vf clipping")

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)



class MetaBO(object):
    def __init__(self, config, PPOModel=None):
        self.env = gym.make('MetaBo-v3')
        self.receiver = Queue()
        self.sender = Queue()
        self.env.set_messanger(self.receiver, self.sender)
        self.env.set_config_space(config_space_map=config)
        self.env.reset()

        if PPOModel is None:
            self.PPOModel = PPOTL('MLPAF', self.env, verbose=0, _init_setup_model=True)
        else:
            self.PPOModel = PPOTL('MLPAF', self.env, verbose=0, _init_setup_model=False)
            policy = PPOModel.policy
            policy.action_space = self.env.action_space
            policy.observation_space = self.env.observation_space
            self.PPOModel.action_space = self.env.action_space
            self.PPOModel.set_policy(policy)
            self.PPOModel.set_env(self.env)

        self.env.set_policy(self.PPOModel.policy)

    def reset(self):
        pass

    def fit(self, xs, ys):
        if not self.env.is_alive():
            self.env.start()
        self.sender.put((xs, ys))
        res = None
        while res != "Done":
            res = self.receiver.get()
        print("\n")
        start = timeit.default_timer()
        self.PPOModel.learn(total_timesteps=10)
        print("training time: {}".format(timeit.default_timer() - start))

    def next_batch(self,batch_size, config_space, visited):
        start = timeit.default_timer()

        config_space_dims = np.asarray([len(space) for name, space in config_space.space_map.items()])
        inverse_conf_space_dimes = np.asarray([np.prod(config_space_dims[:i]) for i in range(len(config_space_dims))])

        maxes = self.env.next_batch(batch_size)
        indices = []
        for item in maxes:
            indices.append(space_dindex(item, inverse_conf_space_dimes))

        print("inference time: {}".format(timeit.default_timer() - start))

        return indices

    def get_base_model(self):
        return self.PPOModel

def space_dindex(action, inverse_conf_space_dimes):
    index = 0
    i = 0
    for len_i, addr_i in zip(inverse_conf_space_dimes, action):
        index += len_i * addr_i
    return index

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
            if __REGRESSOR_LIB__=="GPy":
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

    def next_batch(self, visited, visited_indexes, batch_size, test_points):
        if __measure_time__:
            start = timeit.default_timer()

        if __USE_CPP_BACKEND__:
            maximums = self.regressor.next_batch(batch_size, visited_indexes)
            if __measure_time__:
                stop = timeit.default_timer()
                print('Time-find maximums: ', stop - start)
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
