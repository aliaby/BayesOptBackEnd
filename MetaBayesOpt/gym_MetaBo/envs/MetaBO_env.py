import gym
from multiprocessing import Process, Queue
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

import timeit

import torch as th

class MetaBoEnv(gym.Env, Process):

    metadata = {'render.modes': ['human']}

    def __init__(self, name=None, sender=None, reciver=None, config_space_map=None, features = None):

        super(MetaBoEnv, self).__init__()

        self.sender = sender
        self.reciver = reciver
        self.n_features = 0
        if config_space_map != None:
            self.space_size = [len(x.entities) for x in config_space_map.values()]
            self.config_space_map = config_space_map
            self.action_space, self.observation_space = self.create_spaces(config_space_map)
        else:
            self.space_size = None
            self.config_space_map = None
            self.action_space, self.observation_space = None, None

        self.state = 0
        self.regressor = GaussianProcessRegressor(copy_X_train=True, normalize_y=True)
        self.visited_xs = []
        self.visited_yx = []
        self.best_ys = 0
        self.best_ys_gp = 0
        self.best_xs_gp = None

        self.policy = None

        self.rng = None
        self.seeded_with = None
        self.seed(42)
        self.n_features = 22
        self.N = 10000
        self.obs = None
        self.state_space = None
        self.state_space_len = 10
        self.feature_space = None
        self.search_space = None

    def set_config_space(self, config_space_map):
        self.space_size = [len(x.entities) for x in config_space_map.values()]
        self.config_space_map = config_space_map
        self.action_space, self.observation_space = self.create_spaces(config_space_map)

    def step(self, action):
        xs = self.get_features(action)
        reward = self._rewrad(xs)
        info = {}
        self.update_maximum(xs)
        self.obs = obs = self.feature_space[0]#self.get_state(self.state_space[self.rng.randint(0, self.state_space_len)])
        done = False
        return obs, reward, done, info

    def reset(self):
        self.state_space = [[[self.rng.randint(0, len) for len in self.space_size] for i in range(self.N - len(self.visited_xs))] for i in range(self.state_space_len)]
        self.feature_space = [self.get_state(self.state_space[i]) for i in range(self.state_space_len)]

        return self.feature_space[0]
        # return self.get_state(self.get_features([self.rng.randint(0, len) for len in self.space_size]))

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        self.rng = np.random.RandomState()
        self.seeded_with = seed
        self.rng.seed(self.seeded_with)

    def set_messanger(self, sender, reciver):
        self.sender = sender
        self.reciver = reciver

    def run(self):
        while True:
            xs, ys = self.reciver.get()
            self.fit_regressor(xs, ys)
            self.sender.put("Done")

    def _rewrad(self, xs_features):
        ys, _ = self.regressor.predict(xs_features, return_std=True)

        return -(self.best_ys - ys)

    def create_spaces(self, config_space_map):
        return self.create_action_space(config_space_map), self.create_observation_space(config_space_map)

    def create_action_space(self, config_space_map):
        dims = []
        for _, tile in config_space_map.items():
            dims.append(len(tile.entities))
            if dims[-1] <= 0:
                dims[-1] = 1
        print(dims)
        return gym.spaces.MultiDiscrete(dims)

    def create_observation_space(self, config_space_map):
        return gym.spaces.Box(low=-1000.0, high=1000.0,
                                                shape=(self.N, self.n_features),
                                                dtype=np.float32)

    def get_state(self, X):
        states = []
        Ys, stds = self.regressor.predict(X, return_std=True)
        for xs, ys, std in zip(X, Ys, stds):
            state = []
            var = std ** 2
            state.append(ys)
            state.append(var)
            state = np.concatenate((state, self.get_features(xs)[0]), axis=0)
            states.append(state)
        for xs, ys in zip(self.visited_xs, self.visited_yx):
            state = []
            state.append(ys)
            state.append(0)
            state = np.concatenate((state, self.get_features(xs)[0]), axis=0)
            states.append(state)
        return states

    def fit_regressor(self, xs, ys):
        ys = np.reshape(ys, (len(ys), 1))
        self.regressor.fit(xs, ys)
        self.visited_xs = xs
        self.visited_yx = ys
        self.best_ys = np.max(self.visited_yx / np.linalg.norm(self.visited_yx))

    def get_features(self, action):
        feature = np.asarray([])
        for index, tile in zip(action, self.config_space_map.values()):
            if index > len(tile.entities):
                index = index % len(tile.entities)
            if hasattr(tile.entities[index - 1], 'size'):
                feature = np.concatenate((feature, tile.entities[index - 1].size), axis=0)
            else:
                feature = np.concatenate((feature, [tile.entities[index - 1].val]), axis=0)
        return [feature]

    def next_batch(self, batch_size):
        actions, values = self.find_maximums()
        maximums = []
        for _ in range(batch_size):
            index = th.argmax(values)
            maximums.append(np.asarray(th.Tensor.cpu(actions[index])))
            values[index] = -1000000

        return maximums

    def set_policy(self, policy):
        self.policy = policy

    def find_maximums(self):
        if self.search_space is None:
            self.search_space = [self.get_state([[self.rng.randint(0, len) for len in self.space_size] for i in range(self.N - len(self.visited_xs))]) for _ in range(256)]
        return self.policy.get_value(self.search_space)

    def update_maximum(self, xs):
        mean = self.regressor.predict(xs)
        if mean > self.best_ys_gp or self.best_xs_gp is None:
            self.best_xs_gp = xs
            self.best_ys = mean

