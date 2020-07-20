import gym
from multiprocessing import Process, Queue
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor


class MetaBoEnv(gym.Env, Process):
    metadata = {'render.modes': ['human']}

    def __init__(self, name=None, sender=None, reciver=None, config_space_map=None):

        super(MetaBoEnv, self).__init__()

        self.sender = sender
        self.reciver = reciver
        self.n_features = 0
        self.space_size = np.prod([len(x) for x in config_space_map])
        self.config_space_map = config_space_map
        self.action_space, self.observation_space = self.create_spaces(config_space_map)

        self.state = 0
        self.regressor = GaussianProcessRegressor(copy_X_train=True, normalize_y=True)
        self.visited_xs = []
        self.visited_yx = []
        self.best_ys = 0

        self.rng = None
        self.seeded_with = None

    def step(self, action):

        # self.sender.put({"action":action, "state":self.state})
        # reward = self.reciver.get()["reward"]
        # print("reward : {}".format(reward))
        xs = self.get_features(action)
        reward = self._rewrad(xs)
        info = {}

        obs = self.get_state(xs)
        done = False
        return obs, reward, done, info

    def reset(self):
        return self.get_state(self.get_features(self.rng.randint(0, self.space_size)))

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
            pass

    def _rewrad(self, xs_features):
        ys, _ = self.regressor.predict(xs_features, return_std=True)

        return -(self.best_ys - ys)

    def create_spaces(self, config_space_map):
        return self.create_action_space(config_space_map), self.create_observation_space(config_space_map)

    def create_action_space(self, config_space_map):
        dims = []
        for tile in config_space_map:
            dims.append((0, len(tile.entities) - 1))

        return gym.spaces.MultiDiscrete(dims)

    def create_observation_space(self, config_space_map):
        dims = []
        for tile in config_space_map:
            dims.append((0, len(tile.entities) - 1))

        return gym.spaces.Tuple(gym.spaces.Box(np.array((0, 1)), np.array((0, 10))),
                                gym.spaces.MultiDiscrete(dims))

    def get_state(self, xs):
        state = []
        ys, std = self.regressor.predict(xs, return_std=True)
        var = std ** 2
        state.append(ys)
        state.append(var)
        state.append(xs)

    def fit_regressor(self, xs, ys):
        ys = np.reshape(ys, (len(ys), 1))
        self.regressor.fit(xs, ys)

        self.visited_xs = np.concatenate(self.visited_xs, xs)
        self.visited_yx = np.concatenate(self.visited_yx, ys)
        self.best_ys = np.max(self.visited_yx / np.liang.norm(self.visited_yx))

    def get_features(self, action):
        feature = []

        for index, tile in zip(action, self.config_space_map):
            feature.append(tile.entities[index])

        return feature.reshape([-1, ])

    def next_batch(self, xs):
        return self.regressor.predict(xs, return_std=True)
