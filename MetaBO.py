import numpy as np
import timeit
from multiprocessing import Queue
import gym
from MetaBayesOpt.PPOTL import PPOTL
from gym_MetaBo.envs.MetaBO_env import MetaBoEnv

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