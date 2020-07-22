# import gym
# from gym_MetaBo.envs.MetaBO_env import MetaBoEnv
#
# from stable_baselines3 import PPO
# from threading import Thread
# from multiprocessing import Process, Queue
#
# # gym.register(
# #     id='MetaBo-v0',
# #     entry_point='gym_MetaBo.envs:MetaBoEnv',
# # )
#
#
#
#
# class PPOProcess(Thread):
#     def __init__(self, model, total_timesteps=64):
#
#         super(PPOProcess,self).__init__()
#         self.model = model
#         self.total_timesteps = total_timesteps
#
#     def run(self):
#         self.model.learn(total_timesteps=self.total_timesteps)
#
# class reward_function(Process):
#
#     def __init__(self, sender, reciver):
#         super(reward_function,self).__init__()
#         self.sender = sender
#         self.receiver = reciver
#
#     def run(self):
#         while True:
#             cmd = self.receiver.get()
#             # print(cmd)
#             self.sender.put({"reward":cmd["action"]^cmd["state"]})
#
#
# env = gym.make('MetaBo-v0')
# s = Queue()
# r = Queue()
# env.set_messanger(s,r)
# env.start()
#
# rf = reward_function(r, s)
# rf.start()
#
# model = PPO('MlpPolicy', env, verbose=1)
#
# ppoproc = PPOProcess(model)
#
# ppoproc.run()
#
#
# obs = env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     print("acion {}, state {}, obs {}, reward {}".format(action, _states, obs, reward))


import gym

from stable_baselines3 import PPO

env = gym.make('CartPole-v1')

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10)


model.learn(total_timesteps=10000)


obs = env.reset()

for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()