from gym.envs.registration import register

register(
    id='MetaBo-v0',
    entry_point='gym_MetaBo.envs:MetaBoEnv',
)