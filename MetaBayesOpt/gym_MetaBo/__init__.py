from gym.envs.registration import register


register(
    id='MetaBo-v3',
    entry_point='gym_MetaBo.envs:MetaBoEnv',
)