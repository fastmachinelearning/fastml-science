from gym.envs.registration import register

register(
    id='ExaBooster-v1',
    entry_point='gym_accelerators.envs:ExaBooster_v1',
)

