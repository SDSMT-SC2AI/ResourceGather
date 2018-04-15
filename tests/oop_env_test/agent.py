import numpy as np
from context import set_path
set_path()
from a3c.base_agent import BaseAgent
from tests.oop_env_test.network import Policy, Trainer


class Simple(BaseAgent):
    policy_cls = Policy
    trainer_cls = Trainer

    def __init__(self, name, parent, optimizer, policy_kwargs, trainer_kwargs):
        super().__init__(name, parent, optimizer, policy_kwargs, trainer_kwargs)

    def process_observation(self, obs):
        env_obs, feed_back = obs
        reward, n_steps, ends = env_obs[0]
        net_in = np.array([[(n_steps // 3) % 3, n_steps % 3]])
        return reward + feed_back, net_in, ends

    @classmethod
    def policy_kwargs(cls, num_actions, episode, **kwargs):
        return cls.policy_cls.policy_kwargs(
            input_size=2,
            num_actions=num_actions,
            episode=episode, **kwargs)

    @classmethod
    def trainer_kwargs(cls, **kwargs):
        return cls.trainer_cls.trainer_kwargs(**kwargs)
