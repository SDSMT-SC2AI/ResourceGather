import numpy as np
from .network import Policy, Trainer
from .context import set_path
set_path()
from a3c.base_agent import BaseAgent


class Simple(BaseAgent):
    def __init__(self, name, parent, optimizer, policy_kwargs, trainer_kwargs):
        super().__init__(name, parent, optimizer, policy_kwargs, trainer_kwargs)

    def setup_policy(self, name, input_size, num_actions, **policy_kwargs):
        return Policy(name, input_size, num_actions, **policy_kwargs)

    def setup_trainer(self, name, parent, optimizer, policy, **trainer_kwargs):
        return Trainer(name, parent, optimizer, policy, **trainer_kwargs)

    def process_observation(self, obs):
        reward = obs[0][0]
        n_steps = obs[0][1]
        ends = obs[0][2]
        net_in = np.array([[(n_steps // 3) % 3, n_steps % 3]])
        return reward, net_in, ends
