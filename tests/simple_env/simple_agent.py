import numpy as np
import network
from base_agent import BaseAgent
hidden_layer_size = 3

policy_spec = network.Policy.policy_spec(input_size=2,
                                         hidden_layer_size=3,
                                         q_range=(30, 31),
                                         max_episodes=1000)
trainer_spec = network.Trainer.trainer_spec(consistency_coefficient=0.3,
                                            advantage_coefficient=2.0,
                                            discount_factor=0.9,
                                            max_grad_norm=5.0)


class Simple(BaseAgent):
    def __init__(self, name, parent, optimizer, episode, action_spec):
        policy_spec.update(action_spec)
        super().__init__(name, parent, optimizer, network, episode, policy_spec, trainer_spec)

    @staticmethod
    def process_observation(obs, flags=None):
        reward = obs[0][0]
        n_steps = obs[0][1]
        ends = obs[0][2]
        net_in = np.array([[(n_steps // 3) % 3, n_steps % 3]])
        return reward, net_in, ends


