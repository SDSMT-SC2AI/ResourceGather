import numpy as np
import network
from base_agent import BaseAgent
hidden_layer_size = 3
network_spec = {
    "input size": 2,
    "hidden layer size": hidden_layer_size,
    "number of actions": 3
}


class Actions:
    def __init__(self):
        self.choices = [[0], [1], [2]]

    def act(self, choice):
        if choice in range(len(self.choices)):
            return 1, self.choices[choice]
        else:
            return -1, self.choices[0]


class Simple(BaseAgent):
    def __init__(self, name, parent, optimizer):
        super().__init__(name, parent, optimizer, network, Actions(), network_spec)

    def step(self, sess, observation):
        choice, action_dist, value = self.policy.step(sess, observation, 1)
        return self.action_space.act(choice), choice, action_dist, value

    @staticmethod
    def process_observation(obs, flags=None):
        reward = obs[0][0]
        n_steps = obs[0][1]
        ends = obs[0][2]
        net_in = np.array([[(n_steps // 3) % 3, n_steps % 3]])
        return reward, net_in, ends


