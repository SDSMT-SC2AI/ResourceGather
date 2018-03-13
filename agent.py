from base_agent import BaseAgent
import actions
import observer
import network
hidden_layer_size = (3*observer.observation_size)//2
network_spec = {
            "input size": observer.observation_size,
            "hidden layer size": hidden_layer_size,
            "number of actions": actions.action_space_size
        }


class Smart(BaseAgent):
    def __init__(self, name, parent, optimizer):
        super().__init__(name, parent, optimizer, network, actions, network_spec)

    @staticmethod
    def process_observation(obs, flags=None):
        return observer.process_observation(obs, flags)
