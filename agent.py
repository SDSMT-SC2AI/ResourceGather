from base_agent import BaseAgent
# import actions
import observer
import network
hidden_layer_size = (3*observer.observation_size)//2

policy_spec = network.Policy.policy_spec(input_size=observer.observation_size,
                                         hidden_layer_size=hidden_layer_size,
                                         num_actions=3)  # actions.action_space_size)
trainer_spec = network.Trainer.trainer_spec()


class Smart(BaseAgent):
    def __init__(self, name, parent, optimizer, episode):
        super().__init__(name, parent, optimizer, network, episode, policy_spec, trainer_spec)

    @staticmethod
    def process_observation(obs, flags=None):
        return observer.process_observation(obs, flags)
