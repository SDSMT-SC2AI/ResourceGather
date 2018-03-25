from base_agent import BaseAgent
import observer
import network
hidden_layer_size = (3*observer.observation_size)//2

policy_spec = network.Policy.policy_spec(input_size=observer.observation_size,
                                         hidden_layer_size=hidden_layer_size,
                                         q_range=(7, 10),
                                         max_episodes=100000)  # actions.action_space_size)
trainer_spec = network.Trainer.trainer_spec()


class Smart(BaseAgent):
    def __init__(self, name, parent, optimizer, episode, action_spec):
        policy_spec.update(action_spec)
        super().__init__(name, parent, optimizer, network, episode, policy_spec, trainer_spec)

    @staticmethod
    def process_observation(obs, flags=None):
        return observer.process_observation(obs, flags)
