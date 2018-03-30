from base_agent import BaseAgent
import observer
import network
hidden_layer_size = (3*observer.observation_size)//2

policy_spec = network.Policy.policy_spec(
            input_size=20,
            num_actions=12,
            max_episodes=2500,
            q_range=(30, 31),
            hidden_layer_size=30,
            base_explore_rate=0.1,                 
            min_explore_rate=0.01)
trainer_spec = network.Trainer.trainer_spec()


class Smart(BaseAgent):
    def __init__(self, name, parent, optimizer, episode, action_spec):
        policy_spec.update(action_spec)
        super().__init__(name, parent, optimizer, network, episode, policy_spec, trainer_spec)

    @staticmethod
    def process_observation(obs, flags=None):
        return observer.process_observation(obs, flags)
