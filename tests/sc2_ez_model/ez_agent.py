import numpy as np
import network
from base_agent import BaseAgent


hidden_layer_size = 3

policy_spec = network.Policy.policy_spec(input_size=2,
                                         hidden_layer_size=3,
                                         q_range=(30, 31),
                                         max_episodes=2500,
                                         min_explore_rate=0.01)
trainer_spec = network.Trainer.trainer_spec(consistency_coefficient=0.3,
                                            advantage_coefficient=2.0,
                                            discount_factor=0.9,
                                            max_grad_norm=5.0)



class Smart(BaseAgent):
    def __init__(self, name, parent, optimizer, episode, action_space, flags):
        policy_spec.update(action_space.action_spec)
        super().__init__(name, parent, optimizer, network, episode, policy_spec, trainer_spec)
        self.action_space = action_space
        self.flags = flags

    def process_observation(self, obs):
        return observer.process_observation(obs, self.action_space, self.flags)



def process_observation(env, action_space, flags=None):
    reward = env.reward
    available_actions = np.array(action_space.check_available_actions(env))
    minerals = np.array([env.minerals])
    used, supply = env.supply
    food_available = np.array([supply - used])
    number_of_bases = np.array([len(env.bases) / flags.max_bases])
    larva_by_base = np.asarray(get_larva_by_base(env, env.bases, flags))
    obs = np.concatenate([available_actions, minerals, food_available, number_of_bases, larva_by_base])
    episode_end = (env.time_elapsed == env.time_limit)
    return reward, np.reshape(obs, (1, len(obs))), episode_end


def get_larva_by_base(env, bases, flags=None):
    """Returns a list of how many larva are at each base"""
    base_larva = []
    for base in bases:
        base_larva.append([base.larva])
    for i in range(flags.max_bases - len(base_larva)):
        base_larva.append(0)
    
    return base_larva
