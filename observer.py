from Action_Space import Action_Space
import numpy as np
from common.helper_functions import GetUnits, InDistSqRange
from pysc2.env import environment
observation_size = 0


def process_observation(env_obs, flags=None):
    reward = env_obs[0].raw_obs.score.score
    available_actions = np.array(Action_Space.check_available_actions(env_obs[0].observation['available_actions'], env_obs[0]))
    minerals = np.array([min(env_obs[0].raw_obs.player_common.minerals, flags.max_mineral_cost) / flags.max_mineral_cost])
    food_available = np.array([env_obs[0].raw_obs.player_common.food_cap - env_obs[0].raw_obs.player_common.food_used])
    bases = GetUnits(86, env_obs[0].raw_obs.raw_data.units)
    number_of_bases = np.array([len(bases) / flags.max_bases])
    larva_by_base = np.asarray(get_larva_by_base(env_obs, bases, flags))
    obs = np.concatenate([available_actions, minerals, food_available, number_of_bases, larva_by_base])
    episode_end = (env_obs[0].step_type == environment.StepType.LAST)
    return reward, obs, episode_end



def get_larva_by_base(env_obs, bases, flags=None):
    """Returns a list of how many larva are at each base"""
    all_larva = GetUnits(151, env_obs[0].raw_obs.raw_data.units)
    base_larva = []
    for base in bases:
        base_larva_list = 0
        for larva in all_larva:
            if (InDistSqRange(base.pos, larva.pos, 12)): # Larva are < 9 dist away, gave a buffer
                base_larva_list += 1
        if (base_larva_list != 0):
            base_larva.append(base_larva_list)
    for i in range(flags.max_bases - len(base_larva)):
        base_larva.append(0)
    
    return base_larva
