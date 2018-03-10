import actions
import numpy as np
from pysc2.env import environment
observation_size = 0


def process_observation(env_obs, flags=None):
    reward = env_obs.observation.reward
    available_actions = np.array([actions.check_available_actions(env_obs.observation['available_actions'])])
    minerals = np.array([min(env_obs.raw_obs.player_common.minerals, flags.max_mineral_cost) / flags.max_mineral_cost])
    food_available = np.array([env_obs.raw_obs.player_common.food_cap - env_obs.raw_obs.player_common.food_used])
    number_of_bases = np.array([get_number_of_bases(env_obs, flags) / flags.max_bases])
    larva_by_base = np.array(get_larva_by_base(env_obs, flags))
    obs = np.concatenate([available_actions, minerals, food_available, number_of_bases, larva_by_base])
    episode_end = (env_obs.observation.step_type == environment.StepType.LAST)
    return reward, obs, episode_end


def get_number_of_bases(env_obs, flags=None):
    number_bases = 0
    return number_bases


def get_larva_by_base(env_obs, flags=None):
    larva = [0]*flags.max_bases
    return larva
