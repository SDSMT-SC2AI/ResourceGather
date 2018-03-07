observation_size = 0


def process_observation(env_obs, flags=None):
	reward = None
	obs = None
	episode_end = True
	return reward, obs, episode_end