class SimpleEnv:
    def __init__(self):
        self.episode_step = 0

    def reset(self):
        self.episode_step = 0
        return [[0, self.episode_step, False], ]

    def step(self, actions):
        self.episode_step += 1
        reward = actions[0]**2
        return [[reward, self.episode_step, self.episode_step > 100], ]
