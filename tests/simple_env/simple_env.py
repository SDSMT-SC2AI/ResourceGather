class SimpleEnv:
    def __init__(self):
        self.episode_step = 0

    def reset(self):
        return [[0, self.episode_step, False], ]

    def step(self, actions):
        self.episode_step += 1
        return [[len(actions), self.episode_step, self.episode_step > 10], ]
