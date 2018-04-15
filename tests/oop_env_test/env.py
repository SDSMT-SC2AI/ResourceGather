class SimpleEnv:
    def __init__(self, mode="Basic"):
        self.episode_step = 0
        self.reward = 0
        self.r_func = {
            "Basic": lambda v: self.basic(v),
            "Accumulating": lambda v: self.accumulate(v)
        }[mode]

    def reset(self):
        self.episode_step = 0
        self.reward = 0
        return [[0, self.episode_step, False], ]

    def step(self, actions):
        self.episode_step += 1
        reward = self.get_reward(actions[0]**2)
        return [[reward, self.episode_step, False], ]

    def get_reward(self, value):
        return self.r_func(value)

    def accumulate(self, value):
        self.reward += value
        return self.reward

    def basic(self, value):
        self.reward = value
        return self.reward
