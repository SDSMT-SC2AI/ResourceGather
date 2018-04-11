from objects import Base


class IdealizedSC2Env:
    def __init__(self):
        self.reset()

    def reset(self):
        self.game_step = 0

        # State information
        self.bases = [Base()]
        self.bases[0].minerals.drones = 12
        self.focus = self.bases[0]
        self.clock_rate = 0.1
        self.overlords = 1
        self.spawning_pool = False
        self.actions_in_progress = []
        self.minerals = 0
        self.gas = 0
        self.resources_collected = 0
        self.resource_collection_rate = 0
        self.feedback = {}

        # returns reward, observation, and game_end flag
        return [[None, None, None], ]

    def step(self, actions):
        return [[None, None, None], ]



