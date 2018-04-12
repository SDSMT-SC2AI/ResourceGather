from objects import Base
from actions import ActionError
import actions as env_actions


class IdealizedSC2Env:
    def __init__(self, game_loops_per_agent_step, time_limit):
        self.time_limit = time_limit
        self.game_loops_per_agent_step = game_loops_per_agent_step
        self.reset()

    def reset(self):
        self.time_elapsed = 0
        Base.reset()

        # State information
        self.bases = [Base(self)]
        self.bases[0].minerals.drones = 12
        self.focus = self.bases[0]
        self.target = None
        self.clock_rate = 0.1
        self.overlords = 1
        self.spawning_pool = False
        self.actions_in_progress = []
        self.minerals = 50
        self.gas = 0
        self.resources_collected = 0
        self.resource_collection_rate = 0
        self.log = []
        self.reward = 0

        # returns reward, observation, and game_end flag
        return [[self.reward, self, False], ]

    def step(self, actions):
        valid_action = True
        for a in actions:
            try:
                self.actions_in_progress.append(a())
            except ActionError as e:
                valid_action = False
                self.log.append(e)

        for _ in range(self.game_loops_per_agent_step):
            self.tick()

        self.reward = self.resource_collection_rate
        if not valid_action:
            self.reward -= 1000

        return [[self.reward, self, self.time_elapsed > self.time_limit], ]

    def tick(self):
        for base in self.bases:
            minerals_gathered, gas_harvested = base.tick()
            self.minerals += minerals_gathered
            self.gas += gas_harvested
            self.resources_collected += minerals_gathered + gas_harvested
            self.resource_collection_rate = minerals_gathered + gas_harvested

        completed_actions = []
        for index, action in enumerate(self.actions_in_progress):
            if action.tick():
                completed_actions.append(index)

        for index in completed_actions:
            self.actions_in_progress.pop(index)

        self.time_elapsed += self.clock_rate

    def observe(self):
        pass

    # Returns the total supply and the supply used
    def get_supply(self):
        supply = 8 * self.overlords
        used = 0
        for base in self.bases:
            supply += 6
            used += base.unassigned_drones
            used += base.minerals.drones + base.geyserA.drones + base.geyserB.drones
            used += 2*base.queens

        supply = min(200, supply)
        return used, supply

    def get_available_actions(self):
        available_actions = set()
        for act in [env_actions.BuildDrone,
                    env_actions.BuildBase,
                    env_actions.BuildQueen,
                    env_actions.BuildOverlord,
                    env_actions.BuildSpawningPool,
                    env_actions.BuildExtractor,
                    env_actions.Select,
                    env_actions.Target,
                    env_actions.SetRallyMinerals,
                    env_actions.TransferDrone,
                    env_actions.InjectLarva,
                    env_actions.NoOp]:
            try:
                act.verify(self)
                available_actions.add(act)
            except ActionError:
                pass








