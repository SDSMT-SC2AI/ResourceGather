from .objects import Base
from .actions import ActionError
from . import actions as env_actions


summary_fmt = \
    """Idealized StarCraft II Environment Game State:
    time elapsed: {env.time_elapsed:.1f}
    minerals:     {env.minerals:8.1f}
    gas:          {env.gas:8.1f}
    
    resources collected: {env.resources_collected:.1f}
    resource collection rate: {env.resource_collection_rate:.2f}
    
    supply: {env.supply[0]:3d} / {env.supply[1]:3d}
    bases:     {env.number_bases}
    larva:     {env.larva:d}
    drones:    {env.drones}
    queens:    {env.queens}
    overlords: {env.overlords}
    spwnpool:  {env.spawning_pool}

"""

base_info_fmt = \
    """Base {base_id}:
    minerals: {base.minerals.remaining:8.1f} / {base.minerals.max_capacity:8.1f}
    geyserA: {base.geyserA.remaining:7.1f} / {base.geyserA.max_capacity:7.1f}  (Extractor: {base.geyserA.has_extractor})
    geyserB: {base.geyserB.remaining:7.1f} / {base.geyserB.max_capacity:7.1f}  (Extractor: {base.geyserB.has_extractor})
    
    larva: {base.larva:.1f}
    queens: {base.queens} (queued: {base.queens_queued})
    unassigned drones:  {base.unassigned_drones:2d} (queued: {base.drones_queued})
    
    drones at minerals: {base.minerals.drones:2d} / {base.minerals.equiv_max:4.1f}
    drones at GeyserA:  {base.geyserA.drones:2d} / {base.geyserA.equiv_max:4.1f}
    drones at GeyserB:  {base.geyserB.drones:2d} / {base.geyserB.equiv_max:4.1f}
        
    resource collection rate: {base.resource_collection_rate:.2f}

"""


class IdealizedSC2Env:
    actions = env_actions.actions
    def __init__(self, game_loops_per_agent_step, time_limit=720, silent_errors=False, verbose=False):
        self.time_limit = time_limit
        self.silent_errors = silent_errors
        self.verbose = verbose
        self.game_loops_per_agent_step = game_loops_per_agent_step
        self.reset()

    def reset(self):
        self.time_elapsed = 0

        # State information
        self.bases = [Base(self)]
        self.bases[0].minerals.drones = 12
        self.focus = self.bases[0]
        self.target = None
        self.clock_rate = 0.1
        self.overlords = 1
        self.spawning_pool = False
        self.actions_in_progress = set()
        self.minerals = 50
        self.gas = 0
        self.resources_collected = 0
        self.resource_collection_rate = 0
        self.log = []
        self.reward = 0
        self.base_index = 0

        # returns reward, observation, and game_end flag
        return [[self.reward, self, False], ]

    def step(self, actions):
        valid_action = True
        for a in actions:
            try:
                self.actions_in_progress.add(a(self))
            except ActionError as e:
                valid_action = False
                if self.silent_errors:
                    self.log.append(e)
                else:
                    raise e

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
            self.resource_collection_rate = (minerals_gathered + gas_harvested)/self.clock_rate

        completed_actions = []
        for action in self.actions_in_progress:
            if action.tick():
                completed_actions.append(action)

        for action in completed_actions:
            self.actions_in_progress.remove(action)

        self.time_elapsed += self.clock_rate

    @property
    def drones(self):
        drones = 0
        for base in self.bases:
            drones += base.unassigned_drones
            drones += base.minerals.drones
            drones += base.geyserA.drones
            drones += base.geyserB.drones
        return drones

    @property
    def queens(self):
        queens = 0
        for base in self.bases:
            queens += base.queens
        return queens

    @property
    def number_bases(self):
        return len(self.bases)

    @property
    def larva(self):
        larva = 0
        for base in self.bases:
            larva += int(base.larva)
        return larva

    @property
    def supply(self):
        supply = 8 * self.overlords
        used = 0
        for base in self.bases:
            supply += 6
            used += base.unassigned_drones + base.drones_queued
            used += base.minerals.drones + base.geyserA.drones + base.geyserB.drones
            used += 2*base.queens + 2*base.queens_queued

        supply = min(200, supply)
        return used, supply

    @property
    def available_actions(self):
        available_actions = set()
        for act in self.actions:
            try:
                act.verify(self)
                available_actions.add(act)
            except ActionError:
                pass
        return available_actions

    def __str__(self):
        s = summary_fmt.format(env=self)
        if self.verbose:
            for number, base in enumerate(self.bases):
                s += base_info_fmt.format(base_id=number, base=base)
        return s
