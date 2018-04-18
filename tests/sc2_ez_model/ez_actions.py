from collections import deque
from .environment.actions import \
    BuildQueen, BuildOverlord, BuildDrone, BuildSpawningPool, \
    BuildBase, Select, InjectLarva, NoOp


class ActionEnum:
    no_op = 0
    build_base = 1
    build_drone = 2
    build_queen = 3
    build_overlord = 4
    inject_larva = 5


class Action_Space:
    choices = {
        ActionEnum.build_base: lambda self, obs: self.build_base(obs),
        ActionEnum.build_drone: lambda self, obs: self.build_drone(obs),
        ActionEnum.build_queen: lambda self, obs: self.build_queen(obs),
        ActionEnum.build_overlord: lambda self, obs: self.build_overlord(obs),
        ActionEnum.inject_larva: lambda self, obs: self.inject_larva(obs),
        ActionEnum.no_op: lambda self, obs: self.no_op(),
    }
    action_spec = {
        'number of actions': len(choices)
    }

    def __init__(self):
        self.actionq = deque([])
        self.base_count = 1

    def act(self, choice, obs):
        # if (choice != 0):
            # print(choice)
        if choice < len(self.choices):
            return self.choices[choice](self, obs[1])
        else:
            return self.choices[ActionEnum.no_op](self, obs[1])

    def reset(self):
        self.actionq = deque([])
        self.base_count = 1

    def action_step(self, _=None):
        if not self.actionq:
            return lambda env: NoOp(env), 0
        return self.actionq.popleft(), 0

    def no_op(self):
        self.actionq.append(lambda env: NoOp(env))
        return 0

    def build_overlord(self, obs):
        for base in obs.bases:
            obs.focus = base
            if BuildOverlord in obs.available_actions:
                self.actionq.append(lambda env: BuildOverlord(env))
                used, available = obs.supply
                return used - available
        return -1

    def build_base(self, obs):
        if self.base_count >= 20:
            return -1

        for base in obs.bases:
            if base.minerals.drones > 1:
                obs.focus = base.minerals
                if BuildBase in obs.available_actions:
                    self.actionq.append(lambda env: BuildBase(env))
                    self.base_count += 1
                    return 1
        return -1

    def build_drone(self, obs):
        for base in obs.bases:
            if base.larva >= 1:
                obs.focus = base
                if BuildDrone in obs.available_actions:
                    self.actionq.append(lambda env: BuildDrone(env))
                    return base.minerals.equiv_max - base.minerals.drones
        return -1

    def build_spawning_pool(self, obs):
        for base in obs.bases:
            if base.minerals.drones > 1:
                obs.focus = base.minerals
                if BuildSpawningPool in obs.available_actions:
                    self.actionq.append(lambda env: BuildSpawningPool(env))
                    return 1
        return -1

    def build_queen(self, obs):
        for base in obs.bases:
            if base.queens < 1:
                if obs.spawning_pool:
                    obs.focus = base
                    if BuildQueen in obs.available_actions:
                        self.actionq.append(lambda env: BuildQueen(env))
                        return 1
                else:
                    return self.build_spawning_pool(obs)
        return -1

    def inject_larva(self, obs):
        for base in obs.bases:
            obs.focus = base
            if InjectLarva in obs.available_actions:
                self.actionq.append(lambda env: InjectLarva(env))
                return 1
        return -1


    # takes in the available actions from the observation (should be a list of action_ids) and returns a list of 0's and 1's with respect to our action space.
    # 0 if the i_th action is not available, 1 if it is available. 
    def check_available_actions(self, env):
        #avalable functions: build_hatch, build_geyser, train_drone, train_overlord, train_queen, inject_larva, move_screen1, move_screen2, move_screen3, move_screen4, harvest_mins, harvest_gas        
        actions = [0]*len(self.choices)

        larva_available = 0
        gas_flag = True
        ext_flag = False
        queen_flag = False
        injectable = False

        for base in env.bases:
            larva_available += base.larva
            if base.queens > 0:
                queen_flag = True
            if base.injectable:
                injectable = True

        used, supply = env.supply
        supply_available = supply - used

        #hatch check
        if env.minerals >= 300 and len(env.bases) < 20:  # flag?
            actions[ActionEnum.build_base] = 1

        #drone conditions
        if env.minerals >= 50 and larva_available > 0 and supply_available > 0:
            actions[ActionEnum.build_drone] = 1

        #overlord conditions
        if env.minerals >= 100 and larva_available > 0:
            actions[ActionEnum.build_overlord] = 1

        #queen conditions
        if env.minerals >= 125 and env.spawning_pool and supply_available > 1:
            actions[ActionEnum.build_queen] = 1

        #inject check
        if injectable:
            actions[ActionEnum.inject_larva] = 1

        actions[ActionEnum.no_op] = 1

        return actions