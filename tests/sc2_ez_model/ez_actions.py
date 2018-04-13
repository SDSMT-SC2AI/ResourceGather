from collections import deque
from .environment.actions import \
    BuildQueen, BuildOverlord, BuildDrone, BuildSpawningPool, \
    BuildBase, Select, InjectLarva, NoOp


class Action_Space:
    choices = [
        lambda self, obs: self.build_base(obs),
        lambda self, obs: self.build_drone(obs),
        lambda self, obs: self.build_queen(obs),
        lambda self, obs: self.build_overlord(obs),
        lambda self, obs: self.inject_larva(obs),
        lambda self, obs: self.no_op,
    ]
    action_spec = {
        'number of actions': len(choices)
    }

    def __init__(self):
        self.actionq = deque([])

    def act(self, choice, obs):
        if choice < len(self.choices):
            return self.choices[choice](self, obs[0])
        else:
            return self.choices[-1]

    def no_op(self):
        self.actionq.append(lambda env: NoOp(env))
        return 0

    def build_overlord(self, obs):
        for base in obs.bases:
            obs.focus = base
            if BuildOverlord in obs.available_actions:
                self.actionq.append(lambda env: Select(env, base))
                self.actionq.append(lambda env: BuildOverlord(env))
                return 0
            else:
                return -100
        return -100

    def build_base(self, obs):
        if obs.number_bases > 5:
            return -100

        for base in obs.bases:
            if base.minerals.drones > 1:
                obs.focus = base.minerals
                if BuildBase in obs.available_actions:
                    self.actionq.append(lambda env: Select(env, base.minerals))
                    self.actionq.append(lambda env: BuildBase(env))
                    return 0
                else:
                    return -100
        return -100

    def build_drone(self, obs):
        for base in obs.bases:
            if base.minerals.drones < base.minerals.equiv_max and base.larva >= 1:
                obs.focus = base
                if BuildDrone in obs.available_actions:
                    self.actionq.append(lambda env: Select(env, base))
                    self.actionq.append(lambda env: BuildDrone(env))
                    return 0
                else:
                    return -100
        return -100

    def build_spawning_pool(self, obs):
        for base in obs.bases:
            if base.minerals.drones > 1:
                obs.focus = base.minerals
                if BuildSpawningPool in obs.available_actions:
                    self.actionq.append(lambda env: Select(env, base.minerals))
                    self.actionq.append(lambda env: BuildSpawningPool(env))
                    return 0
                else:
                    return -100
        return -100

    def build_queen(self, obs):
        for base in obs.bases:
            if base.queens < 1:
                if obs.spawning_pool:
                    obs.focus = base
                    if BuildQueen in obs.available_actions:
                        self.actionq.append(lambda env: Select(env, base))
                        self.actionq.append(lambda env: BuildQueen(env))
                        return 0
                    else:
                        return -100
                else:
                    return self.build_spawning_pool(obs)
        return -100

    def inject_larva(self, obs):
        for base in obs.bases:
            obs.focus = base
            if InjectLarva in obs.available_actions:
                self.actionq.append(lambda env: Select(env, base))
                self.actionq.append(lambda env: InjectLarva(env))
                return 0
        return -100

