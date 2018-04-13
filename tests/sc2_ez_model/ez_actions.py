from collections import deque
from .environment.actions import \
    BuildQueen, BuildOverlord, BuildDrone, BuildSpawningPool, BuildBase, Select

class Action_Space:
    choices = [
        build_base
    ]
    action_spec = {
        'number of actions': len(choices)
    }

    def __init__(self):
        self.actionq = deque([])


    def act(self, choice, obs):
        if choice < len(self.choices):
            self.choices[choice](obs)
        else:
            

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

    def train_queen(self, obs):
        obs = obs[0]
        #if no pool is built redirect to building it instead
        for base in obs.bases:
            if base.queens < 1:
                if obs.spawning_pool:
                    if BuildQueen in obs.available_actions:
                        self.actionq.append(lambda env: Select(env, base))
                        self.actionq.append(lambda env: BuildQueen(env))
                        return 0
                    else:
                        return -100
                else:
                    return self.build_spawning_pool(obs)
        return -100



        #select a hatchery
        units = obs.observation["screen"][_UNIT_TYPE]
        unit_y, unit_x = (units == _HATCHERY).nonzero()
        if len(unit_x) == 0:
            return
        target = [unit_x.mean(), unit_y.mean()]
        self.pointq.append(target)
        self.actionq.append("Select_Point_screen")
        #que a queen
        self.actionq.append("Train_Queen_quick")
