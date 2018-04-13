from .environment.model import IdealizedSC2Env
globals().update({a.__name__: a for a in IdealizedSC2Env.actions})

class Action_Space:
    choices = actions
    action_spec = {
        'number of actions': len(choices)
    }


    def act(choice):
        if choice in range(len(choices)):
            return 1, choices[choice]
        else:
            return -1, choices[0]


    def train_Queen(self, obs):
        #if no pool is built redirect to building it instead
        if not self.pool_flag:
            if not self.build_Spawning_Pool(obs, 0):
                return

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
