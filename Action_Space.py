'''
This contains a class designed to limit the action space for training purposes,
as well as the functions needed to exicute the actions allowed in this space.
'''
import os
import sys
from collections import deque
from absl import flags
from absl.flags import FLAGS
#from sklearn.cluster import KMeans

from pysc2.env import sc2_env
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.maps import ladder
from pysc2.lib import features
from pysc2.lib import point
from enum import Enum

from common.helper_functions import GetUnits

#units
_DRONE = 104
_HATCHERY = 86
_LARVA = 151
_QUEEN = 126

#Actions
_BUILD_POOL = actions.FUNCTIONS.Build_SpawningPool_screen.id
_BUILD_HATCHERY = actions.FUNCTIONS.Build_Hatchery_screen.id
_BUILD_GAS = actions.FUNCTIONS.Build_Extractor_screen.id
_GATHER_RESOURCES = actions.FUNCTIONS.Harvest_Gather_Drone_screen.id
_INJECT_LARVA = actions.FUNCTIONS.Effect_InjectLarva_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_SELECT_LARVA = actions.FUNCTIONS.select_larva.id
_TRAIN_DRONE = actions.FUNCTIONS.Train_Drone_quick.id
_TRAIN_QUEEN = actions.FUNCTIONS.Train_Queen_quick.id
_TRAIN_OVERLORD = actions.FUNCTIONS.Train_Overlord_quick.id
_MOVE_CAMERA = actions.FUNCTIONS.move_camera.id
_SMART = actions.FUNCTIONS.Smart_screen.id
_MULTI_SELECT = actions.FUNCTIONS.select_rect.id
_MAX_AVAIL_ACTIONS = 8


#feature info
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

#action arguments
_NOT_QUEUED = [0]
_QUEUED = [1]
_PLAYER_SELF = 1
_PLAYER_MINERALS = 1
_SUPPLY_USED = 3
_SUPPLY_CAP = 4
_LARVA_AVALABLE = 10

# expansion map coordinites for moving the camera and the screen coords for the hatch
# Top left points
_BOTTOM_START = [52, 50]
_BOTTOM_SECOND = [41, 48]
_BOTTOM_SECOND_SCREEN = [40, 33]
_BOTTOM_THIRD = [51, 41]
_BOTTOM_THIRD_SCREEN = [30, 27]
_BOTTOM_FOURTH = [42, 41]
_BOTTOM_FOURTH_SCREEN = [30, 30]


# Bottom right points
_TOP_START = [10, 17]
_TOP_SECOND = [22, 18]
_TOP_SECOND_SCREEN = [43, 43]
_TOP_THIRD = [13, 26]
_TOP_THIRD_SCREEN = [43, 42]
_TOP_FOURTH = [22, 26]
_TOP_FOURTH_SCREEN = [43, 37]

# Top Geyser
_TOP_GEYSER = [64,16]
# 2nd Top Geyser
_TOP_2ND_GEYSER = [24,48]

class ActionEnum():
    """Enumerator for alliance"""
    build_Hatchery = 0
    build_Gas_Gyser = 1
    train_Drone = 2
    train_Overlord = 3
    train_Queen = 4
    inject_Larva = 5
    harvest_Minerals = 6
    harvest_Gas = 7
    no_op = 8

class Action_Space:
    # define the units and their actions
    valid_units = {126: ("Move_screen", "Effect_InjectLarva_screen"), #queen
       104: ("Move_screen", "Harvest_Gather_screen", "Build_Hatchery_screen", "Build_SpawningPool_screen", "Build_Extractor_screen"), #drone
       151: ("Train_Drone_quick", "Train_Overlord_quick"), #larva
       86: ("Train_Queen_quick")}  # hatchery
    pool_flag = False
    top_left = True
    action_spec = {
    'num_actions': _MAX_AVAIL_ACTIONS # Actions should be in a dict or something so we can run len() etc. on them
    }

    def __init__(self):
        self.busy_units = {}
        self.actionq = deque(["No_Op"]*10)
        self.pointq = deque([])
        self.expo_count = 0
        self.action_Dict = {
            ActionEnum.build_Hatchery   : self.build_Hatchery,
            ActionEnum.build_Gas_Gyser  : self.build_Gas_Gyser,
            ActionEnum.train_Drone      : self.train_Drone,
            ActionEnum.train_Overlord   : self.train_Overlord,
            ActionEnum.train_Queen      : self.train_Queen,
            ActionEnum.inject_Larva     : self.inject_Larva,
            ActionEnum.harvest_Minerals : self.harvest_Minerals,
            ActionEnum.harvest_Gas      : self.harvest_Gas,
            ActionEnum.no_op            : lambda a, b, c=None: self.actionq.append("No_Op")
        }

        #avalable functions: build_hatch, build_geyser, train_drone, train_overlord, train_queen, inject_larva, move_screen1, move_screen2, move_screen3, move_screen4, harvest_mins, harvest_gas
        return

    def Start_pos(self, obs):
        start_y, start_x = (obs.observation["minimap"][_PLAYER_RELATIVE]).nonzero()
        if start_y.mean() <= 31:
            self.top_left = True
        else:
            self.top_left = False


    # takes in the available actions from the observation (should be a list of action_ids) and returns a list of 0's and 1's with respect to our action space.
    # 0 if the i_th action is not available, 1 if it is available. 
    def check_available_actions(self, obs):
        #avalable functions: build_hatch, build_geyser, train_drone, train_overlord, train_queen, inject_larva, move_screen1, move_screen2, move_screen3, move_screen4, harvest_mins, harvest_gas        
        player_info = obs.observation["player"]
        units = obs.observation["screen"][_UNIT_TYPE]
        actions = [0]*len(self.action_Dict)

        larva_Available = len(GetUnits(_LARVA, obs.raw_obs.raw_data.units))# - self.actionq.count("Train_Drone_quick") - self.actionq.count("Train_Overlord_quick")
        #TODO get queen and gas info from feture layers
        queen_flag = False
        hatcheries = GetUnits(_HATCHERY, obs.raw_obs.raw_data.units)
        if len(GetUnits(_QUEEN, obs.raw_obs.raw_data.units)) != 0:
            queen_flag = True
        drone_flag = False
        if len(GetUnits(_DRONE, obs.raw_obs.raw_data.units)) != 0:
            drone_flag = True
        hatch_flag = False
        if len(hatcheries) != 0:
            hatch_flag = True
         
        gas_flag = True
        ext_flag = False
        ext_y, ext_x = (units == 88).nonzero()
        if len(ext_y) != 0:
            ext_flag = True

        supply_Available = player_info[_SUPPLY_CAP] - player_info[_SUPPLY_USED]

        #hatch check
        if(player_info[_PLAYER_MINERALS]>=300 and drone_flag and self.expo_count < 3):
            actions[ActionEnum.build_Hatchery] = 1

        #geyser check
        if(player_info[_PLAYER_MINERALS]>=25 and drone_flag and ext_flag and gas_flag):
            actions[ActionEnum.build_Gas_Gyser] = 1

        #drone conditions
        if(player_info[_PLAYER_MINERALS]>=50 and larva_Available > 0 and supply_Available > 0):
            actions[ActionEnum.train_Drone] = 1

        #overlord conditions
        if(player_info[_PLAYER_MINERALS]>=100 and larva_Available > 0):
            actions[ActionEnum.train_Overlord] = 1

        #queen conditions
        if(player_info[_PLAYER_MINERALS]>=200 and hatch_flag and supply_Available > 1):
            actions[ActionEnum.train_Queen] = 1

        #inject check
        if(queen_flag):
            actions[ActionEnum.inject_Larva] = 1

        #min check
        if(drone_flag):
            actions[ActionEnum.harvest_Minerals] = 1
            actions[ActionEnum.harvest_Gas] = 1

        actions[ActionEnum.no_op] = 1

        return actions

    #takes an integer action index  (corresponding to the i_th action in the action space) and returns 1 if the action is available and can be added to the queue, -1 if not.
    def act(self, index, obs, drone_id):
        avalable = self.check_available_actions(obs)
        
        print("Action: ", index)
        if(avalable[index] == 1):
            self.action_Dict[index](obs, drone_id)
            return 1
        else:
            return -1

    def action_step(self):
        if self.actionq:
            action = self.actionq.popleft()
        else:
            action = "No_Op"

        target = [0,0]
        lhs = [0,0]
        rhs = [0,0]

        if ((action == "Select_Point_screen")
           | (action == "Effect_InjectLarva_screen")
           | (action == "Smart_Click")
           | (action == "Build_Hatchery_screen")
           | (action == "Build_Extractor_screen")
           | (action == "Build_SpawningPool_screen")
           | (action == "move_camera")):
            target = self.pointq.popleft()
            # print("Action step target: ", target)
        elif action == "multi_select":
            lhs = self.pointq.popleft()
            rhs = self.pointq.popleft()

        return {
            "Build_Extractor_screen" : lambda target, lhs, rhs: (_BUILD_GAS, [_NOT_QUEUED, target]),
            "Build_Hatchery_screen" : lambda target, lhs, rhs: (_BUILD_HATCHERY, [_NOT_QUEUED, target]),
            "Build_SpawningPool_screen" : lambda target, lhs, rhs: (_BUILD_POOL, [_NOT_QUEUED, target]),
            "Effect_InjectLarva_screen" : lambda target, lhs, rhs: (_INJECT_LARVA, [_NOT_QUEUED, target]),
            "Smart_Click" : lambda target, lhs, rhs: (_SMART, [_NOT_QUEUED, target]),
            "select_larva" : lambda target, lhs, rhs: (_SELECT_LARVA, []),
            "Train_Drone_quick" : lambda target, lhs, rhs: (_TRAIN_DRONE, [_NOT_QUEUED]),
            "Train_Overlord_quick" : lambda target, lhs, rhs: (_TRAIN_OVERLORD, [_NOT_QUEUED]),
            "Train_Queen_quick" : lambda target, lhs, rhs: (_TRAIN_QUEEN, [_NOT_QUEUED]),     
            "Select_Point_screen" : lambda target, lhs, rhs: (_SELECT_POINT, [_NOT_QUEUED, target]),
            "multi_select" : lambda target, lhs, rhs: (_MULTI_SELECT, [_NOT_QUEUED, lhs, rhs]),
            "move_camera" : lambda target, lhs, rhs: ( _MOVE_CAMERA , [target]),
            "No_Op" : lambda target, lhs, rhs: (actions.FUNCTIONS.no_op.id, [])
            }[action](target, lhs, rhs)

    # Action space functions
    def build_Hatchery(self, obs, drone_id):
        #first select a drone
        units = obs.observation["screen"][_UNIT_TYPE]
        unit_y, unit_x = (units == _DRONE).nonzero()
        if len(unit_x) == 0:
            return

        lhs_corner = [unit_x[0], unit_y[0]]
        rhs_corner = [unit_x[0] + 1, unit_y[0] + 1]
        self.pointq.append(lhs_corner)
        self.pointq.append(rhs_corner)
        self.actionq.append("multi_select")
        # target = [unit_x[0], unit_y[0]]
        # self.pointq.append(target)
        # self.actionq.append("Select_Point_screen")
        #get the coords for the next base and build there
        if self.top_left:
            map_target = {
                0 : _TOP_SECOND,
                1 : _TOP_THIRD,
                2 : _TOP_FOURTH
                }[self.expo_count]
            screen_target = {
                0 : _TOP_SECOND_SCREEN,
                1 : _TOP_THIRD_SCREEN,
                2 : _TOP_FOURTH_SCREEN
                }[self.expo_count]
        else:
           map_target = {
                0 : _BOTTOM_SECOND,
                1 : _BOTTOM_THIRD,
                2 : _BOTTOM_FOURTH
                }[self.expo_count]
           screen_target = {
                0 : _BOTTOM_SECOND_SCREEN,
                1 : _BOTTOM_THIRD_SCREEN,
                2 : _BOTTOM_FOURTH_SCREEN
                }[self.expo_count]
        
        self.pointq.append(map_target)
        self.actionq.append("move_camera")
        # for _ in range(15):
            # self.actionq.append("No_Op")
        

        self.pointq.append(screen_target)
        self.actionq.append("Build_Hatchery_screen")
        # for _ in range(3000):
            # self.actionq.append("No_Op")
        self.pointq.append(_TOP_START)
        self.actionq.append("move_camera")
        self.expo_count += 1

    # Builds a geyser if one is taken it builds the other
    def build_Gas_Gyser(self, obs, drone_id):
        #select a drone
        units = obs.observation["screen"][_UNIT_TYPE]
        unit_y, unit_x = (units == _DRONE).nonzero()
        if len(unit_x) == 0:
            return
        target = [unit_x[0], unit_y[0]]
        self.pointq.append(target)
        self.actionq.append("Select_Point_screen")

        #find an unused geyser and build on it
        unit_y, unit_x = (units == 342).nonzero()
        half = len(unit_x)/2
        #only works for spawning in the bottom, need to figure out how to average location for placement
        target1 = [unit_x[:int(half)].mean(), unit_y[:int(half)].mean()]
        target2 = [unit_x[int(half):].mean(), unit_y[int(half):].mean()]
        #find built geysers
        unit_y, unit_x = (units == 88).nonzero()
        if len(unit_x) == 0:
            return
        if(target1 == [unit_x.mean(), unit_y.mean()]):
            self.pointq.append(target2)
        else:
            self.pointq.append(target1)
        self.actionq.append("Build_Extractor_screen")

    def build_Spawning_Pool(self, obs, drone_id):
        #step on is selecting a dron to build with
        units = obs.observation["screen"][_UNIT_TYPE]
        pool = GetUnits(89, obs.raw_obs.raw_data.units) # ZERG_SPAWNINGPOOL
        if len(pool) != 0 and pool[0].build_progress == 1.0:
            self.pool_flag = True
            return True
        
        unit_y, unit_x = (units == _DRONE).nonzero()
        #grab the first drone for now
        if len(unit_x) == 0:
            return False
        target = [unit_x[0], unit_y[0]]
        self.pointq.append(target)
        self.actionq.append("Select_Point_screen")

        #then build the pool
        units = obs.observation["screen"][_UNIT_TYPE]
        unit_y, unit_x = (units == _HATCHERY).nonzero()
        if len(unit_x) == 0:
            return False
        #location is just left of the nearest hatchery
        target = [unit_x.mean()-12, unit_y.mean()]
        self.pointq.append(target)
        self.actionq.append("Build_SpawningPool_screen")
        return False
    
    #TODO integrate drone ID tags
    def harvest_Minerals(self, obs, drone_id):
        #select a drone
        units = obs.observation["screen"][_UNIT_TYPE]
        unit_y, unit_x = (units == _DRONE).nonzero()
        if len(unit_x) == 0:
            return

        lhs_corner = [unit_x[0], unit_y[0]]
        rhs_corner = [unit_x[0] + 1, unit_y[0] + 1]
        self.pointq.append(lhs_corner)
        self.pointq.append(rhs_corner)
        self.actionq.append("multi_select")

        # target = [unit_x[0], unit_y[0]]
        # self.pointq.append(target)
        # self.actionq.append("Select_Point_screen")

        #find a mineral patch and que clicking it
        units = obs.observation["screen"][_UNIT_TYPE]
        unit_y, unit_x = (units == 341).nonzero()
        if len(unit_x) == 0:
            return
        target = [unit_x[0], unit_y[0]]
        self.pointq.append(target)
        self.actionq.append("Smart_Click")


    #TODO integrate drone ID tags
    def harvest_Gas(self, obs, drone_id):
        #select a drone
        units = obs.observation["screen"][_UNIT_TYPE]
        unit_y, unit_x = (units == _DRONE).nonzero()
        if len(unit_x) == 0:
            return

        # Centering ourselves
        self.pointq.append(_TOP_START)
        self.actionq.append("move_camera")

        # Select rectangle that encompases a drone
        lhs_corner = [unit_x[0], unit_y[0]]
        rhs_corner = [unit_x[0] + 1, unit_y[0] + 1]
        self.pointq.append(lhs_corner)
        self.pointq.append(rhs_corner)
        self.actionq.append("multi_select")

        # select geyser point
        self.pointq.append(_TOP_GEYSER)
        self.actionq.append("Build_Extractor_screen")

        #find an extractor and que clicking it
        # units = obs.observation["screen"][_UNIT_TYPE]
        # unit_y, unit_x = (units == 88).nonzero()
        # if len(unit_x) == 0:
        #     return
        # target = [unit_x.mean(), unit_y.mean()]
        # self.pointq.append(target)        
        # self.actionq.append("Smart_Click")

    def inject_Larva(self, obs, queen_id):
        #select a queen
        units = obs.observation["screen"][_UNIT_TYPE]
        unit_y, unit_x = (units == _QUEEN).nonzero()
        if len(unit_x) == 0:
            return
        target = [unit_x[0], unit_y[0]]
        self.pointq.append(target)
        self.actionq.append("Select_Point_screen")

        #find a hatchery and inject it
        units = obs.observation["screen"][_UNIT_TYPE]
        unit_y, unit_x = (units == _HATCHERY).nonzero()
        if len(unit_x) == 0:
            return
        target = [unit_x.mean(), unit_y.mean()]
        self.pointq.append(target)
        self.actionq.append("Effect_InjectLarva_screen")
        
    def move_Screen1(self, obs, base_num):
       if self.top_left:
            map_target = _TOP_START
       else:
           map_target = _BOTTOM_START
       self.pointq.append(map_target)
       self.actionq.append("move_camera")

    def move_Screen2(self, obs, base_num):
       if self.top_left:
            map_target = _TOP_SECOND
       else:
           map_target = _BOTTOM_SECOND
       self.pointq.append(map_target)
       self.actionq.append("move_camera")

    def move_Screen3(self, obs, base_num):
       if self.top_left:
            map_target = _TOP_THIRD
       else:
           map_target = _BOTTOM_THIRD
       self.pointq.append(map_target)
       self.actionq.append("move_camera")

    def move_Screen4(self, obs, base_num):
       if self.top_left:
            map_target =  _TOP_FOURTH
       else:
           map_target = _BOTTOM_FOURTH
       self.pointq.append(map_target)
       self.actionq.append("move_camera")

    def train_Drone(self, obs, larva_id):
        #find larva position
        units = obs.observation["screen"][_UNIT_TYPE]
        unit_y, unit_x = (units == _LARVA).nonzero()
        if len(unit_x) == 0:
            return
        # target = [unit_x[0], unit_y[0]]

        #que clicking a larva and morphing it to a drone
        # self.pointq.append(target)
        # self.actionq.append("Select_Point_screen")
        unit_y, unit_x = (units == _HATCHERY).nonzero()
        if len(unit_x) == 0:
            return
        target = [unit_x.mean(), unit_y.mean()]
        self.pointq.append(target)
        self.actionq.append("Select_Point_screen")
        self.actionq.append("select_larva")
        self.actionq.append("Train_Drone_quick")        
                
    def train_Overlord(self, obs, drone_id):
        #find larva position
        units = obs.observation["screen"][_UNIT_TYPE]
        unit_y, unit_x = (units == _LARVA).nonzero()
        if len(unit_x) == 0:
            return
        target = [unit_x[0], unit_y[0]]

        #que clicking a larva and morphing it to a overlord
        self.pointq.append(target)
        self.actionq.append("Select_Point_screen")
        self.actionq.append("Train_Overlord_quick")

    def train_Queen(self, obs, hatch_id):
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
        
    #function for checking if drones are doing a non-interuptable task
    def drone_busy(drone_id):
        if drone_id in busy_units:
            return True
        return False



def main():
    return

if __name__ == '__main__':
    main()
