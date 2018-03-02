'''
This contains a class designed to limit the action space for training purposes,
as well as the functions needed to exicute the actions allowed in this space.
'''
import os
import sys
from collections import deque
from absl import flags
from absl.flags import FLAGS

from pysc2.env import sc2_env
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.maps import ladder
from pysc2.lib import features
from pysc2.lib import point

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

#feature info
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

#action arguments
_NOT_QUEUED = [0]
_QUEUED = [1]
_PLAYER_SELF = 1

#TODO even though the screen coords are what the game give you they dont work for building!? just need to fiddle more with the values
#expansion map coordinites for moving the camera and the screen coords for the hatch
#Top left points
_BOTTOM_START = [52,50]
_BOTTOM_SECOND = [42,48]
_BOTTOM_SECOND_SCREEN = [34, 30]
_BOTTOM_THIRD = [51, 41]
_BOTTOM_THIRD_SCREEN = [27, 29]
_BOTTOM_FOURTH = [42, 41]
_BOTTOM_FOURTH_SCREEN = [30, 30]
#Bottom right points
_TOP_START = [12, 13]
_TOP_SECOND = [22,18]
_TOP_SECOND_SCREEN = [44, 42]
_TOP_THIRD = [13,26]
_TOP_THIRD_SCREEN = [40, 43]
_TOP_FOURTH = [22,26]
_TOP_FOURTH_SCREEN = [37, 42]

class Action_Space:
    # define the units and their actions
    valid_units = {126: ("Move_screen", "Effect_InjectLarva_screen"), #queen
       104: ("Move_screen", "Harvest_Gather_screen", "Build_Hatchery_screen", "Build_SpawningPool_screen", "Build_Extractor_screen"), #drone
       151: ("Train_Drone_quick", "Train_Overlord_quick"), #larva
       86: ("Train_Queen_quick")}  # hatchery
    pool_flag = False
    top_left = True

    #Setup the action and point ques
    def __init__(self):
        self.busy_units = {}
        self.actionq = deque([])
        self.pointq = deque([])
        return

    #A quick check for which side of the map youre on
    def Start_pos(self, obs):
        start_y, start_x = (obs.observation["minimap"][_PLAYER_RELATIVE]).nonzero()
        if start_y.mean() <= 31:
            self.top_left = True
        else:
            self.top_left = False

    #This function gets the next action out of the que and returns it
    def action_step(self):
        if self.actionq:
            action = self.actionq.popleft()
        else:
            action = "No_Op"

        #only some actions need a point
        if ((action == "Select_Point_screen")
           | (action == "Effect_InjectLarva_screen")
           | (action == "Harvest_Gather_screen")
           | (action == "Build_Hatchery_screen")
           | (action == "Build_Extractor_screen")
           | (action == "Build_SpawningPool_screen")
           | (action == "move_camera")):
            target = self.pointq.popleft()
        else:
            target = [0,0]

        return {
            "Build_Extractor_screen" : actions.FunctionCall(_BUILD_GAS, [_NOT_QUEUED, target]),
            "Build_Hatchery_screen" : actions.FunctionCall(_BUILD_HATCHERY, [_NOT_QUEUED, target]),
            "Build_SpawningPool_screen" : actions.FunctionCall(_BUILD_POOL, [_NOT_QUEUED, point]),
            "Effect_InjectLarva_screen" : actions.FunctionCall(_INJECT_LARVA, [_NOT_QUEUED, target]),
            "Harvest_Gather_screen" : actions.FunctionCall(_GATHER_RESOURCES, [_NOT_QUEUED, target]),
            "Train_Drone_quick" : actions.FunctionCall(_TRAIN_DRONE, [_NOT_QUEUED]),
            "Train_Overlord_quick" : actions.FunctionCall(_TRAIN_OVERLORD, [_NOT_QUEUED]),
            "Train_Queen_quick" : actions.FunctionCall(_TRAIN_QUEEN, [_NOT_QUEUED]),     
            "Select_Point_screen" : actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target]),
            "move_camera" : actions.FunctionCall( _MOVE_CAMERA , [target]),
            "No_Op" : actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
            }[action]

    # Action space functions
    #build a hatchery in the selected expo
    def build_Hatchery(self, obs, expo_num):
        #first select a drone
        units = obs.observation["screen"][_UNIT_TYPE]
        unit_y, unit_x = (units == _DRONE).nonzero()
        target = [unit_x[0], unit_y[0]]
        self.pointq.append(target)
        self.actionq.append("Select_Point_screen")
        #get the coords for the choosen base and build there
        if self.top_left:
            map_target = {
                2 : _TOP_SECOND,
                3 : _TOP_THIRD,
                4 : _TOP_FOURTH
                }[expo_num]
            screen_target = {
                2 : _TOP_SECOND_SCREEN,
                3 : _TOP_THIRD_SCREEN,
                4 : _TOP_FOURTH_SCREEN
                }[expo_num]
        else:
           map_target = {
                2 : _BOTTOM_SECOND,
                3 : _BOTTOM_THIRD,
                4 : _BOTTOM_FOURTH
                }[expo_num]
           screen_target = {
                2 : _BOTTOM_SECOND_SCREEN,
                3 : _BOTTOM_THIRD_SCREEN,
                4 : _BOTTOM_FOURTH_SCREEN
                }[expo_num]
        #need to move the camera then build
        self.pointq.append(map_target)
        self.actionq.append("move_camera")
        self.pointq.append(screen_target)
        self.actionq.append("Build_Hatchery_screen")

    #build on the closet geyser   
    def build_Gas_Gyser(self, obs, drone_id):
        #select a drone
        units = obs.observation["screen"][_UNIT_TYPE]
        unit_y, unit_x = (units == _DRONE).nonzero()
        target = [unit_x[0], unit_y[0]]
        self.pointq.append(target)
        self.actionq.append("Select_Point_screen")

        #find an unused geyser and build on it
        unit_y, unit_x = (units == 342).nonzero()
        half = len(unit_x)/2
        #only works for spawning in the bottom, need to figure out how to average location for placement
        target = [unit_x[:int(half)].mean(), unit_y[:int(half)].mean()]
        self.pointq.append(target)
        self.actionq.append("Build_Extractor_screen")

    #build a spawning pool to the left of your current hatch
    def build_Spawning_Pool(self, obs, drone_id):
        #step on is selecting a dron to build with
        units = obs.observation["screen"][_UNIT_TYPE]
        unit_y, unit_x = (units == _DRONE).nonzero()
        #grab the first drone for now
        target = [unit_x[0], unit_y[0]]
        self.pointq.append(target)
        self.actionq.append("Select_Point_screen")

        #then build the pool
        units = obs.observation["screen"][_UNIT_TYPE]
        unit_y, unit_x = (units == _HATCHERY).nonzero()
        #location is just left of the nearest hatchery
        target = [unit_x.mean()-4, unit_y.mean()]
        self.pointq.append(target)
        self.actionq.append("Build_SpawningPool_screen")
        self.pool_flag = True
        
    #send a selected drone to harvest minerals
    def harvest_Minerals(self, obs, drone_id):
        #select a drone
        units = obs.observation["screen"][_UNIT_TYPE]
        unit_y, unit_x = (units == _DRONE).nonzero()
        target = [unit_x[0], unit_y[0]]
        self.pointq.append(target)
        self.actionq.append("Select_Point_screen")

        #find a mineral patch and que clicking it
        units = obs.observation["screen"][_UNIT_TYPE]
        unit_y, unit_x = (units == 341).nonzero()
        target = [unit_x[0], unit_y[0]]
        self.pointq.append(target)
        self.actionq.append("Harvest_Gather_screen")

    #send aselected drone to harvest gas
    def harvest_Gas(self, obs, drone_id):
        #select a drone
        units = obs.observation["screen"][_UNIT_TYPE]
        unit_y, unit_x = (units == _DRONE).nonzero()
        target = [unit_x[0], unit_y[0]]
        self.pointq.append(target)
        self.actionq.append("Select_Point_screen")

        #find an extractor and que clicking it
        units = obs.observation["screen"][_UNIT_TYPE]
        unit_y, unit_x = (units == 88).nonzero()
        target = [unit_x.mean(), unit.mean()]
        self.pointq.append(target)
        self.actionq.append("Harvest_Gather_screen")

    #try and find a queen and inject with it
    def inject_Larva(self, obs, queen_id):
        #select a queen
        units = obs.observation["screen"][_UNIT_TYPE]
        unit_y, unit_x = (units == _QUEEN).nonzero()
        target = [unit_x[0], unit_y[0]]
        self.pointq.append(target)
        self.actionq.append("Select_Point_screen")

        #find a hatchery and inject it
        units = obs.observation["screen"][_UNIT_TYPE]
        unit_y, unit_x = (units == _HATCHERY).nonzero()
        target = [unit_x.mean(), unit_y.mean()]
        self.pointq.append(target)
        self.actionq.append("Effect_InjectLarva_screen")
    
    #a function for moving between your expos    
    def move_Screen(self, obs, base_num):
       if self.top_left:
            map_target = {
                1 : _TOP_START,
                2 : _TOP_SECOND,
                3 : _TOP_THIRD,
                4 : _TOP_FOURTH
                }[base_num]
       else:
           map_target = {
                1 : _BOTTOM_START,
                2 : _BOTTOM_SECOND,
                3 : _BOTTOM_THIRD,
                4 : _BOTTOM_FOURTH
                }[base_num]
       self.pointq.append(map_target)
       self.actionq.append("move_camera")

    #train a drone at the current hatch
    def train_Drone(self, obs):
        #find larva position
        units = obs.observation["screen"][_UNIT_TYPE]
        unit_y, unit_x = (units == _LARVA).nonzero()
        if unit_x.size < 1:
            return
        target = [unit_x[0], unit_y[0]]

        #que clicking a larva and morphing it to a drone
        self.pointq.append(target)
        self.actionq.append("Select_Point_screen")
        self.actionq.append("Train_Drone_quick")
    
    #train a overlord at the current hatch
    def train_Overlord(self, obs):
        #find larva position
        units = obs.observation["screen"][_UNIT_TYPE]
        unit_y, unit_x = (units == _LARVA).nonzero()
        if unit_x.size < 1:
            return
        target = [unit_x[0], unit_y[0]]

        #que clicking a larva and morphing it to a overlord
        self.pointq.append(target)
        self.actionq.append("Select_Point_screen")
        self.actionq.append("Train_Overlord_quick")

    #train a queen at the currect hatch
    def train_Queen(self, obs):
        #if no pool is built redirect to building it instead
        if self.pool_flag == False:
            self.build_Spawning_Pool(0, obs)
            return

        #select a hatchery
        units = obs.observation["screen"][_UNIT_TYPE]
        unit_y, unit_x = (units == _HATCHERY).nonzero()
        target = [unit_x.mean(), unit_y.mean()]
        self.pointq.append(target)
        self.actionq.append("Select_Point_screen")
        #que a queen
        self.actionq.append("Train_Queen_quick")
        
    # compare the provided ID number to the ID of the units allowed
    def get_actions(self, unit_id):
        # return the valid actions for the given unit 
        if unit_id in self.valid_units:
            return self.valid_units[unit_id]
        return "Unit not defined in action space"
    #function for checking if drones are doing a non-interuptable task
    def drone_busy(drone_id):
        if drone_id in busy_units:
            return True
        return False
    #function for finding valid expos
    def get_base_coord():
        #need to find a method for getting base expansion coordinates.
        return [0,0]


def main():
    return

if __name__ == '__main__':
    main()
