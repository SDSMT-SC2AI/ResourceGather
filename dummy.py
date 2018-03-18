# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A random agent for starcraft."""
# import numpy

# from pysc2.agents import base_agent
# from pysc2.lib import actions


# class Dummy(base_agent.BaseAgent):
#     """A random agent for starcraft."""
#     map_name = "AbyssalReefLE_RL"
    
#     def __init__(self):
#         super(Dummy, self).__init__()
#         self.step_num = 0


#     def step(self, obs):
#         super(Dummy, self).step(obs)

#         print(self.step_num)
#         self.step_num += 1
#         return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
