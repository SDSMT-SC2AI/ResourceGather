"""A smart A2C agent for starcraft."""
from pysc2.agents import base_agent
from pysc2.lib import actions
from network import A2C

default_action_spec = {}
default_input_spec = {}

class Smart(base_agent.BaseAgent):
    """A random agent for starcraft."""
    map_name = "AbyssalReefLE_RL"

    def __init__(self, is_training=True):
        super().__init__()
        self.step_num = 0
        self.is_training = is_training
        self.input_spec = default_input_spec
        self.action_spec = default_action_spec
        self.controller = A2C(self.input_spec, self.action_spec)

    def step(self, obs):
        super().step(obs)
        action = self.controller.step(obs)
        print(self.step_num)
        self.step_num += 1
        return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])