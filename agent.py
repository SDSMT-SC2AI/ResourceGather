#-------------------------------------------------------------------------------
# @package agent This is essentially a wrapper for BaseAgent which PySC2 expects
# you to implement
#
from base_agent import BaseAgent
import observer
import network
hidden_layer_size = (3*observer.observation_size)//2

policy_spec = network.Policy.policy_spec(
            input_size=20,
            num_actions=800,
            max_episodes=2500,
            q_range=(30, 31),
            hidden_layer_size=30,
            base_explore_rate=0.1,                 
            min_explore_rate=0.01)
trainer_spec = network.Trainer.trainer_spec()


##
## @brief      Smart class. Necessary for PySC2. Updates policy_spec and sets up
##             the action_space and flags members. Also stores the
##             process_observation function which is a necessary component for
##             PySC2.
##
class Smart(BaseAgent):


    ##
    ## @brief      Constructs the object. Updates the policy_spec and sets
    ##             member variables.
    ##
    ## @param      self          The object
    ## @param      name          The name
    ## @param      parent        The parent
    ## @param      optimizer     The optimizer
    ## @param      episode       The episode
    ## @param      action_space  The action space
    ## @param      flags         The flags
    ##
    def __init__(self, name, parent, optimizer, episode, action_space, flags):
        policy_spec.update(action_space.action_spec)
        super().__init__(name, parent, optimizer, network, episode, policy_spec, trainer_spec)
        self.action_space = action_space
        self.flags = flags




    ##
    ## @brief      This is the process observation wrapper function that PySC2
    ##             expects to be with the
    ##
    ## @param      self  The object
    ## @param      obs   The observation
    ##
    ## @return     Returns the reward, observation, and if the episode has ended
    ##
    def process_observation(self, obs):
        return observer.process_observation(obs, self.action_space, self.flags)
