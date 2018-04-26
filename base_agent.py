##
## @package base_agent
##  This is the BaseAgent that PySC2 is expecting.
##

from common.helper_functions import update_target_graph


##
## @brief      This is the BaseAgent class that PySC2 expects. It contains
##             functions necessary to run an agent.
##
class BaseAgent:



    ##
    ## @brief      Constructs the object and initializes policy, trainer, and
    ##             update_local_policy members.
    ##
    ## @param      self          The object
    ## @param      name          The name
    ## @param      parent        The parent
    ## @param      optimizer     The optimizer
    ## @param      network       The network
    ## @param      episode       The episode
    ## @param      policy_spec   The policy specifier
    ## @param      trainer_spec  The trainer specifier
    ## @param      hyper_params  The hyper parameters
    ##
    def __init__(self, name, parent, optimizer, network, episode, policy_spec, trainer_spec, hyper_params):
        self.policy = network.Policy(name, episode, policy_spec, hyper_params)
        self.trainer = network.Trainer(name, optimizer, self.policy, trainer_spec, hyper_params)
        self.update_local_policy = update_target_graph(parent, name)




    ##
    ## @brief      This is a wrapper function used to have the trainer train.
    ##
    ## @param      self          The object
    ## @param      sess          The sess
    ## @param      actions       The actions
    ## @param      rewards       The rewards
    ## @param      observations  The observations
    ## @param      values        The values
    ##
    ## @return     Returns the results from network.py's train function
    ##
    def train(self, sess, actions, rewards, observations, values):
        return self.trainer.train(sess, observations, actions, rewards, values)



    ##
    ## @brief      Wrapper for policy.step.
    ##
    ## @param      self         The object
    ## @param      sess         The sess
    ## @param      observation  The observation
    ##
    ## @return     Returns the results from network.py's step function
    ##
    def step(self, sess, observation):
        choice, value = self.policy.step(sess, observation)
        return choice[0], value[0]



    ##
    ## @brief      Wrapper for policy.get_values.
    ##
    ## @param      self  The object
    ## @param      sess  The sess
    ## @param      obs   The obs
    ##
    ## @return     Returns the results from network.py's get_values function
    ##
    def value(self, sess, obs):
        return self.policy.get_value(sess, obs)[0]




    ##
    ## @brief      Makes a call to helper function update_target_graph, which
    ##             then gets fed into the currently running TensorFlow session.
    ##
    ## @param      self  The object
    ## @param      sess  The TensorFlow session
    ##
    ##
    def update_policy(self, sess):
        sess.run(self.update_local_policy)




    ##
    ## @brief      This function raises an error if called. We should not be
    ##             making a call to this particular process_observation.
    ##
    ## @param      self  The object
    ## @param      obs   The observation
    ##
    ##
    def process_observation(self, obs):
        raise NotImplementedError
