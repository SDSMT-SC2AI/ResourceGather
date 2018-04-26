##
## @package network This is the primary file for the A3C network. It contains
## the train class as well as the policy class
##
import tensorflow as tf
import numpy as np
from common.helper_functions import bisection, discount, select_from



##
## @brief      This is the trainer class for our network. It defines the
##             TensorFlow graph nodes that will be used to calculate loss, run
##             optimizers, and apply gradients to our network.
##
class Trainer:


    @staticmethod
    ##
    ## @brief      This is the trainer specifications.
    ##
    ## @param      accuracy_coefficient     The accuracy coefficient
    ## @param      advantage_coefficient    The advantage coefficient
    ## @param      consistency_coefficient  The consistency coefficient
    ## @param      max_grad_norm            The maximum graduated normalize, aka grad clipping value
    ## @param      discount_factor          The discount factor
    ##
    ## @return     Returns a dictionary of the items passed in.
    ##
    def trainer_spec(accuracy_coefficient=1.0,
                     advantage_coefficient=10.0,
                     consistency_coefficient=3.0,
                     max_grad_norm=40.0,
                     discount_factor=0.6):
        return {
            'accuracy coefficient': accuracy_coefficient,
            'advantage coefficient': advantage_coefficient,
            'consistency coefficient': consistency_coefficient,
            'max gradient norm': max_grad_norm,
            'discount factor': discount_factor
        }



    ##
    ## @brief      Constructs the object and initializes the TensorFlow graph.
    ##
    ## @param      self          The object
    ## @param      scope         The scope
    ## @param      optimizer     The optimizer
    ## @param      policy        The policy
    ## @param      trainer_spec  The trainer specifications
    ## @param      hyper_params  The hyper parameters
    ##
    def __init__(self, scope, optimizer, policy, trainer_spec, hyper_params):
        self.policy = policy
        self.hp = hyper_params
        with tf.variable_scope(scope):
            self.actions = tf.placeholder(shape=[None], dtype=tf.int64, name="actions")
            self.rewards = tf.placeholder(shape=[None], dtype=tf.float32)
            self.values = tf.placeholder(shape=[None], dtype=tf.float32)
            q = select_from(policy.q, self.actions)
            discounted_rewards = discount(self.rewards, self.hp.discount, self.values[-1])

            self.accuracy_loss = tf.reduce_mean(tf.square(q - discounted_rewards)) * (1 - self.hp.discount)
            self.consistent_loss = tf.reduce_mean(tf.square(q - self.values[1:]))
            self.advantage = tf.reduce_mean(q - tf.reduce_mean(policy.q, axis=1))
            self.loss = self.hp.accuracy_coef * self.accuracy_loss + self.hp.consist_coef * self.consistent_loss - self.hp.advantage_coef * self.advantage

            # Get gradients from local network using local losses
            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            # self.gradients - gradients of loss wrt local_vars
            self.gradients = tf.gradients(self.loss, local_vars)
            self.var_norms = tf.global_norm(local_vars)
            grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, self.hp.max_grad_norm)

            # Apply local gradients to global network
            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
            self.apply_grads = optimizer.apply_gradients(zip(grads, global_vars))




    ##
    ## @brief      The train function is called when the buffer is full or
    ##             episode quantity met. This calls for a TensorFlow session to
    ##             be ran. It passes in the feed_dict items and returns the
    ##             fetches list items. TensorFlow will know that it needs to
    ##             fetch self.loss so it will expect everything necessary to
    ##             calculate it in the feed_dict.
    ##
    ## @param      self     The object
    ## @param      sess     The sess
    ## @param      obs      The obs
    ## @param      actions  The actions
    ## @param      rewards  The rewards
    ## @param      values   The values
    ##
    ## @return     A list of lists.
    ##
    def train(self, sess, obs, actions, rewards, values):
        return sess.run(
            fetches=[[
                self.loss,
                self.accuracy_loss,
                self.consistent_loss,
                self.advantage,
                self.grad_norms,
                self.var_norms],  # returns
                [self.apply_grads]],  # ops
            feed_dict={
                self.policy.input: obs,
                self.policy.exploration_rate: self.policy.random_explore_rate,
                self.actions: actions,
                self.rewards: rewards,
                self.values: values}  # inputs
        )[0]





##
## @brief      The Policy class defines the basic structure of the network such
##             as how the hidden layers are setup.
##
class Policy:



    @staticmethod
    ##
    ## @brief      Creates a dictionary of policy_spec values to be returned.
    ##
    ## @param      input_size         The input size
    ## @param      num_actions        The number actions
    ## @param      max_episodes       The maximum episodes
    ## @param      q_range            The quarter range
    ## @param      hidden_layer_size  The hidden layer size
    ## @param      base_explore_rate  The base explore rate
    ## @param      min_explore_rate   The minimum explore rate
    ##
    ## @return     Returns a dictionary of the items passed in.
    ##
    def policy_spec(input_size=None,
                    num_actions=None,
                    max_episodes=None,
                    q_range=None,
                    hidden_layer_size=None,
                    base_explore_rate=0.1,
                    min_explore_rate=None):
        return {
            "q_range": q_range,
            "input_size": input_size,
            "hidden_layer_size": hidden_layer_size or (3*input_size)//2,
            "num_actions": num_actions,
            "base_explore_rate": base_explore_rate,
            "min_explore_rate": min_explore_rate or base_explore_rate / 4,
            "max_episodes": max_episodes
        }




    ##
    ## @brief      Constructs the object. Defines the structure of the network
    ##             and sets up the TensorFlow graph. The input layer as well as
    ##             hidden layers and output are 
    ##
    ## @param      self          The object
    ## @param      scope         The scope
    ## @param      episode       The episode
    ## @param      policy_spec   The policy specifier
    ## @param      hyper_params  The hyper parameters
    ##
    def __init__(self, scope, episode, policy_spec, hyper_params):
        self.hyper_params = hyper_params
        self.network_spec = policy_spec
        self.random_explore_rate = np.random.beta(self.hyper_params.alpha, self.hyper_params.beta)

        with tf.variable_scope(scope):
            # Define the exploration rate reduction policy
            self.exploration_rate = tf.placeholder(tf.float32, shape=[])
            adjust = tf.minimum(1.0, tf.cast(episode, dtype=tf.float32) /
                                tf.cast(policy_spec['max_episodes'], dtype=tf.float32))
            factor = (1 - adjust) * self.network_spec['base_explore_rate'] + adjust * policy_spec['min_explore_rate']
            self.exploration = self.exploration_rate * factor

            # Define the neural net operations
            self.input = tf.placeholder(shape=[None, policy_spec['input_size']], dtype=tf.float32, name="input")

            # print("Hidden layer size: ", policy_spec['hidden_layer_size'])
            hidden1 = tf.contrib.layers.fully_connected(
                inputs=self.input,
                num_outputs=policy_spec['hidden_layer_size'],
                activation_fn=tf.nn.selu,
                biases_initializer=tf.random_uniform_initializer(-1, 1)
            )
            hidden2 = tf.contrib.layers.fully_connected(
                inputs=hidden1,
                num_outputs=policy_spec['hidden_layer_size'],
                activation_fn=tf.nn.selu,
                biases_initializer=tf.random_uniform_initializer(-1, 1)
            )
            hidden3 = tf.contrib.layers.fully_connected(
                inputs=hidden2,
                num_outputs=policy_spec['hidden_layer_size'],
                activation_fn=tf.nn.selu,
                biases_initializer=tf.random_uniform_initializer(-1, 1)
            )
            self.q = tf.contrib.layers.fully_connected(
                inputs=hidden3,
                num_outputs=policy_spec["num_actions"],
                weights_initializer=tf.orthogonal_initializer(0.1),
                biases_initializer=tf.random_uniform_initializer(*policy_spec['q_range'])
            )

            best = tf.argmax(self.q, axis=-1)
            p_best = 1 - self.exploration
            p_other = self.exploration / tf.cast(tf.shape(self.q)[-1], dtype=tf.float32)
            self.probs = tf.one_hot(best, tf.shape(self.q)[-1], dtype=tf.float32) * p_best + p_other

            self.action = tf.reshape(tf.multinomial(tf.log(self.probs), 1), [-1])
            self.value = tf.reduce_sum(self.probs * self.q, axis=1) / tf.reduce_sum(self.probs, axis=1)




    ##
    ## @brief      On an episode reset we want to reset the random explore rate.
    ##
    ## @param      self  The object
    ##
    ##
    def reset(self):
        self.random_explore_rate = np.random.beta(self.hyper_params.alpha, self.hyper_params.beta)




    ##
    ## @brief      The network needs to step through iterations with the
    ##             environment, this is the function to accomplish this. Runs a
    ##             TensorFlow session to calculate the action and value that
    ##             help steer the network and training.
    ##
    ## @param      self  The object
    ## @param      sess  The session
    ## @param      obs   The observation
    ##
    ## @return     A list containing the action and value.
    ##
    def step(self, sess, obs):
        return sess.run(fetches=[self.action, self.value],  # returns
                        feed_dict={self.input: obs,
                                   self.exploration_rate: self.random_explore_rate})  # input




    ##
    ## @brief      Gets the value by running a TensorFlow session and
    ##             calculating the value as it runs through the graph.
    ##
    ## @param      self  The object
    ## @param      sess  The session
    ## @param      obs   The observation
    ##
    ## @return     The value.
    ##
    def get_value(self, sess, obs):
        return sess.run(fetches=self.value,  # returns
                        feed_dict={self.input: obs,
                                   self.exploration_rate: self.random_explore_rate})  # input
