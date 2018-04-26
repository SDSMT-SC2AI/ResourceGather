import tensorflow as tf
import numpy as np
from common.helper_functions import bisection, discount, select_from


class BaseTrainer:
    def __init__(self, scope, parent, optimizer, policy, *,
                 # key word arguments
                 discount_rate=0.99, consistency_factor=0.1, advantage_factor=0.1):
        self.policy = policy
        with tf.variable_scope(scope):
            with tf.name_scope("Trainer"):
                self.actions = tf.placeholder(shape=[None], dtype=tf.int64, name="actions")
                self.rewards = tf.placeholder(shape=[None], dtype=tf.float32, name="rewards")
                self.values = tf.placeholder(shape=[None], dtype=tf.float32, name="values")
                q = select_from(policy.q, self.actions)
                p = select_from(policy.probs, self.actions)
                discounted_rewards = discount(self.rewards, discount_rate, self.values[-1])

                # Compute losses
                self.accuracy_loss = tf.multiply(
                    (1 - discount_rate),
                    tf.reduce_sum((1 - p) * tf.square(discounted_rewards - q)),
                    name="AccuracyLoss"
                )
                self.consistent_loss = tf.reduce_sum(
                    tf.square(q - self.values[1:] - self.rewards),
                    name="ConsistencyLoss"
                )
                self.advantage = tf.reduce_sum(
                    (1 - p) * tf.square(tf.reduce_mean(policy.q) - q),
                    name="AdvantageLoss"
                )
                self.loss = tf.add_n([self.accuracy_loss,
                                      consistency_factor * self.consistent_loss,
                                      advantage_factor * self.advantage], name="Loss")

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.var_norms = tf.global_norm(local_vars)
                grads = self.get_gradients(local_vars)

                # Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, parent)
                self.apply_grads = optimizer.apply_gradients(zip(grads, global_vars))

    @staticmethod
    def trainer_kwargs(discount_rate=0.99, consistency_factor=0.1, advantage_factor=0.1):
        return locals()

    def get_gradients(self, local_vars):
        self.gradients = tf.gradients(self.loss, local_vars, name="Gradients")
        self.grad_norms = tf.global_norm(self.gradients, name="GradientNorms")
        return self.gradients

    def train(self, sess, actions, obs, rewards, values):
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
                self.policy.base_exploration_rate: self.policy.explore_rate,
                self.actions: actions,
                self.rewards: rewards,
                self.values: values}  # inputs
        )[0]


class BasePolicy:
    def __init__(self, scope, input_size, num_actions):
        self.explore_rate = None
        self.reset()
        with tf.variable_scope(scope):
            with tf.name_scope("Policy"):
                # Define the exploration rate reduction policy
                self.base_exploration_rate = tf.placeholder(tf.float32, shape=[], name="InputExploreRate")
                self.exploration_rate = self.setup_exploration_rate_policy()

                # Define the neural net operations
                self.input = tf.placeholder(shape=[None, input_size], dtype=tf.float32, name="Input")
                self.q = self.setup_q_network(num_actions)
                self.probs, self.action = self.setup_selection_policy()
                self.action = tf.reshape(tf.multinomial(tf.log(self.probs), 1), [-1])
                self.value = tf.reduce_sum(self.probs * self.q, axis=1) / tf.reduce_sum(self.probs, axis=1)

    def reset(self, explore_rate=0.1):
        self.explore_rate = explore_rate or self.explore_rate

    def setup_q_network(self, num_actions):
        raise NotImplementedError("Need to define a function that estimates Q(s,a).\n"
                                  "Signature: setup_q_network(self, num_actions) -> "
                                  "q: tensor(<scope>/Q:0, [?,num_actions]")

    def setup_exploration_rate_policy(self):
        return tf.constant(self.explore_rate, tf.float32, shape=[], name="ExplorationRate")

    def setup_selection_policy(self):
        best = tf.argmax(self.q, axis=-1)
        p_best = 1 - self.exploration_rate
        p_other = self.exploration_rate / tf.cast(tf.shape(self.q)[-1], dtype=tf.float32)
        probs = tf.add(tf.one_hot(best, tf.shape(self.q)[-1], dtype=tf.float32) * p_best, p_other, name="ActionProbs")
        action = tf.reshape(tf.multinomial(tf.log(probs), 1), [-1], name="SelectedAction")
        return probs, action

    def step(self, sess, obs):
        return sess.run(fetches=[self.action, self.value],  # returns
                        feed_dict={self.input: obs,
                                   self.base_exploration_rate: self.explore_rate})  # input

    def get_value(self, sess, obs):
        return sess.run(fetches=self.value,  # returns
                        feed_dict={self.input: obs,
                                   self.base_exploration_rate: self.explore_rate})  # input

    @staticmethod
    def policy_kwargs(input_size, num_actions):
        return locals()