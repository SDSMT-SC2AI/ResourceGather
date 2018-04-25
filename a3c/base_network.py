import tensorflow as tf
import numpy as np
from common.helper_functions import bisection, discount, select_from


class BaseTrainer:
    def __init__(self, scope, parent, optimizer, policy, *,
                 # key word arguments
                 discount_rate=0.99, consistency_factor=0.1, advantage_factor=0.1):
        self.policy = policy
        self.fetches = None
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
                    tf.reduce_sum((1-p)*tf.square(discounted_rewards - q)),
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
                self.var_norms = tf.global_norm(local_vars, name="VariableNorms")
                grads = self.get_gradients(local_vars)

                # Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, parent)
                self.apply_grads = optimizer.apply_gradients(zip(grads, global_vars), name="GradientUpdate")

    @staticmethod
    def trainer_kwargs(discount_rate=0.99, consistency_factor=0.1, advantage_factor=0.1):
        return locals()

    def get_gradients(self, local_vars):
        self.gradients = tf.gradients(self.loss, local_vars, name="Gradients")
        self.grad_norms = tf.global_norm(self.gradients, name="GradientNorms")
        return self.gradients

    def train(self, sess, actions, obs, rewards, values, feed_dict=None):
        return sess.run(
            fetches=[{
                "loss": self.loss,
                "accuracy": self.accuracy_loss,
                "consistency": self.consistent_loss,
                "advantage": self.advantage,
                "grad_norms": self.grad_norms,
                "var_norms": self.var_norms}  # returns
                .update(self.fetches or {}),
                [self.apply_grads]],  # ops
            feed_dict={
                self.policy.input: obs,
                self.actions: actions,
                self.rewards: rewards,
                self.values: values}  # inputs
            .update(feed_dict or {})
        )[0]


class BasePolicy:
    def __init__(self, scope, input_size, num_actions):
        self.reset()
        self.q = None
        self.action = None
        self.value = None
        self.policy = None

        with tf.variable_scope(scope):
            with tf.name_scope("Policy"):
                self.input = tf.placeholder(shape=[None, input_size], dtype=tf.float32, name="Input")
                self.setup_feats()
                self.setup_q(num_actions)
                self.setup_policy()
                self.setup_action()
                self.setup_value()
                # Define the neural net operations

        self.step_fetches = {
            "action": self.action[0],
            "value": self.value[0]
        }

        self.value_fetches = {
            "value": self.value[0],
        }

        self.feed_dict = lambda obs: {
            self.input: obs
        }

    def reset(self):
        pass

    def setup_feats(self):
        """
        If desired this function is intended to set the attribute self.feats which can be thought
        of as a new encoding of the state send in to the input tensor. Typically this is the
        neural network prior to the output layer.
        """
        pass

    def setup_q(self, num_actions):
        """
        This function performs the necessary tensorflow operations to set the attribute self.q
        which returns an array representing the function Q(a | s) given each state in the
        provided input tensor on sess.run.

        :param num_actions: The size of the output is the number of available actions.
        """
        static_q = tf.Variable(tf.random_normal([num_actions]), name="StaticPrediction")
        self.q = tf.tile(static_q, [tf.shape(self.input)[0], 1], name="QValue")

    def setup_action(self):
        """
        This function performs the necessary tensorflow operations to set the attribute self.action
        which will return the selected action given each state provided in the input tensor on
        sess.run.
        """
        self.action = tf.reshape(tf.multinomial(tf.log(self.policy), 1), [-1])

    def setup_value(self):
        """
        This function performs the necessary tensorflow operations to set the attribute self.value
        which will return the predicted value for each state in the provided input on sess.run.

        By Default, we're assuming the value is the Q(a | s) where a is the selected action.
        """
        self.value = select_from(self.q, self.action)

    def setup_policy(self):
        unnormalized = tf.exp(self.q - tf.reduce_max(self.q, axis=[-1],
                                                     keep_dims=True, name="MaxQ"),
                              name="MaxOneProbRatios")
        self.policy = tf.divide(unnormalized, tf.reduce_sum(unnormalized, keep_dims=True),
                                name="ActionProbs")

    def default_exploration_policy(self, exploration_rate_range, max_episodes, global_episode,
                                   seed=None, alpha=1.25, beta=6.0):
        miner, maxer = exploration_rate_range

        # Generate a random Beta-variate as your base exploration rate whenever self.update_ex_rate_op is executed
        beta_rv = tf.distributions.Beta(alpha, beta).sample(seed=seed, name="RandomBetaVariate")
        random_ex_rate = tf.Variable(1.0, trainable=False, name="BaseExplorationRate")
        self.update_ex_rate_op = random_ex_rate.assign(beta_rv)

        # Set exploration rate as a the beta variate with maximum value as a linear decaying thing....
        adjust = tf.minimum(1.0, tf.cast(global_episode, dtype=tf.float32) /
                            tf.cast(max_episodes, dtype=tf.float32))
        factor = (1 - adjust) * maxer + adjust * miner
        self.exploration_rate = tf.multiply(factor, random_ex_rate, name="RandomExplorationRate")
        best = tf.argmax(self.q, axis=-1)
        p_best = 1 - self.exploration_rate
        p_other = self.exploration_rate / tf.cast(tf.shape(self.q)[-1], dtype=tf.float32)
        probs = tf.add(tf.one_hot(best, tf.shape(self.q)[-1], dtype=tf.float32) * p_best, p_other, name="ActionProbs")
        action = tf.reshape(tf.multinomial(tf.log(probs), 1), [-1], name="SelectedAction")
        return probs, action

    def step(self, sess, obs):
        return sess.run(fetches=self.step_fetches,  feed_dict=self.feed_dict(obs))  # input

    def get_value(self, sess, obs):
        return sess.run(fetches=self.value_fetches,  feed_dict=self.feed_dict(obs))

    @staticmethod
    def policy_kwargs(input_size, num_actions):
        return locals()