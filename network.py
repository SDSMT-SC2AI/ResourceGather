import tensorflow as tf
import numpy as np
from common.helper_functions import bisection, discount, select_from


class Trainer:
    @staticmethod
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

    def __init__(self, scope, optimizer, policy, trainer_spec, hyper_params):
        self.policy = policy
        self.hp = hyper_params
        with tf.variable_scope(scope):
            with tf.name_scope("Trainer"):
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


class Policy:
    @staticmethod
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

    def __init__(self, scope, episode, policy_spec, hyper_params):
        self.hyper_params = hyper_params
        self.network_spec = policy_spec
        self.random_explore_rate = np.random.beta(self.hyper_params.alpha, self.hyper_params.beta)

        with tf.variable_scope(scope):
            with tf.name_scope("Policy"):
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

    def reset(self):
        self.random_explore_rate = np.random.beta(self.hyper_params.alpha, self.hyper_params.beta)

    def step(self, sess, obs):
        return sess.run(fetches=[self.action, self.value],  # returns
                        feed_dict={self.input: obs,
                                   self.exploration_rate: self.random_explore_rate})  # input

    def get_value(self, sess, obs):
        return sess.run(fetches=self.value,  # returns
                        feed_dict={self.input: obs,
                                   self.exploration_rate: self.random_explore_rate})  # input
