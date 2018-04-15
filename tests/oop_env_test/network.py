import tensorflow as tf
import numpy as np
from context import set_path
set_path()
from a3c.base_network import BasePolicy, BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, scope, parent, optimizer, policy, *,
                 discount_rate=0.99, advantage_nerf=0.2, max_grad_norm=5.0):
        self.max_grad_norm = max_grad_norm
        super().__init__(scope, parent, optimizer, policy,
                         discount_rate=discount_rate,
                         advantage_nerf=advantage_nerf)

    def get_gradients(self, local_vars):
        self.gradients = tf.gradients(self.loss, local_vars, name="Gradients")
        grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, self.max_grad_norm)
        return grads

    @staticmethod
    def trainer_kwargs(discount_rate=0.99, advantage_nerf=0.2, max_grad_norm=5.0):
        return locals()


class Policy(BasePolicy):
    def __init__(self, scope, input_size, num_actions, episode, *,
                 q_range=(10,10.01), hidden_layer_size=None,
                 rate_policy_alpha=1.25, rate_policy_beta=5.65,
                 max_explore_rate=0.1, min_explore_rate=0.001, max_episodes=10000):
        self.alpha = rate_policy_alpha
        self.beta = rate_policy_beta
        self.episode = episode
        self.q_range = q_range
        self.hidden_layer_size = hidden_layer_size or 2 * input_size
        self.max_explore_rate = max_explore_rate
        self.min_explore_rate = min_explore_rate
        self.max_episodes = max_episodes
        super().__init__(scope, input_size, num_actions)

    def reset(self, explore_rate=None):
        self.explore_rate = explore_rate or np.random.beta(self.alpha, self.beta)

    def setup_q_network(self, num_actions):
        # print("Hidden layer size: ", policy_spec['hidden_layer_size'])
        hidden1 = tf.contrib.layers.fully_connected(
            inputs=self.input,
            num_outputs=self.hidden_layer_size,
            activation_fn=tf.nn.elu,
            biases_initializer=tf.random_uniform_initializer(-1, 1),
            scope="QNetHidden1"
        )
        # hidden1 = tf.add(hidden1, tf.random_normal(shape=tf.shape(hidden1), stddev=0.1))

        return tf.contrib.layers.fully_connected(
            inputs=hidden1,
            num_outputs=num_actions,
            weights_initializer=tf.orthogonal_initializer(0.1),
            biases_initializer=tf.random_uniform_initializer(*self.q_range),
            scope="QNetFinal"
        )

    def setup_exploration_rate_policy(self):
        # Define the exploration rate reduction policy
        adjust = tf.minimum(1.0, tf.cast(self.episode, dtype=tf.float32) /
                            tf.cast(self.max_episodes, dtype=tf.float32))
        factor = (1 - adjust) * self.max_explore_rate + adjust * self.min_explore_rate
        return tf.multiply(self.base_exploration_rate, factor, name="ExplorationRate")

    @staticmethod
    def policy_kwargs(input_size, num_actions, episode=None, q_range=(10, 10.01), hidden_layer_size=None,
                      rate_policy_alpha=1.25, rate_policy_beta=5.65,
                      max_explore_rate=0.1, min_explore_rate=0.001, max_episodes=10000):
        return locals()
