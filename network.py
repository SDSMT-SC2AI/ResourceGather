import tensorflow as tf
from common.helper_functions import bisection, discount, select_from


class Trainer:
    @staticmethod
    def trainer_spec(accuracy_coefficient=1.0,
                     consistency_coefficient=0.5,
                     max_grad_norm=40.0,
                     discount_factor=0.99):
        return {
            'accuracy coefficient': accuracy_coefficient,
            'consistency coefficient': consistency_coefficient,
            'max gradient norm': max_grad_norm,
            'discount factor': discount_factor
        }

    def __init__(self, scope, optimizer, policy, trainer_spec):
        self.policy = policy
        with tf.variable_scope(scope):
            a_c, c_c, max_grad_norm, gamma = \
                [trainer_spec[k] for k in ['accuracy coefficient', 'consistency coefficient',
                                           'max gradient norm', 'discount factor']]

            self.actions = tf.placeholder(shape=[None], dtype=tf.int64, name="actions")
            self.rewards = tf.placeholder(shape=[None], dtype=tf.float32)
            self.values = tf.placeholder(shape=[None], dtype=tf.float32)
            q = select_from(policy.q, self.actions)
            discounted_rewards = discount(self.rewards, gamma, self.values[-1])

            self.accuracy_loss = tf.reduce_mean(tf.square(q - discounted_rewards))
            self.consistent_loss = tf.reduce_mean(tf.square(q - self.values[1:]))
            self.loss = a_c * self.accuracy_loss + c_c * self.consistent_loss

            # Get gradients from local network using local losses
            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            # self.gradients - gradients of loss wrt local_vars
            self.gradients = tf.gradients(self.loss, local_vars)
            self.var_norms = tf.global_norm(local_vars)
            grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, max_grad_norm)

            # Apply local gradients to global network
            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
            self.apply_grads = optimizer.apply_gradients(zip(grads, global_vars))

    def train(self, sess, obs, actions, rewards, values):
        return sess.run(
            fetches=[[
                self.loss,
                self.accuracy_loss,
                self.consistent_loss,
                self.grad_norms,
                self.var_norms],  # returns
                [self.apply_grads]],  # ops
            feed_dict={
                self.policy.input: obs,
                self.actions: actions,
                self.rewards: rewards,
                self.values: values}  # inputs
        )[0]


class Policy:
    @staticmethod
    def policy_spec(input_size=None,
                    num_actions=None,
                    max_episodes=None,
                    hidden_layer_size=None,
                    error_discount=0.95,
                    base_explore_rate=0.1,
                    min_explore_rate=None):
        return {
            "input size": input_size,
            "hidden layer size": hidden_layer_size or (3*input_size)//2,
            "error discount": error_discount,
            "number of actions": num_actions,
            "base exploration rate": base_explore_rate,
            "minimum exploration rate": min_explore_rate or base_explore_rate / 4,
            "max episodes": max_episodes
        }

    def __init__(self, scope, episode, policy_spec):
        self.network_spec = policy_spec
        with tf.variable_scope(scope):
            # Define some variables for quantifying the error of our q estimator
            self.error_discount = tf.constant(policy_spec['error discount'], dtype=tf.float32)
            self.error_discount_sum = tf.constant(1.0 / (1 - policy_spec['error discount']), dtype=tf.float32)
            self.error_factor = tf.Variable(
                initial_value=1.0,
                trainable=False,
                dtype=tf.float32,
                name="error_factor"
            )
            self.q_error = tf.Variable(
                initial_value=tf.ones([policy_spec["number of actions"]]),
                trainable=False,
                dtype=tf.float32,
                name="q_error"
            )
            self.previous_q = tf.Variable(
                expected_shape=(),
                trainable=False,
                dtype=tf.float32,
                name="previous_q"
            )
            self.previous_a = tf.Variable(
                expected_shape=(),
                trainable=False,
                dtype=tf.int64,
                name="previous_a"
            )

            # Define the exploration rate reduction policy
            self.base_exploration_rate = tf.constant(policy_spec['base exploration rate'])
            self.min_exploration_rate = tf.constant(policy_spec['minimum exploration rate'])
            self.max_episodes = tf.constant(policy_spec['max episodes'])
            self.exploration = self.min_exploration_rate + \
                ((self.base_exploration_rate - self.min_exploration_rate)*episode)/self.max_episodes

            # Define the neural net operations
            self.input = tf.placeholder(shape=[None, policy_spec["input size"]], dtype=tf.float32, name="input")

            hidden1 = tf.contrib.layers.fully_connected(
                inputs=self.input,
                num_outputs=policy_spec["hidden layer size"],
                activation_fn=tf.nn.elu,
                biases_initializer=tf.random_uniform_initializer(-1, 1)
            )
            # hidden1 = tf.add(hidden1, tf.random_normal(shape=tf.shape(hidden1), stddev=0.1))

            self.q = tf.contrib.layers.fully_connected(
                inputs=hidden1,
                num_outputs=policy_spec["number of actions"],
                weights_initializer=tf.orthogonal_initializer()
            )

            # Calculate the likelihoods of the median of the maximum of q with normal random noise
            # scale proportional to the estimated rmse error for that particular action
            loc = self.q
            scale = tf.multiply(self.error_factor, tf.sqrt(self.q_error))
            scale = tf.tile(tf.reshape(scale, [1, -1]), [tf.shape(loc)[0], 1])

            def get_probs(loc, scale):
                dist = tf.distributions.Normal(loc, scale)
                return 2 * (1 - dist.cdf(bisection(lambda x: 0.5 - tf.reduce_sum(1-dist.cdf(x)),
                                                   x=tf.reduce_max(loc))))

            probs = tf.map_fn(get_probs, (loc, scale), back_prop=False)
            self.action = tf.squeeze(tf.multinomial(tf.log(probs), 1))
            self.value = tf.reduce_sum(probs * self.q, axis=1) / tf.reduce_sum(probs, axis=1)
            self.probs = probs

            def error_update():
                idx = tf.reshape(self.previous_a, [-1, 1])
                op1 = tf.scatter_update(self.q_error, idx,
                                        self.error_discount * tf.gather_nd(self.q_error, idx)
                                        + tf.square(tf.subtract(self.previous_q, self.value))
                                        / self.error_discount_sum)
                op2 = tf.assign(self.previous_a, self.action)
                idx = tf.reshape(self.action, [-1, 1])
                op3 = tf.assign(self.previous_q, tf.squeeze(tf.gather_nd(self.q, idx)))
                return op1, op2, op3

            self.error_update_op = error_update()

    def step(self, sess, obs):
        return sess.run(fetches=[(self.action, self.value),  # returns
                                 (self.error_update_op,)],  # ops
                        feed_dict={self.input: obs})[0]  # input

    def value(self, sess, obs):
        return sess.run(fetches=self.value,  # returns
                        feed_dict={self.input: obs})  # input
