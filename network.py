import tensorflow as tf
from common.helper_functions import bisection, discount


class Trainer:
    def __init__(self, scope, optimizer, policy, value_c=0.5, policy_c=1, entropy_c=0.01, max_grad_norm=40.0):
        with tf.variable_scope(scope):
            self.actions = tf.placeholder(shape=[None], dtype=tf.int64, name="actions")
            self.rewards = tf.placeholder(shape=[None], dtype=tf.float32)
            self.observations = tf.placeholder(shape=[None, None], dtype=tf.float32)
            self.values = tf.placeholder(shape=[None], dtype=tf.float32)

            #self.target_v = tf.placeholder(shape=[None], dtype=tf.float32, name="target_v")
            #self.advantages = tf.placeholder(shape=[None], dtype=tf.float32, name="advantages")

            # Loss functions
            self.value_loss = tf.reduce_sum(tf.square(self.target_v - tf.squeeze(policy.value_fn)))
            negative_log_prob_actions = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=policy.policy_fn,
                labels=self.actions)
            self.policy_loss = tf.reduce_sum(self.advantages * tf.exp(-negative_log_prob_actions))
            self.entropy = tf.reduce_sum(tf.exp(policy.policy_fn) * policy.policy_fn)

            self.loss = value_c * self.value_loss - policy_c * self.policy_loss - entropy_c * self.entropy

            # Get gradients from local network using local losses
            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            # self.gradients - gradients of loss wrt local_vars
            self.gradients = tf.gradients(self.loss, local_vars)
            self.var_norms = tf.global_norm(local_vars)
            grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, max_grad_norm)

            # Apply local gradients to global network
            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
            self.apply_grads = optimizer.apply_gradients(zip(grads, global_vars))


class Policy:
    def __init__(self, scope, network_spec, max_episodes):
        self.network_spec = network_spec
        with tf.variable_scope(scope):
            # Define some variables for quantifying the error of our q estimator
            self.error_discount = tf.constant(network_spec['error discount'], dtype=tf.float32)
            self.error_discount_sum = tf.constant(1.0 / (1 - network_spec['error discount']), dtype=tf.float32)
            self.error_factor = tf.Variable(
                initial_value=1.0,
                trainable=False,
                dtype=tf.float32,
                name="error_factor"
            )
            self.q_error = tf.Variable(
                initial_value=tf.ones([network_spec["number of actions"]]),
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
            self.base_exploration_rate = tf.constant(network_spec['base exploration rate'])
            self.min_exploration_rate = tf.constant(network_spec['minimum exploration rate'])
            self.max_episodes = tf.constant(max_episodes)
            self.episode = tf.placeholder(shape=(), dtype=tf.int32)
            self.exploration = self.min_exploration_rate + \
                ((self.base_exploration_rate - self.min_exploration_rate)*self.episode)/self.max_episodes

            # Define the neural net operations
            self.input = tf.placeholder(shape=[None, network_spec["input size"]], dtype=tf.float32, name="input")

            hidden1 = tf.contrib.layers.fully_connected(
                inputs=self.input,
                num_outputs=network_spec["hidden layer size"],
                activation_fn=tf.nn.elu,
                biases_initializer=tf.random_uniform_initializer(-1, 1)
            )
            hidden1 = tf.add(hidden1, tf.random_normal(shape=tf.shape(hidden1), stddev=0.1))

            self.q = tf.contrib.layers.fully_connected(
                inputs=hidden1,
                num_outputs=network_spec["number of actions"],
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

    def step(self):
        return {
            'input': [self.input, self.episode],
            'output': [self.action, self.value, self.error_update_op]
        }

    def value(self):
        return {
            'input': [self.input, self.episode],
            'output': [self.value]
        }


