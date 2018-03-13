import tensorflow as tf
mse = tf.losses.mean_squared_error


class Trainer:
    def __init__(self, scope, optimizer, policy, value_c=0.5, policy_c=1, entropy_c=0.01, max_grad_norm=40.0):
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
        self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

        # Loss functions
        self.value_loss = tf.reduce_sum(tf.square(self.target_v - tf.squeeze(policy.value_fn)))
        negative_log_prob_actions = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=policy.policy_fn,
            labels=self.actions)
        self.policy_loss = tf.reduce_sum(self.advantages * negative_log_prob_actions)
        self.entropy = tf.reduce_sum(tf.exp(policy.policy_fn) * policy.policy_fn, axis=1)

        self.loss = value_c * self.value_loss + policy_c * self.policy_loss - entropy_c * self.entropy

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
    def __init__(self, scope, network_spec):
        self.network_spec = network_spec
        with tf.variable_scope(scope):
            self.input = tf.placeholder(shape=[None, network_spec["input size"]], dtype=tf.float32)
            hidden1 = tf.contrib.layers.fully_connected(
                inputs=self.input,
                num_outputs=network_spec["hidden layer size"],
                activation_fn=tf.nn.elu,
                biases_initializer=tf.random_uniform_initializer(-1, 1)
            )
            self.value_fn = tf.contrib.layers.fully_connected(
                inputs=hidden1,
                num_outputs=1
            )
            policy_raw = tf.contrib.layers.fully_connected(
                inputs=hidden1,
                num_outputs=network_spec["number of actions"],
                weights_initializer=tf.orthogonal_initializer()
            )
            policy_shifted = policy_raw - tf.reduce_max(policy_raw, 1, keep_dims=True)
            self.policy_fn = policy_shifted - tf.log(tf.reduce_sum(tf.exp(policy_shifted)))
            self.exploration_rate = tf.placeholder(shape=None, dtype=tf.float32)
            self.action = tf.squeeze(tf.multinomial(self.policy_fn / (self.exploration_rate + 1e-3), 1))

    def step(self, sess, observation, exploration_rate=1):
                # returns tuple(action, value, policy):
                #   a random action according to policy, the value function result, and current policy
                return sess.run([self.action, self.policy_fn, self.value_fn], feed_dict={
                    self.input: observation,
                    self.exploration_rate: exploration_rate})

    def value(self, sess, observation, exploration_rate=1):
                # returns the value
                return sess.run(self.value_fn, feed_dict={
                    self.input: observation,
                    self.exploration_rate: exploration_rate})
