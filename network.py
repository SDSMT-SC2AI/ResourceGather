import tensorflow as tf


# MG2033/A2C @ github.com
class BasePolicy:
    def __init__(self, sess, input_spec, output_spec, name="train"):
        self.name = name
        self.sess = sess
        self.input_spec = input_spec
        self.output_spec = output_spec
        self.input = None
        self.value_state = None
        self.action_policy = None

    def step(self, observation):
        raise NotImplementedError("step function not implemented")

    def value(self, observation):
        raise NotImplementedError("value function not implemented")


class Policy(BasePolicy):
    def __init__(self, sess, input_spec, output_spec, name="train", exploration_rate=1):
        super().__init__(sess, input_spec, output_spec, name)
        with tf.name_scope(name + "policy_input"):
            self.input = tf.placeholder(tf.uint8, input_spec["policy input shape"])
        with tf.variable_scope("policy", reuse=tf.AUTO_REUSE):
            hidden1 = tf.contrib.layers.fully_connected(
                inputs=self.input,
                num_outputs=output_spec["number of hidden nodes"],
                activation_fn=tf.nn.elu,
                biases_initializer=tf.initializers.random_uniform(-1, 1)
            )
            self.value_fn = tf.contrib.layers.fully_connected(
                inputs=hidden1,
                num_outputs=1
            )
            policy_raw = tf.contrib.layers.fully_connected(
                inputs=hidden1,
                num_outputs=output_spec["number of actions"],
                weights_initializer=tf.initializers.orthogonal()
            )
            self.policy_fn = policy_raw - tf.reduce_max(policy_raw, 1, keep_dims=True)

            with tf.name_scope("value"):
                self.value = self.value_fn[:, 0]

            with tf.name_scope("action"):
                self.exploration_rate = tf.Variable(exploration_rate + 1e-3, trainable=False)
                self.action = tf.multinomial(self.policy_fn / exploration_rate, 1)

    def step(self, observation, exploration_rate=1):
        # returns tuple(action, value, policy):
        #   a random action according to policy, the value function result, and current policy
        return self.sess.run([self.action, self.value, self.policy_fn],
                             feed_dict={self.input: observation,
                                        self.exploration_rate: exploration_rate + 1e-3})

    def value(self, observation, exploration_rate=1):
        # returns the value
        return self.sess.run(self.value,
                             feed_dict={self.input: observation,
                                        self.exploration_rate: exploration_rate + 1e-3})



class Model:
    def __init__(self, ):
        pass


class A2C:
    def __init__(self, input_spec, action_spec):
        pass
