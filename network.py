import tensorflow as tf
import typing
mse = tf.losses.mean_squared_error

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
    def __init__(self, sess,
                 entropy_coeff=0.01,
                 value_function_coeff=0.5,
                 max_gradient_norm=1,
                 optimizer_params=None,
                 args=None):
        self.actions = None
        self.advantage = None
        self.reward = None
        self.policy_gradient_loss = None
        self.value_function_loss = None
        self.optimize = None
        self.entropy = None
        self.loss = None
        self.learning_rate = None
        self.num_actions = None
        self.input_spec = None
        self.output_spec = None

        self.policy = Policy
        self.sess = sess
        self.value_function_coeff = value_function_coeff
        self.entropy_coeff = entropy_coeff
        self.max_gradient_norm = max_gradient_norm
        self.optimizer_params = optimizer_params

    def init_input(self):
        with tf.name_scope('input'):
            self.actions = tf.placeholder(tf.int32, [None])
            self.advantage = tf.placeholder(tf.float32, [None])
            self.reward = tf.placeholder(tf.float32, [None])
            self.learning_rate = tf.placeholder(tf.float32, [None])

    def init_network(self):
        self.policy = self.policy(self.sess, self.input_spec, self.output_spec)
        with tf.variable_scope('train_output'):
            negative_log_prob_action = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.policy.policy_fn,
                labels=self.actions)
            self.policy_gradient_loss = tf.reduce_mean(self.advantage * negative_log_prob_action)
            self.value_function_loss = mse(self.reward, tf.squeeze(self.policy.value_fn))
            self.entropy = tf.reduce_sum(tf.exp(self.policy.policy_fn)*self.policy.policy_fn, axis=1)
            self.loss = self.policy_gradient_loss \
                - self.entropy_coeff * self.entropy \
                + self.value_function_coeff * self.value_function_loss

            with tf.variable_scope("policy"):
                params = tf.trainable_variables()
            grads = tf.gradients(self.loss, params)

            # gradient clipping
            if self.max_gradient_norm is not None:
                grads, grad_norm = tf.clip_by_global_norm(grads, self.max_gradient_norm)

            grads = list(zip(grads, params))
            optimizer = tf.train.AdamOptimizer(**self.optimizer_params)
            self.optimize = optimizer.apply_gradients(grads)

    def build(self, input_spec, output_spec):
        self.output_spec = output_spec
        self.input_spec = input_spec
        self.init_input()
        self.init_network()


class BaseTrainer:
    def __init__(self, sess, model, args):
        self.model = model
        self.args = args
        self.sess = sess

        self.summary_placeholders = {}
        self.summary_ops = {}

        self.__init_global_saver()

    def save(self):
        print("Saving model...")
        self.saver.save(self.sess, self.args.checkpoint_dir)

    def __init_global_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.args.max_to_keep)
        self.summary_writer = tf.summary.FileWriter(self.args.summary_dir, self.sess.graph)

    def __init_model(self):
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

    def __load_model(self):
        latest_checkpoint = tf.train.latest_checkpoint(self.args.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
            print("Checkpoint loaded\n\n")
        else:
            print("No checkpoints available!\n\n")


class Trainer(BaseTrainer):
    def __init__(self, sess, model, args):
        pass


class A2C:
    def __init__(self, input_spec, action_spec: typing.List[int]):
        pass
