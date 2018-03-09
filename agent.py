import tensorflow as tf
import actions
import observer
import network
hidden_layer_size = (3*observer.observation_size)//2


network_spec = {
            "input size": observer.observation_size,
            "hidden layer size": hidden_layer_size,
            "number of actions": len(actions.action_space)
        }

# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


class Smart:
    def __init__(self, name, parent, optimizer):
        self.network_spec = network_spec
        self.policy = network.Policy(name, self.network_spec)
        self.trainer = network.Trainer(name, optimizer, self.policy)
        self.update_local_policy = update_target_graph(parent, name)

    def step(self, sess, observation):
        choice, action_dist, value = self.policy.step(sess, observation)
        return actions.act(choice), choice, action_dist, value

    def train(self, sess, actions, action_dists, discounted_rewards, values, advantages):
        return sess.run([self.trainer.value_loss,
                         self.trainer.policy_loss,
                         self.trainer.entropy,
                         self.trainer.grad_norms,
                         self.trainer.var_norms,
                         self.trainer.apply_grads], feed_dict={
            self.trainer.actions: actions,
            self.trainer.action_dists: action_dists,
            self.trainer.target_v: discounted_rewards,
            self.trainer.values: values,
            self.trainer.advantages: advantages
        })

    def value(self, sess, obs):
        self.policy.value(sess, obs)

    def update_policy(self, sess):
        sess.run(self.update_local_policy)
