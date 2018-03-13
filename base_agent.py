from common.helper_functions import update_target_graph
import tensorflow as tf


class BaseAgent:
    def __init__(self, name, parent, optimizer, network, action_space, network_spec):
        self.network_spec = network_spec
        self.policy = network.Policy(name, self.network_spec)
        self.trainer = network.Trainer(name, optimizer, self.policy)
        self.update_local_policy = update_target_graph(parent, name)
        self.action_space = action_space

    def step(self, sess, observation):
        choice, action_dist, value = self.policy.step(sess, observation)
        return self.action_space.act(choice), choice, action_dist, value

    def train(self, sess, actions, action_dists, discounted_rewards, values, advantages):
        print(sess.run(tf.shape(self.policy.policy_fn), feed_dict={
          sel
        })
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

    @staticmethod
    def process_observation(obs):
        raise NotImplementedError
