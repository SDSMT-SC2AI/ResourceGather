from common.helper_functions import update_target_graph


class BaseAgent:
    def __init__(self, name, parent, optimizer, network, episode, policy_spec, trainer_spec):
        self.policy = network.Policy(name, episode, policy_spec)
        self.trainer = network.Trainer(name, optimizer, self.policy, trainer_spec)
        self.update_local_policy = update_target_graph(parent, name)

    def train(self, sess, actions, rewards, observations, values):
        return self.trainer.train(sess, observations, actions, rewards, values)

    def step(self, sess, observation):
        choice, value = self.policy.step(sess, observation)
        return choice[0], value[0]

    def value(self, sess, obs):
        return self.policy.get_value(sess, obs)[0]

    def update_policy(self, sess):
        sess.run(self.update_local_policy)

    @staticmethod
    def process_observation(obs):
        raise NotImplementedError
