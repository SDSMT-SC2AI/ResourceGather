from common.helper_functions import update_target_graph


class BaseAgent:
    def __init__(self, name, parent, optimizer, network_kwargs, trainer_kwargs):
        self.policy = self.setup_policy(name, **network_kwargs)
        self.trainer = self.setup_trainer(name, parent, optimizer, self.policy, **trainer_kwargs)
        self.update_local_policy = update_target_graph(parent, name)

    def train(self, sess, actions,  observations, rewards, values):
        return self.trainer.train(sess, actions, observations, rewards, values)

    def step(self, sess, observation):
        choice, value = self.policy.step(sess, observation)
        return choice[0], value[0]

    def value(self, sess, obs):
        return self.policy.get_value(sess, obs)[0]

    def setup_policy(self, name, input_size, num_actions, **_):
        raise NotImplementedError("Need to define how the network is constructed.\n"
                                  "Signature: setup_policy(self, name, input_size, num_actions) -> policy: BasePolicy")

    def setup_trainer(self, name, parent, optimizer, policy, **_):
        raise NotImplementedError("Need to define how the network is constructed.\n"
                                  "Signature: setup_trainer(self, name, optimizer, policy) -> trainer: BaseTrainer")

    def update_policy(self, sess):
        sess.run(self.update_local_policy)

    def process_observation(self, obs):
        raise NotImplementedError
