from common.helper_functions import update_target_graph


class BaseAgent:
    policy_cls = None
    trainer_cls = None

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

    @classmethod
    def setup_policy(cls, name, input_size, num_actions, **policy_kwargs):
        return cls.policy_cls(name, input_size, num_actions, **policy_kwargs)

    @classmethod
    def setup_trainer(cls, name, parent, optimizer, policy, **trainer_kwargs):
        return cls.trainer_cls(name, parent, optimizer, policy, **trainer_kwargs)

    def update_policy(self, sess):
        sess.run(self.update_local_policy)

    def process_observation(self, obs):
        raise NotImplementedError
