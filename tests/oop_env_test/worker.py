from context import set_path
set_path()
from a3c.base_worker import BaseWorker


class Worker(BaseWorker):
    def initialize(self, actions):
        super().initialize()
        self.actions = actions

    def reset(self):
        env_obs = super().reset()
        self.actions.reset()
        return env_obs, 0

    def do_actions(self, choice, caller_vars):
        env_obs = caller_vars["env_obs"]
        feed_back = self.actions.act(choice, env_obs[0])

        while True:
            act_call, feed = self.actions.action_step(env_obs)
            env_obs = self.env.step(actions=[act_call])
            feed_back += feed
            if not self.actions.actionq:
                break

        return env_obs, feed_back
