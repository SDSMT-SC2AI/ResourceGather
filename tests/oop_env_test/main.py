import sys
from context import set_path
set_path()
from tests.oop_env_test.agent import Simple
from tests.oop_env_test.actions import ActionSpace
from tests.oop_env_test.env import SimpleEnv
from tests.oop_env_test.worker import Worker
from a3c.base_main import BaseMain


def should_stop(episode_step_count, **_):
    print(episode_step_count, _)
    return episode_step_count > 100


class Main(BaseMain):
    def __init__(self, load_model=False, max_episodes=2000,
                 buffer_min=10, buffer_max=30):
        super().__init__(load_model, max_episodes, buffer_min, buffer_max)
        self.policy_kwargs = Simple.policy_kwargs(num_actions=len(ActionSpace.choices),
                                                  episode=None,
                                                  hidden_layer_size=1,
                                                  max_explore_rate=0.5,
                                                  min_explore_rate=0.005,
                                                  max_episodes=int(0.8*self.max_episodes))
        self.trainer_kwargs = Simple.trainer_kwargs()

    def setup_global_network(self, name):
        self.policy_kwargs.update(episode=self.global_episodes)
        return Simple.setup_policy(name, **self.policy_kwargs)

    def setup_agent(self, name, parent, optimizer, global_episodes):
        return Simple(name, parent, optimizer, self.policy_kwargs, self.trainer_kwargs)

    def setup_env(self):
        return SimpleEnv(mode="Basic")

    def setup_worker(self, name, number, main_inst, env, agent, model_path, global_episodes,
                     buffer_min, buffer_max, max_episodes):
        worker = Worker(number, main_inst, env, agent, global_episodes,
                        model_path=model_path,
                        summary_dir=self.run_dir,
                        buffer_min=buffer_min,
                        buffer_max=buffer_max,
                        max_episodes=max_episodes,
                        should_stop=lambda episode_step_count, **_: episode_step_count > 100)
        worker.initialize(ActionSpace())
        return worker


if __name__ == "__main__":
    Main().main()
