import tensorflow as tf
import numpy as np
from common.parse_args import ensure_dir
from common.helper_functions import BaseEvent, StepEvent

saver = tf.train.Saver(max_to_keep=5)


class SaveEvent(StepEvent):
    def __init__(self, args, interval):
        def save(d):
            path = d.model_path + "/model-" + str(d["episode_count"]) + ".cptk"
            saver.save(d.sess, path)
            print("Saved Model")
        super().__init__(args, interval, [save])

    def condition(self):
        return self["args"].number == 0 and super().condition()


class BaseWorker:
    def __init__(self, number, saver, env, agent, global_episodes, *,
                 # keyword args
                 name=None, model_path=None, summary_dir="workerData/",
                 episodes_per_model_checkpoint=250, buffer_min=10, buffer_max=30,
                 max_episodes=10000, custom_events=None):
        # Important links
        self.episode = self.Episode(agent, env, buffer_min, buffer_max)
        self.events = [SaveEvent(self, episodes_per_model_checkpoint, saver)] + (custom_events or [])

        # Thread specific fields
        self.number = number
        self.name = (name or "worker_") + str(number)

        # Create directories for summaries and model checkpoints
        self.model_path = model_path or summary_dir + "model"
        self.summary_dir = summary_dir
        ensure_dir(summary_dir)
        ensure_dir(self.model_path)

        # A tf var for the global number of episodes (incremented only on thread 0)
        self.global_episodes = global_episodes
        self.increment = global_episodes.assign_add(1)

        # Setup episode schedule and variables
        self.max_episodes = max_episodes

    # This is the main thread that runs the agent for a set number of episodes
    def work(self, sess, coord):
        print("Starting " + self.name)
        with sess.as_default(), sess.graph.as_default():
            self.episode.set_sess(sess)

            # Run until interrupted or episode is less than the maximum number of episodes for a worker
            while not coord.should_stop() and self.episode["count"] < self.max_episodes:
                self.episode.run()
                for e in self.events:
                    e.step()
                if self.number == 0:
                    sess.run(self.increment)

    @staticmethod
    def default_logging_event(args, interval):
        def print_log(args):
            print("{:6.0f} Episodes: "
                  "loss = {:13.4f}, "
                  "reward = {:8.1f}, ".format(
                    args["count"], args["loss"], args["reward_sum"]))
        return StepEvent(args, interval, calls=[print_log])

    @staticmethod
    def default_tensorboard_event(args, interval, summary_writer):
        class TensorBoardEvent(StepEvent):
            def __init__(self):
                self["episode_rewards"] = np.zeros(interval)
                self["episode_mean_values"] = np.zeros(interval)

                def tensorboard_log(args):
                    mean_reward = np.mean(self["episode_rewards"])
                    mean_value = np.mean(self["episode_mean_values"])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Perf/Loss', simple_value=float(args["loss"]))
                    summary_writer.add_summary(summary, args["count"])
                    summary_writer.flush()
                super().__init__(args, interval, calls=[tensorboard_log])

            def condition(self):
                index = self["T"] - 1
                self["episode_rewards"][index] = self["args"]["reward_sum"]
                self["episode_mean_values"][index] = self["args"]["value_sum"]
                return super().condition()

        return TensorBoardEvent()

    @staticmethod
    def worker_kwargs(name=None, model_path=None, summary_dir="workerData/",
                      episodes_per_record=10, episodes_for_model_checkpoint=250,
                      buffer_min=10, buffer_max=30, max_episodes=10000, logging_callback=None,
                      tensorboard_callback=None):
        return locals()

    class Episode(dict):
        def __init__(self, agent, env, buffer_min, buffer_max):
            super().__init__({
                # Episode buffer
                "actions": [], "rewards": [], "observations": [], "values": [],
                "keys::episode_buffer": ["actions", "rewards", "observations", "values"],

                # Per step scalars
                "action": -9999, "reward": -9999, "observation": -9999, "value": -9999,
                "keys::per_step": ["action", "reward", "observation", "value"],

                # Episode counters
                "step_count": 0,
                "buffer_dumps": 0,
                "keys::episode_counters": ["step_count", "buffer_dumps"],

                # Episode metrics
                "value_sum": -9999, "reward_sum": -9999, "loss": -9999,
                "keys::episode_metrics": ["value_sum", "reward_sum", "loss"],
                "keys::step_metrics": ["value_sum"],
                "keys::buffer_metrics": ["loss"],

                # Static info
                "sess": None,  # Must be set after init
                "buffer_min": buffer_min, "buffer_max": buffer_max,
                "agent": agent, "env": env,
                "count": 0,
                "keys::static_info": ["buffer_min", "buffer_max", "agent", "env", "sess", "count"]
            })

        def set_sess(self, sess):
            self["sess"] = sess

        def act(self):
            raise NotImplementedError("This function has access to all variables in the class's dictionary,\n"
                                      "including the currently selected 'action'. It must set 'reward', \n"
                                      "'observation', and return a flag indicating the end of the episode.")

        def think(self):
            """This function must set 'action' and 'value' fields."""
            self.update(self["agent"].step(self["sess"], self["observation"]))

        def step(self):
            return self._step()

        def _step(self):
            # Get action from agent and do that action
            self["step_count"] += 1
            self.think()
            if self.act():
                return False

            # Append data to the experience buffer
            self["actions"].append(self["action"])
            self["rewards"].append(self["reward"])
            self["observations"].append(self["observation"])
            self["values"].append(self["value"])

            if len(self["rewards"]) >= self["buffer_max"]:
                self["values"].append(self["agent"].step(self["sess"], self["observation"])[1])
                self.train()
                for k in self["keys::episode_buffer"]:
                    self[k] = self[-self["buffer_min"]:]
                self["buffer_dumps"] += 1
            return True

        def reset(self):
            self.update({k: [] for k in self["keys::episode_buffer"]})
            self.update({k: -9999 for k in self["keys::per_step"] + self["keys::episode_metrics"]})
            self.update({k: 0 for k in self["keys::episode_counters"]})
            env_obs = self["env"].reset()
            self["reward"], self["observation"], end_flag = self["agent"].process_observation(env_obs)

        def train(self):
            """
            At a minimum, this function must set accumulate the episode metric, 'loss' as well as
            perform a training step through the agent. The length of values must be 1 greater than the
            length of actions, observations, and rewards.
            """
            metrics = self["agent"].train(
                self["sess"],
                self["actions"],  # actions
                self["observations"],  # observations
                self["rewards"],  # rewards
                self["values"]  # value
            )
            self["loss"] += metrics["loss"]

        def run(self):
            self["count"] += 1
            self.reset()
            while self.step():
                pass
            return self.on_end()

        def on_end(self):
            self["reward_sum"] += self["reward"]
            self["value_sum"] += self["value"]
            self["values"].append(self["value"])
            if len(self["actions"]) >= self["buffer_min"]:
                self.train()
                self["buffer_dumps"] += 1
            for k in self["keys::step_metrics"]:
                self[k] /= self["step_count"] if self["step_count"] > 0 else 1
            for k in self["keys::buffer_metrics"]:
                self[k] /= self["buffer_dumps"] if self["buffer_dumps"] > 0 else 1

