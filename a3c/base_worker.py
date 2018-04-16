import tensorflow as tf
import numpy as np
from common.parse_args import ensure_dir

mse = tf.losses.mean_squared_error
per_point = 10


class BaseWorker:
    def __init__(self, number, main, env, agent, global_episodes, *,
                 # keyword args
                 name=None, model_path=None, summary_dir="workerData/",
                 episodes_per_record=10, episodes_for_model_checkpoint=250,
                 buffer_min=10, buffer_max=30, max_episodes=10000, logging_callback=None,
                 tensorboard_callback=None, should_stop=lambda **_: None):
        self.number = number
        self.name = (name or "worker_") + str(number)

        # Create a directory for model checkpoints and tensorboard summaries
        self.model_path = model_path or summary_dir + "model"
        ensure_dir(summary_dir)
        ensure_dir(self.model_path)

        # When to train on the episode buffer and when to how much to flush
        self.buffer_min = buffer_min
        self.buffer_max = buffer_max

        # A tf var for the global number of episodes (incremented only on thread 0)
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)

        # Setup episode schedule and variables
        self.episodes_per_record = episodes_per_record
        self.episodes_per_model_checkpoint = episodes_for_model_checkpoint
        self.episode_rewards = np.zeros(episodes_per_record)
        self.episode_lengths = np.zeros(episodes_per_record)
        self.episode_mean_values = np.zeros(episodes_per_record)
        self.max_episodes = max_episodes

        # Setup summary writer for worker
        self.summary_writer = tf.summary.FileWriter(summary_dir + self.name)

        # Standard out logging callback
        self.logging_callback = logging_callback or self.default_logging_callback

        # Tensorboard callback
        self.tensorboard_callback = tensorboard_callback or self.default_tensorboard_callback

        # Should Stop callback
        self.should_stop = should_stop

        # Set variables for important components
        self.main = main
        self.agent = agent
        self.env = env

    # Initialization routine
    def initialize(self, *_):
        pass

    # Prepare worker for a new episode returns env_obs
    def reset(self):
        self.agent.policy.reset()
        return self.env.reset()

    # Using the data rollout of actions, rewards, observations, and values, call the Trainer and update
    # the global network
    def train(self, rollout, sess, bootstrap_value):
        actions = np.array(rollout[0])
        rewards = np.array(rollout[1])
        observations = np.concatenate(rollout[2])
        values = np.array(rollout[3] + [bootstrap_value])

        loss, accuracy, consistency, advantage, grad_norms, var_norms = \
            self.agent.train(sess, actions, observations, rewards, values)
        self.agent.update_policy(sess)
        
        return loss, \
            accuracy, \
            consistency, \
            advantage, \
            grad_norms, \
            var_norms

    # This function is used to execute actions in the environment
    def do_actions(self, choice, caller_vars):
        raise NotImplementedError("Need to define a function that takes an integer choice, and "
                                  "returns an observation from the environment.\n"
                                  "Signature: do_actions(self, choice) -> env_obs")

    # This is the main thread that runs the agent for a set number of episodes
    def work(self, sess, coord, saver):
        self.episode_count = sess.run(self.global_episodes)
        self.summary_writer.add_graph(sess.graph)
        total_steps = 0
        print("Starting " + self.name)
        with sess.as_default(), sess.graph.as_default():

            # Run until interrupted or episode is less than the maximum number of episodes for a worker
            while not coord.should_stop() and self.episode_count < self.max_episodes:
                # Pull in global trainable variables
                self.agent.update_policy(sess)

                # set up some variables to watch and the episode buffer
                episode_buffer = [[] for _ in range(4)]
                episode_values = 0
                episode_reward = 0
                episode_step_count = 0
                buffer_dumps = 0
                loss = accuracy = consistency = advantage = gradient_norms = var_norms = 0

                # Start new episode
                env_obs = self.reset()
                reward, obs, episode_end = self.agent.process_observation(env_obs)

                # loop until episode is complete
                while not episode_end and not self.should_stop(**locals()):
                    choice, value = self.agent.step(sess, obs) 
                    env_obs = self.do_actions(choice, locals())
                    reward, obs, episode_end = self.agent.process_observation(env_obs)

                    for i, v in enumerate([choice, reward, obs, value]):
                        episode_buffer[i].append(v)

                    episode_values += value
                    episode_reward += reward
                    total_steps += 1
                    episode_step_count += 1

                    if episode_end:
                        break

                    # if the buffer is full, train and flush
                    if len(episode_buffer[0]) == self.buffer_max:
                        buffer_dumps += 1
                        bootstrap = self.agent.value(sess, obs)
                        v = self.train(episode_buffer, sess, bootstrap)
                        loss += v[0]
                        accuracy += v[1]
                        consistency += v[2]
                        advantage += v[3]
                        gradient_norms += v[4]
                        var_norms += v[5]
                        episode_buffer = [feed[-self.buffer_min:] for feed in episode_buffer]

                # wrap up episode
                self.on_episode_end(locals())
                self.logging_callback(**locals())

                # write tensorboard summaries after a number of episodes
                if self.episode_count % self.episodes_per_record == 0:
                    summary = self.tensorboard_callback(**locals())
                    self.summary_writer.add_summary(summary, self.episode_count)
                    self.summary_writer.flush()

                if self.number == 0:
                    # If we're thread zero save model to model path after a number of episodes
                    if self.episode_count % self.episodes_per_model_checkpoint == 0:
                        saver.save(sess, self.model_path + '/model-' + str(self.episode_count) + '.cptk')
                        print("Saved Model")
                    sess.run(self.increment)

    # called immediately after the completion of an episode, parent method MUST be called for inheriting classes
    def on_episode_end(self, caller_vars):
        self.episode_rewards[self.episode_count % self.episodes_per_record] = caller_vars["episode_reward"]
        self.episode_lengths[self.episode_count % self.episodes_per_record] = caller_vars["episode_step_count"]
        self.episode_mean_values[self.episode_count % self.episodes_per_record] = \
            caller_vars["episode_values"] / caller_vars["episode_step_count"]
        self.episode_count += 1

        if self.main.max_score < caller_vars["episode_reward"]:
            self.main.max_score = caller_vars["episode_reward"]
        self.main.running_avg_score = (2.0 / 101) * (
                    caller_vars["episode_reward"] - self.main.running_avg_score) + self.main.running_avg_score
        self.main.episodes[self.number] = self.episode_count
        self.main.steps[self.number] = caller_vars["total_steps"]

        # Update the network using the episode buffer at the end of the episode
        if len(caller_vars["episode_buffer"]) > self.buffer_min:
            caller_vars["buffer_dumps"] += 1
            bootstrap = self.agent.value(caller_vars["sess"], caller_vars["obs"])
            v = self.train(caller_vars["episode_buffer"], caller_vars["sess"], bootstrap)
            caller_vars["loss"] += v[0]
            caller_vars["accuracy"] += v[1]
            caller_vars["consistency"] += v[2]
            caller_vars["advantage"] += v[3]
            caller_vars["gradient_norms"] += v[4]
            caller_vars["var_norms"] += v[5]

        caller_vars["loss"] /= caller_vars["buffer_dumps"]
        caller_vars["accuracy"] /= caller_vars["buffer_dumps"]
        caller_vars["consistency"] /= caller_vars["buffer_dumps"]
        caller_vars["advantage"] /= caller_vars["buffer_dumps"]
        caller_vars["gradient_norms"] /= caller_vars["buffer_dumps"]
        caller_vars["var_norms"] /= caller_vars["buffer_dumps"]

    @staticmethod
    def default_logging_callback(self, loss, accuracy, consistency, advantage, episode_reward, **_):
        print("{:6.0f} Episodes: "
              "loss = {:13.4f}, "
              "accuracy = {:13.4g}, "
              "consistency = {:13.4g}, "
              "advantage = {:13.4g}, "
              "reward = {:8.1f}, ".format(
                np.sum(self.main.episodes), loss, accuracy,
                consistency, advantage, episode_reward))

    @staticmethod
    def default_tensorboard_callback(self, accuracy, consistency, advantage,
                                     gradient_norms, var_norms, **_):
        mean_reward = np.mean(self.episode_rewards)
        var_reward = np.var(self.episode_rewards)
        mean_value = np.mean(self.episode_mean_values)
        summary = tf.Summary()
        summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
        summary.value.add(tag="Perf/VarReward", simple_value=float(var_reward))
        summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
        summary.value.add(tag='Losses/Accuracy', simple_value=float(accuracy))
        summary.value.add(tag='Losses/Consistency', simple_value=float(consistency))
        summary.value.add(tag='Losses/Advantage', simple_value=float(advantage))
        summary.value.add(tag='Losses/Grad Norm', simple_value=float(gradient_norms))
        summary.value.add(tag='Losses/Var Norm', simple_value=float(var_norms))
        return summary

    @staticmethod
    def worker_kwargs(name=None, model_path=None, summary_dir="workerData/",
                      episodes_per_record=10, episodes_for_model_checkpoint=250,
                      buffer_min=10, buffer_max=30, max_episodes=10000, logging_callback=None,
                      tensorboard_callback=None):
        return locals()


