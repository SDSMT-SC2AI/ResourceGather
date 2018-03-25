import tensorflow as tf
import numpy as np
mse = tf.losses.mean_squared_error


class Worker:
    def __init__(self, name, main, env, actions, agent_cls, optimizer, model_path, global_episodes,
                 buffer_min=10, buffer_max=30, flags=None):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.optimizer = optimizer
        self.buffer_min = buffer_min
        self.buffer_max = buffer_max
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_" + str(self.number))
        self.flags = flags
        self.main = main

        # Create the local copy of the agent which inherits the global network parameters
        self.agent = agent_cls(self.name, 'global', optimizer, self.global_episodes, actions.action_spec)
        self.actions = actions

        print('Initializing environment #{}...'.format(self.number))
        self.env = env

    def train(self, rollout, sess, bootstrap_value):
        actions = np.array(rollout[0])
        rewards = np.array(rollout[1])
        observations = np.concatenate(rollout[2])
        values = np.array(rollout[3] + [bootstrap_value])

        #print(rewards)
        #print(values)

        loss, accuracy, consistency, advantage, grad_norms, var_norms = \
            self.agent.train(sess, actions, rewards, observations, values)
        self.agent.update_policy(sess)
        
        return loss, \
            accuracy, \
            consistency, \
            advantage, \
            grad_norms, \
            var_norms

    def do_actions(self, choice):
        env_obs = None
        feed_back, queue = self.actions.act(choice)
        for action in queue:
            env_obs = self.env.step(actions=[action])
        return feed_back, env_obs

    def work(self, max_episode_length, sess, coord, saver):
        loss = accuracy = consistency = advantage = gradient_norms = var_norms = 0
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                self.agent.update_policy(sess)

                episode_buffer = [[] for _ in range(4)]
                episode_values = []
                episode_reward = 0
                episode_step_count = 0

                # Start new episode
                env_obs = self.env.reset()
                reward, obs, episode_end = self.agent.process_observation(env_obs, self.flags)

                while not episode_end:
                    choice, value = self.agent.step(sess, obs)
                    feedback, env_obs = self.do_actions(choice)
                    reward, obs, episode_end = self.agent.process_observation(env_obs, self.flags)

                    for i, v in enumerate([choice, reward + feedback, obs, value]):
                        episode_buffer[i].append(v)

                    episode_values.append(value)
                    episode_reward += reward + feedback
                    total_steps += 1
                    episode_step_count += 1

                    if episode_end:
                        break

                    if len(episode_buffer[0]) == self.buffer_max and episode_step_count != max_episode_length - 1:
                        bootstrap = self.agent.value(sess, obs)
                        loss, accuracy, consistency, advantage, gradient_norms, var_norms = \
                            self.train(episode_buffer, sess, bootstrap)
                        episode_buffer = [feed[-self.buffer_min:] for feed in episode_buffer]

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))
                episode_count += 1

                if self.main._max_score < episode_reward:
                    self.main._max_score = episode_reward
                self.main._running_avg_score = (2.0 / 101) * (episode_reward - self.main._running_avg_score) + self.main._running_avg_score
                self.main._episodes[self.number] = episode_count
                self.main._steps[self.number] = total_steps

                #print(
                 #   "{} Step #{} Episode #{} Reward: {}".format(self.name, total_steps, episode_count, episode_reward))
                #print("Total Steps: {}\tTotal Episodes: {}\tMax Score: {}\tAvg Score: {}".format(
                  #  np.sum(self.main._steps), np.sum(self.main._episodes), self.main._max_score, self.main._running_avg_score))

                print("{:6.0f} Episodes: "
                      "loss = {:10.4f}, "
                      "accuracy = {:10.4f}, "
                      "consistency = {:10.4f}, "
                      "advantage = {:10.4f}, "
                      "reward = {:4d}".format(
                        np.sum(self.main._episodes), loss, accuracy, consistency, advantage, episode_reward))

                # Update the network using the episode buffer at the end of the episode
                if len(episode_buffer) > self.buffer_min:
                    bootstrap = self.agent.value(sess, obs)
                    loss, accuracy, consistency, advantage, gradient_norms, var_norms = \
                        self.train(episode_buffer, sess, bootstrap)

                if episode_count % 50 == 0 and episode_count != 0:
                    if episode_count % 2500 == 0 and self.name == 'worker_0':
                        saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
                        print("Saved Model")

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Accuracy', simple_value=float(accuracy))
                    summary.value.add(tag='Losses/Consistency', simple_value=float(consistency))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(gradient_norms))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(var_norms))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    sess.run(self.increment)

