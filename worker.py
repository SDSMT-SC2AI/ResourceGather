import tensorflow as tf
import numpy as np
import scipy.signal
import observer
import main
from pysc2.env import sc2_env
mse = tf.losses.mean_squared_error


# calculates y[i] = x[i] + gamma*y[i+1]
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class Worker:
    def __init__(self, name, agent_cls, optimizer, model_path, buffer_size, global_episodes, map_name, flags):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.optimizer = optimizer
        self.buffer_size = buffer_size
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_" + str(self.number))
        self.flags = flags

        # Create the local copy of the agent which inherits the global network parameters
        self.agent = agent_cls(self.name, 'global', optimizer)

        print('Initializing environment #{}...'.format(self.number))
        self.env = sc2_env.SC2Env(
            map_name=map_name,
            agent_race=flags.agent_race,
            bot_race=flags.bot_race,
            difficulty=flags.difficulty,
            step_mul=flags.step_mul,
            game_steps_per_episode=flags.game_steps_per_episode,
            screen_size_px=(flags.screen_resolution, flags.screen_resolution),
            minimap_size_px=(flags.minimap_resolution, flags.minimap_resolution),
            visualize=False)

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        actions = rollout[:, 0]
        action_dists = rollout[:, 1]
        rewards = rollout[:, 2]
        values = rollout[:, 3]

        discounted_rewards = discount(rewards, gamma)
        values_plus = np.concatenate([values, np.array([bootstrap_value])])
        advantages = discount(rewards + gamma * values_plus[1:] - values_plus[:-1], gamma)

        value_loss, policy_loss, entropy, grad_norms, var_norms, _ = \
            self.agent.train(sess,
                             actions,
                             action_dists,
                             discounted_rewards,
                             values,
                             advantages)

        return value_loss / len(rollout), policy_loss / len(rollout), entropy / len(rollout), \
            grad_norms, var_norms

    def work(self, max_episode_length, gamma, sess, coord, saver):
        value_loss = policy_loss = entropy = gradient_norms = var_norms = 0
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                self.agent.update_policy(sess)

                episode_buffer = []
                episode_values = []
                episode_reward = 0
                episode_step_count = 0

                # Start new episode
                env_obs = self.env.reset()[0]
                reward, obs, episode_end = observer.process_observation(env_obs, self.flags)

                while not episode_end:
                    actions, choice, action_dist, value = self.agent.step(sess, obs)
                    env_obs = self.env.step(actions=actions)
                    reward, obs, episode_end = observer.process_observation(env_obs, self.flags)

                    episode_buffer.append([choice, action_dist, reward, value[0, 0]])
                    episode_values.append(value[0, 0])

                    episode_reward += reward
                    total_steps += 1
                    episode_step_count += 1

                    if episode_end:
                        break

                    if len(episode_buffer) == self.buffer_size and episode_step_count != max_episode_length - 1:
                        bootstrap = self.agent.value(sess, obs)
                        value_loss, policy_loss, entropy, gradient_norms, var_norms = \
                            self.train(episode_buffer, sess, gamma, bootstrap)

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))
                episode_count += 1

                if main._max_score < episode_reward:
                    main._max_score = episode_reward
                main._running_avg_score = (2.0 / 101) * (episode_reward - main._running_avg_score) + main._running_avg_score
                main._episodes[self.number] = episode_count
                main._steps[self.number] = total_steps

                print(
                    "{} Step #{} Episode #{} Reward: {}".format(self.name, total_steps, episode_count, episode_reward))
                print("Total Steps: {}\tTotal Episodes: {}\tMax Score: {}\tAvg Score: {}".format(
                    np.sum(main._steps), np.sum(main._episodes), main._max_score, main._running_avg_score))

                # Update the network using the episode buffer at the end of the episode
                if len(episode_buffer) != 0:
                    value_loss, policy_loss, entropy, gradient_norms, var_norms = \
                        self.train(episode_buffer, sess, gamma, 0.0)

                if episode_count % 5 == 0 and episode_count != 0:
                    if episode_count % 250 == 0 and self.name == 'worker_0':
                        saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
                        print("Saved Model")

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(value_loss))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(policy_loss))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(entropy))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(gradient_norms))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(var_norms))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    sess.run(self.increment)

