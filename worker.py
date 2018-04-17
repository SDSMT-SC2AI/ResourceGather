import tensorflow as tf
import numpy as np
from common.parse_args import ensure_dir

mse = tf.losses.mean_squared_error


class Worker:
    def __init__(self, number, main, env, actions, agent, global_episodes, *,
                 # keyword args
                 name=None, model_path=None, summary_dir="workerData/",
                 episodes_per_record=10, episodes_for_model_checkpoint=250,
                 buffer_min=10, buffer_max=30, max_episodes=10000):
        self.number = number
        self.name = (name or "worker_") + str(number)

        self.model_path = model_path or summary_dir + "model"
        ensure_dir(summary_dir)
        ensure_dir(self.model_path)

        self.buffer_min = buffer_min
        self.buffer_max = buffer_max
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episodes_for_model_checkpoint = episodes_for_model_checkpoint
        self.episodes_per_record = episodes_per_record
        self.episode_rewards = np.zeros(episodes_per_record)
        self.episode_real_rewards = np.zeros(episodes_per_record)
        self.episode_lengths = np.zeros(episodes_per_record)
        self.episode_mean_values = np.zeros(episodes_per_record)
        self.max_episodes = max_episodes
        self.summary_writer = tf.summary.FileWriter(summary_dir + self.name)
        self.main = main
        self.agent = agent
        self.actions = actions
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

    def do_actions(self, choice, env_obs):
        feed_back = self.actions.act(choice, env_obs[0])

        while True:
            act_call, feed = self.actions.action_step(env_obs)
            env_obs = self.env.step(actions=[act_call])
            feed_back += feed
            if not self.actions.actionq:
                break

        return feed_back, env_obs

    def work(self, sess, coord, saver):
        checkpoint_steps = 0

        per_point = self.episodes_per_record
        episode_count = sess.run(self.global_episodes)
        if self.number == 0:
            self.summary_writer.add_graph(sess.graph)
        total_steps = 0
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            
            while not coord.should_stop() and episode_count < self.max_episodes:
                self.agent.update_policy(sess)

                episode_buffer = [[] for _ in range(4)]
                episode_values = 0
                episode_reward = 0
                episode_step_count = 0
                buffer_dumps = 0
                loss = accuracy = consistency = advantage = gradient_norms = var_norms = 0

                # Start new episode
                env_obs = self.env.reset() # There is only one agent running, so [0]
                self.actions.reset()
                self.agent.policy.reset()
                reward, obs, episode_end = self.agent.process_observation(env_obs)
                collected_last_ep = 0
                while not episode_end:
                    choice, value = self.agent.step(sess, obs) 
                    feedback, env_obs = self.do_actions(choice, env_obs)
                    reward, obs, episode_end = self.agent.process_observation(env_obs)

                    for i, v in enumerate([choice, reward + feedback, obs, value]):
                        episode_buffer[i].append(v)

                    episode_values += value
                    episode_reward += reward + feedback + (env_obs[0][1].resources_collected - collected_last_ep)
                    collected_last_ep = env_obs[0][1].resources_collected
                    episode_step_count += 1

                    if episode_end:
                        break

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

                self.episode_real_rewards[episode_count % per_point] = episode_reward
                self.episode_rewards[episode_count % per_point] = env_obs[0][1].resources_collected
                self.episode_lengths[episode_count % per_point] = episode_step_count
                self.episode_mean_values[episode_count % per_point] = episode_values / episode_step_count
                episode_count += 1

                self.main._episodes[self.number] = episode_count

                print("{:6.0f} Episodes: "
                      "loss = {:13.4f}, "
                      "accuracy = {:13.4g}, "
                      "consistency = {:13.4g}, "
                      "advantage = {:13.4f}, "
                      "reward = {:8.1f}, "
                      "minerals = {:8.1f}, ".format(
                        np.sum(self.main._episodes), loss, accuracy,
                        consistency, advantage, episode_reward,
                        env_obs[0][1].resources_collected))

                # Update the network using the episode buffer at the end of the episode
                if len(episode_buffer) > self.buffer_min:
                    buffer_dumps += 1
                    bootstrap = self.agent.value(sess, obs)
                    v = self.train(episode_buffer, sess, bootstrap)
                    loss += v[0]
                    accuracy += v[1]
                    consistency += v[2]
                    advantage += v[3]
                    gradient_norms += v[4]
                    var_norms += v[5]

                loss /= buffer_dumps
                accuracy /= buffer_dumps
                consistency /= buffer_dumps
                advantage /= buffer_dumps
                gradient_norms /= buffer_dumps
                var_norms /= buffer_dumps

                if episode_count % per_point == 0:
                    mean_reward = np.mean(self.episode_rewards)
                    mean_real_reward = np.mean(self.episode_real_rewards)
                    mean_value = np.mean(self.episode_mean_values)
                    summary = tf.Summary()
                    summary.value.add(tag="Perf/RealReward", simple_value=float(mean_real_reward))
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Accuracy', simple_value=float(accuracy))
                    summary.value.add(tag='Losses/Consistency', simple_value=float(consistency))
                    summary.value.add(tag='Losses/Advantage', simple_value=float(advantage))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(gradient_norms))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(var_norms))
                    self.summary_writer.add_summary(summary, episode_count)
                    self.summary_writer.flush()

                if self.number == 0:
                    if checkpoint_steps >= self.episodes_for_model_checkpoint:
                        checkpoint_steps = 0
                        saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
                        print("Saved Model")
                    else:
                        checkpoint_steps += 1
                        
                sess.run(self.increment)

