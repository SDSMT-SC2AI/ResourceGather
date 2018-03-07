import tensorflow as tf
import numpy as np
import scipy.signal
import observer
from pysc2.env import sc2_env
import typing

mse = tf.losses.mean_squared_error


# calculates y[i] = x[i] + gamma*y[i+1]
def discount(x, gamma):
	return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


# MG2033/A2C @ github.com
class BasePolicy:
	def __init__(self, sess, input_spec, output_spec, name="train"):
		self.name = name
		self.sess = sess
		self.input_spec = input_spec
		self.output_spec = output_spec
		self.input = None
		self.value_state = None
		self.action_policy = None

	def step(self, observation):
		raise NotImplementedError("step function not implemented")

	def value(self, observation):
		raise NotImplementedError("value function not implemented")


class Policy(BasePolicy):
	def __init__(self, sess, input_spec, output_spec, name="train", exploration_rate=1):
		super().__init__(sess, input_spec, output_spec, name)
		with tf.name_scope(name + "policy_input"):
			self.input = tf.placeholder(tf.uint8, input_spec["policy input shape"])
		with tf.variable_scope("policy", reuse=tf.AUTO_REUSE):
			hidden1 = tf.contrib.layers.fully_connected(
				inputs=self.input,
				num_outputs=output_spec["number of hidden nodes"],
				activation_fn=tf.nn.elu,
				biases_initializer=tf.initializers.random_uniform(-1, 1)
			)
			self.value_fn = tf.contrib.layers.fully_connected(
				inputs=hidden1,
				num_outputs=1
			)
			policy_raw = tf.contrib.layers.fully_connected(
				inputs=hidden1,
				num_outputs=output_spec["number of actions"],
				weights_initializer=tf.initializers.orthogonal()
			)
			self.policy_fn = policy_raw - tf.reduce_max(policy_raw, 1, keep_dims=True)

			with tf.name_scope("value"):
				self.value = self.value_fn[:, 0]

			with tf.name_scope("action"):
				self.exploration_rate = tf.Variable(exploration_rate + 1e-3, trainable=False)
				self.action = tf.multinomial(self.policy_fn / exploration_rate, 1)

	def step(self, observation, exploration_rate=1):
		# returns tuple(action, value, policy):
		#   a random action according to policy, the value function result, and current policy
		return self.sess.run([self.action, self.value, self.policy_fn],
							 feed_dict={self.input: observation,
										self.exploration_rate: exploration_rate + 1e-3})

	def value(self, observation, exploration_rate=1):
		# returns the value
		return self.sess.run(self.value,
							 feed_dict={self.input: observation,
										self.exploration_rate: exploration_rate + 1e-3})


class Model:
	def __init__(self, sess,
				 entropy_coeff=0.01,
				 value_function_coeff=0.5,
				 max_gradient_norm=1,
				 optimizer_params=None,
				 args=None):
		self.actions = None
		self.advantage = None
		self.reward = None
		self.policy_gradient_loss = None
		self.value_function_loss = None
		self.optimize = None
		self.entropy = None
		self.loss = None
		self.learning_rate = None
		self.num_actions = None
		self.input_spec = None
		self.output_spec = None

		self.policy = Policy
		self.sess = sess
		self.value_function_coeff = value_function_coeff
		self.entropy_coeff = entropy_coeff
		self.max_gradient_norm = max_gradient_norm
		self.optimizer_params = optimizer_params

	def init_input(self):
		with tf.name_scope('input'):
			self.actions = tf.placeholder(tf.int32, [None])
			self.advantage = tf.placeholder(tf.float32, [None])
			self.reward = tf.placeholder(tf.float32, [None])
			self.learning_rate = tf.placeholder(tf.float32, [None])

	def init_network(self):
		self.policy = self.policy(self.sess, self.input_spec, self.output_spec)
		with tf.variable_scope('train_output'):
			negative_log_prob_action = tf.nn.sparse_softmax_cross_entropy_with_logits(
				logits=self.policy.policy_fn,
				labels=self.actions)
			self.policy_gradient_loss = tf.reduce_mean(self.advantage * negative_log_prob_action)
			self.value_function_loss = mse(self.reward, tf.squeeze(self.policy.value_fn))
			self.entropy = tf.reduce_sum(tf.exp(self.policy.policy_fn) * self.policy.policy_fn, axis=1)
			self.loss = self.policy_gradient_loss \
						- self.entropy_coeff * self.entropy \
						+ self.value_function_coeff * self.value_function_loss

			with tf.variable_scope("policy"):
				params = tf.trainable_variables()
			grads = tf.gradients(self.loss, params)

			# gradient clipping
			if self.max_gradient_norm is not None:
				grads, grad_norm = tf.clip_by_global_norm(grads, self.max_gradient_norm)

			grads = list(zip(grads, params))
			optimizer = tf.train.AdamOptimizer(**self.optimizer_params)
			self.optimize = optimizer.apply_gradients(grads)

	def build(self, input_spec, output_spec):
		self.output_spec = output_spec
		self.input_spec = input_spec
		self.init_input()
		self.init_network()


class Worker():
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
		observations = rollout[:, 0]
		actions = rollout[:, 1]
		rewards = rollout[:, 2]
		values = rollout[:, 3]

		discounted_rewards = discount(rewards, gamma)
		values_plus = np.concatenate([values, np.array([bootstrap_value])])
		advantages = discount(rewards + gamma * values_plus[1:] - values_plus[:-1], gamma)

		value_loss, \
		policy_loss, \
		entropy, \
		grad_norms, \
		var_norms, _ = self.agent.train(sess, discounted_rewards, observations, actions, advantages)

		return value_loss / len(rollout), policy_loss / len(rollout), entropy / len(rollout), \
			   grad_norms, var_norms

	def work(self, max_episode_length, gamma, sess, coord, saver):
		episode_count = sess.run(self.global_episodes)
		total_steps = 0
		print("Starting worker " + str(self.number))
		with sess.as_default(), sess.graph.as_default():
			while not coord.should_stop():
				self.agent.updatePolicy()

				episode_buffer = []
				episode_values = []
				episode_frames = []
				episode_reward = 0
				episode_step_count = 0
				episode_end = False

				# Start new episode
				env_obs = self.env.reset()
				episode_frames.append(env_obs[0])  ##CHECK USAGE LATER
				reward, obs, episode_end = observer.process_observation(env_obs, self.flags)
				pobs = obs

				while not episode_end:
					actions, choice, action_dist, value = self.agent.step(sess, obs)
					env_obs = self.env.step(actions=actions)
					reward, obs, episode_end = observer.process_observation(obs, self.flags)

					if not episode_end:
						episode_frames.append(obs[0])

					episode_buffer.append([pobs, choice, reward, obs, value[0, 0]])
					episode_values.append(value[0, 0])

					episode_reward += reward
					pobs = obs
					total_steps += 1
					episode_step_count += 1

					if episode_end:
						break

					if len(episode_buffer) == self.buffer_size and episode_step_count != max_episode_length - 1:
						bootstrap = self.agent.value(sess, obs)
						value_loss, \
						policy_loss, \
						entropy, \
						gradient_norms, \
						var_norms = self.train(episode_buffer, sess, gamma, bootstrap)

				self.episode_rewards.append(episode_reward)
				self.episode_lengths.append(episode_step_count)
				self.episode_mean_values.append(np.mean(episode_values))
				episode_count += 1

				global _max_score, _running_avg_score, _episodes, _steps
				if _max_score < episode_reward:
					_max_score = episode_reward
				_running_avg_score = (2.0 / 101) * (episode_reward - _running_avg_score) + _running_avg_score
				_episodes[self.number] = episode_count
				_steps[self.number] = total_steps

				print(
					"{} Step #{} Episode #{} Reward: {}".format(self.name, total_steps, episode_count, episode_reward))
				print("Total Steps: {}\tTotal Episodes: {}\tMax Score: {}\tAvg Score: {}".format(np.sum(_steps),
																								 np.sum(_episodes),
																								 _max_score,
																								 _running_avg_score))

				# Update the network using the episode buffer at the end of the episode
				if len(episode_buffer) != 0:
					value_loss, \
					policy_loss, \
					entropy, \
					gradient_norms, \
					var_norms = self.train(episode_buffer, sess, gamma, 0.0)

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


class BaseTrainer:
	def __init__(self, sess, model, args):
		self.model = model
		self.args = args
		self.sess = sess

		self.summary_placeholders = {}
		self.summary_ops = {}

		self.__init_global_saver()

	def save(self):
		print("Saving model...")
		self.saver.save(self.sess, self.args.checkpoint_dir)

	def __init_global_saver(self):
		self.saver = tf.train.Saver(max_to_keep=self.args.max_to_keep)
		self.summary_writer = tf.summary.FileWriter(self.args.summary_dir, self.sess.graph)

	def __init_model(self):
		self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		self.sess.run(self.init)

	def __load_model(self):
		latest_checkpoint = tf.train.latest_checkpoint(self.args.checkpoint_dir)
		if latest_checkpoint:
			print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
			self.saver.restore(self.sess, latest_checkpoint)
			print("Checkpoint loaded\n\n")
		else:
			print("No checkpoints available!\n\n")


class Trainer(BaseTrainer):
	def __init__(self, sess, model, args):
		pass


class A2C:
	def __init__(self, input_spec, action_spec: typing.List[int]):
		pass
