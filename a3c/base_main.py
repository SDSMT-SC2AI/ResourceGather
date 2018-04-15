import sys, psutil, threading
import numpy as np
import tensorflow as tf
from datetime import datetime
from time import sleep

__DEBUG__ = False


class BaseMain:
    def __init__(self, run_dir, agent_module, action_cls, agent_cls, env_cls, worker_cls, max_episodes, buffer_min, buffer_max):
        self.run_dir = run_dir
        self.agent_module = agent_module
        self.action_cls = action_cls
        self.agent_cls = agent_cls
        self.env_cls = env_cls
        self.worker_cls = worker_cls
        self.max_episode_length = 720
        self.saver = None
        self.global_writer = None
        self.load_model = False
        self.buffer_min = max(0, buffer_min)
        self.buffer_max = max(1, buffer_max)
        self.max_episodes = max(1, max_episodes)
        self.get_optimizer = lambda: tf.train.AdamOptimizer(learning_rate=0.00001)

        global __DEBUG__
        self.num_workers = 1 if __DEBUG__ else psutil.cpu_count()

        # watches
        self.max_score = 0
        self.running_avg_score = 0
        self.steps = np.zeros(self.num_workers)
        self.episodes = np.zeros(self.num_workers)

    def main(self, run_dir=None):
        self.run_dir = run_dir or "test_" + datetime.now().strftime("%Y%m%d%H%M%S") + "/"
        self.model_path = run_dir + "model"
        print("Running", run_dir)

        tf.reset_default_graph()
        config = tf.ConfigProto(allow_soft_placement=True)

        workers = self._setup()

        with tf.Session(config=config) as sess:
            coord = tf.train.Coordinator()
            self.global_writer = tf.summary.FileWriter(run_dir + "global", sess.graph)
            if self.load_model:
                print("Loading Model...")
                ckpt = tf.train.get_checkpoint_state(self.model_path)
                self.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())

            worker_threads = []
            for worker in workers:
                t = threading.Thread(target=(lambda: worker.work(self.max_episode_length, sess, coord, self.saver)))
                t.start()
                sleep(0.25)
                worker_threads.append(t)
            coord.join(worker_threads)

    def _setup(self):
        with tf.device("/cpu:0"):
            global_episodes = tf.Variable(0, dtype=tf.int32, name="global_episodes", trainable=False)
            master_network = self.agent_module.network.Policy('global', global_episodes, self.agent_module.policy_spec)

            workers = []
            # Initialize workers
            for i in range(self.num_workers):
                name = "worker_" + str(i) + self.run_dir
                agent = self.setup_agent(
                    name=name,
                    scope='global',
                    optimizer=self.get_optimizer(),
                    global_episodes=global_episodes
                )

                print('Initializing environment #{}...'.format(i))
                env = self.setup_env()
                workers.append(
                    self.setup_worker(
                        name=name, number=i,
                        main_inst=self, env=env, agent=agent,
                        model_path=self.model_path,
                        global_episodes=global_episodes,
                        buffer_min=self.buffer_min, buffer_max=self.buffer_max,
                        max_episodes=self.max_episodes
                    )
                )
            self.saver = tf.train.Saver(max_to_keep=5)
            return workers

    def setup_agent(self, name, scope, optimizer, global_episodes):
        raise NotImplementedError("Need to define a function that sets up the agent.\n"
                                  "Signature: setup_agent(self, name, scope, optimizer, global_episodes)")

    def setup_env(self):
        raise NotImplementedError("Need to define a function that sets up the agent.\n"
                                  "Signature: setup_env(self)")

    def setup_worker(self, name, number, main_inst, env, agent, model_path, global_episodes, buffer_min, buffer_max, max_episodes):
        raise NotImplementedError("Need to define a function that sets up the agent.\n"
                                  "Signature: setup_env(self)")
