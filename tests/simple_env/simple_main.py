import psutil
import os
from os.path import abspath, join, dirname
import sys
import threading
import numpy as np
import tensorflow as tf
sys.path.insert(0, abspath(join(dirname(__file__), "../..")))
from time import sleep
import network
from worker import Worker
from tests.simple_env.simple_env import SimpleEnv
import tests.simple_env.simple_agent as agent

global _max_score, _running_avg_score, _steps, _episodes
# noinspection PyRedeclaration
_max_score = _running_avg_score = 0
# noinspection PyRedeclaration
_steps = _episodes = []


def __main__():
    max_episode_length = 300
    gamma = .1
    load_model = False
    model_path = "./test_model"

    tf.reset_default_graph()

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    global_episodes = tf.Variable(0, dtype=tf.int32, name="global_episodes", trainable=False)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.005, decay=.96,epsilon=1e-10)
    master_network = network.Policy('global', agent.network_spec)
    num_workers = psutil.cpu_count()
    # num_workers = 1

    config = tf.ConfigProto(
        allow_soft_placement=True,
        intra_op_parallelism_threads=num_workers,
        inter_op_parallelism_threads=num_workers)
    config.gpu_options.allow_growth = True

    global _max_score, _running_avg_score, _steps, _episodes
    _max_score = 0
    _running_avg_score = 0
    _steps = np.zeros(num_workers)
    _episodes = np.zeros(num_workers)
    workers = []
    # Initialize workers
    for i in range(num_workers):
        env = SimpleEnv()
        workers.append(
            Worker(i, sys.modules[__name__], env, agent.Simple,
                   optimizer, model_path, global_episodes, buffer_size=10))
    saver = tf.train.Saver(max_to_keep=5)

    with tf.Session(config=config) as sess:
        coord = tf.train.Coordinator()
        if load_model:
            print("Loading Model...")
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        worker_threads = []
        for worker in workers:
            t = threading.Thread(target=(lambda: worker.work(max_episode_length, gamma, sess, coord, saver)))
            t.start()
            sleep(0.05)
            worker_threads.append(t)
        coord.join(worker_threads)


if __name__ == "__main__":
    __main__()