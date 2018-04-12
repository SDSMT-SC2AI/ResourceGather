import psutil
import os
from os.path import abspath, join, dirname
import sys
import threading
import numpy as np
import tensorflow as tf
from time import sleep
sys.path.insert(0, abspath(join(dirname(__file__), "../..")))
import network
from worker import Worker
from tests.sc2_ez_model.environment.model import IdealizedSC2Env
import tests.sc2_ez_model.ez_agent as agent
import tests.sc2_ez_model.ez_actions as actions

global _max_score, _running_avg_score, _steps, _episodes
# noinspection PyRedeclaration
_max_score = _running_avg_score = 0
# noinspection PyRedeclaration
_steps = _episodes = []

"""
def __main__():
    max_episode_length = 300
    load_model = False
    model_path = "./test_model"

    tf.reset_default_graph()

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    with tf.device("/cpu:0"):
        global_episodes = tf.Variable(0, dtype=tf.int32, name="global_episodes", trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
        agent.policy_spec.update(actions.action_spec)
        master_network = network.Policy('global', global_episodes, agent.policy_spec)
        num_workers = psutil.cpu_count()
        # num_workers = 2

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
                Worker(i, sys.modules[__name__], env, actions, agent.Simple,
                       optimizer, model_path, global_episodes,
                       buffer_min=10, buffer_max=30))
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
            t = threading.Thread(target=(lambda: worker.work(max_episode_length, sess, coord, saver)))
            t.start()
            sleep(0.05)
            worker_threads.append(t)
        coord.join(worker_threads)
"""

if __name__ == "__main__":
    __main__()