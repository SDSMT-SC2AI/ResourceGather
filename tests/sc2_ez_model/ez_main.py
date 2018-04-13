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


def __main__():
    max_episode_length = 720
    load_model = False
    model_path = './model'

    agent_cls = agent.Smart

    agent.policy_spec.update(      
            input_size=14,
            num_actions=len(actions.Action_Space.choices),
            max_episodes=1000000,
            q_range=(10000, 10005),
            hidden_layer_size=30,
            base_explore_rate=0.15,                 
            min_explore_rate=0.002
        )

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    tf.reset_default_graph()
    config = tf.ConfigProto(allow_soft_placement=True)

    with tf.device("/cpu:0"):
        global_episodes = tf.Variable(0, dtype=tf.int32, name="global_episodes", trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
        master_network = agent.network.Policy('global', global_episodes, agent.policy_spec)
        num_workers = psutil.cpu_count()
        # num_workers = 1 # Hardcoded to one for quicker testing

        global _max_score, _running_avg_score, _steps, _episodes
        _max_score = 0
        _running_avg_score = 0
        _steps = np.zeros(num_workers)
        _episodes = np.zeros(num_workers)
        workers = []
        # Initialize workers
        for i in range(num_workers):
            name = "worker_" + str(i)
            agent_inst = agent_cls(name, 'global', optimizer, global_episodes, actions.Action_Space())
            env = IdealizedSC2Env(
                    game_loops_per_agent_step=1, 
                    time_limit=720, 
                    silent_errors=False, 
                    verbose=False
            )
            workers.append(
                Worker(
                    name=name,
                    number=i,
                    main=sys.modules[__name__],
                    env=env,
                    actions=actions.Action_Space(),
                    agent=agent_inst,
                    model_path=model_path,
                    global_episodes=global_episodes,
                    buffer_min=105,
                    buffer_max=210,
                    max_episodes=1000000
                )
            )
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
            sleep(0.25)
            worker_threads.append(t)
        coord.join(worker_threads)


if __name__ == "__main__":
    __main__()