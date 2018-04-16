import psutil
from os.path import abspath, join, dirname
from datetime import datetime
import sys
import threading
import numpy as np
import tensorflow as tf
from time import sleep
sys.path.insert(0, abspath(join(dirname(__file__), "../..")))
from worker import Worker
from tests.sc2_ez_model.environment.model import IdealizedSC2Env
import tests.sc2_ez_model.ez_agent as agent
import tests.sc2_ez_model.ez_actions as actions

from hyper_param_search import HyperParams


def __main__():
    seed = 'harharhar'
    while seed == 'harharhar':
        hp = HyperParams(seed)
        tf.reset_default_graph()   

        try:
            run_id = sys.argv[1]
        except IndexError:
            run_id = "test_" + datetime.now().strftime("%Y%m%d%H%M%S")
        print("Running", run_id)
        load_model = False

        agent_cls = agent.Smart

        agent.policy_spec.update(      
                input_size=12,
                num_actions=len(actions.Action_Space.choices),
                max_episodes=1000000,
                q_range=(10, 10.01),
                hidden_layer_size=hp.hidden_nodes,
                base_explore_rate=hp.base_xplr_rate,
                min_explore_rate=0.002
            )

        tf.reset_default_graph()
        #config = tf.ConfigProto(allow_soft_placement=True)

        with tf.device("/cpu:0"):
            global_episodes = tf.Variable(0, dtype=tf.int32, name="global_episodes", trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=hp.learning_rate)
            master_network = agent.network.Policy('global', global_episodes, agent.policy_spec, hp)
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
                name = "worker_" + str(i) + run_id
                agent_inst = agent_cls(name, 'global', optimizer, global_episodes, actions.Action_Space(), hp)
                print('Initializing environment #{}...'.format(i))
                env = IdealizedSC2Env(
                        game_loops_per_agent_step=5,
                        time_limit=720, 
                        silent_errors=False, 
                        verbose=False
                )
                workers.append(
                    Worker(
                        number=i,
                        summary_dir="workerData" + str(seed) + "/",
                        main=sys.modules[__name__],
                        env=env,
                        actions=actions.Action_Space(),
                        agent=agent_inst,
                        global_episodes=global_episodes,
                        buffer_min=480,
                        buffer_max=720,
                        max_episodes=125000
                    )
                )
            saver = tf.train.Saver(max_to_keep=5)

        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            if load_model:
                print("Loading Model...")
                ckpt = tf.train.get_checkpoint_state("workerData/model")
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())

            worker_threads = []
            for worker in workers:
                t = threading.Thread(target=(lambda: worker.work(sess, coord, saver)))
                t.start()
                sleep(0.05)
                worker_threads.append(t)
            coord.join(worker_threads)
        seed = 'hahaha'

if __name__ == "__main__":
    __main__()
