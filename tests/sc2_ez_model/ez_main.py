import psutil
import argparse
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

TRAIN = True
PARAM_SEARCH = True

parser = argparse.ArgumentParser()
training = parser.add_mutually_exclusive_group()
seeds = parser.add_mutually_exclusive_group()
logdir_args = parser.add_mutually_exclusive_group()

training.add_argument("--train", dest="load_ckpt",
                      default=not TRAIN, action="store_false",
                      help="If specified, no checkpoint will be loaded, "
                           "and a new model will be created. (Default: {})".format(TRAIN))

training.add_argument("--test", dest="load_ckpt",
                      default=not TRAIN, action="store_true",
                      help="If specified, a checkpoint will be loaded from "
                           "logdir/model if it exists. (Default: {})".format(not TRAIN))

logdir_args.add_argument("--logdir", metavar="DIRNAME",
                    default="workers", type=str,
                    help="If specified, the model and training metrics will "
                         "be loaded or stored in this location. (Default: 'workers')")

logdir_args.add_argument("--gendir", dest="logdir", default="workerData/",
                    action="store_const", const="test_" + datetime.now().strftime("%Y%m%d%H%M%S"),
                    help="If specified, the logdir will be set to 'test_YYYYmmddHHMMSS' (Default: False)")

parser.add_argument("--hyper-parameter-search", dest="hyper_search", default=False,
                    action="store_true",
                    help="If specified, the program will spin up several experiments with different "
                         "hyper parameters, based on the seed_range. (Default: False)")

seeds.add_argument("--seed", default=0,
                   help="If a hyper parameter search is performed, and a seed_range is not specified, "
                        "a single run will occur with the hyper parameters corresponding to that seed. "
                        "The seed is appended to logdir. (Default: 0)")

seeds.add_argument("--seed_range", default=None, metavar=("LOWER", "UPPER"), nargs=2, type=int,
                   help="The range of seeds to use during the hyper parameter search, the seed is appended "
                        "to logdir. (Default: None)")

parser.add_argument("--max-episodes", dest="max_episodes", metavar="EPISODES",
                    default=10000, type=int,
                    help="Maximum number of episodes per worker thread. (Default: 10000)")


def run(flags):
    hp = flags.hp
    tf.reset_default_graph()
    print("Running", flags.logdir, "...")
    load_model = flags.load_ckpt
    agent_cls = agent.Smart
    seed = flags.seed if flags.hyper_search else ""

    agent.policy_spec.update(
        input_size=12,
        num_actions=len(actions.Action_Space.choices),
        max_episodes=int(0.8*flags.max_episodes),
        q_range=(10, 10.01),
        hidden_layer_size=hp.hidden_nodes,
        base_explore_rate=hp.base_xplr_rate,
        min_explore_rate=0.002
    )

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
            name = "worker_" + str(i)
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
                    summary_dir=flags.logdir + str(seed) + "/",
                    main=sys.modules[__name__],
                    env=env,
                    actions=actions.Action_Space(),
                    agent=agent_inst,
                    global_episodes=global_episodes,
                    buffer_min=480,
                    buffer_max=720,
                    max_episodes=flags.max_episodes
                )
            )
        saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        if load_model:
            print("Loading Model...")
            ckpt = tf.train.get_checkpoint_state(flags.logdir + "/model")
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


def __main__():
    flags = parser.parse_args()
    if flags.hyper_search:
        if flags.seed_range is None:
            flags.hp = HyperParams(flags.seed)
            run(flags)
        else:
            for seed in range(*flags.seed_range):
                flags.seed = seed
                flags.hp = HyperParams(seed)
                run(flags)
    else:
        flags.hp = HyperParams()
        run(flags)


if __name__ == "__main__":
    __main__()
