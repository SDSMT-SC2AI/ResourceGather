from utils import parse_args, ensure_dir

import os
import tensorflow as tf
from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import run_loop
from pysc2.env import sc2_env
from pysc2.lib import stopwatch

from dummy import Dummy

FLAGS = None


def main():
    FLAGS = parse_args()
    stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace
    stopwatch.sw.trace = FLAGS.trace

    agent_cls = Dummy
    maps.get(FLAGS.map or Dummy.map_name)

    tf.reset_default_graph()
    config = tf.ConfigProto(
        allow_soft_placement=True,
        intra_op_parallelism_threads=FLAGS.num_envs,
        inter_op_parallelism_threads=FLAGS.num_envs)
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)

    # Make sure all the directories are ready to go
    ensure_dir(FLAGS.base_output_dir)
    for _dir in [FLAGS.summary_dir, FLAGS.checkpoint_dir, FLAGS.logging_dir, FLAGS.test_dir]:
        ensure_dir(os.path.join(FLAGS.base_output_dir, _dir))

    a2c = A2C(sess, config_args)