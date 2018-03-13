import context
import os
import tensorflow as tf
from tests.simple_env import simple_main
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def test_worker():
    tf.set_random_seed(0)

    print("Worker Tests: ")
    print("-------------------------------------------------")

    simple_main.__main__()


if __name__ == "__main__":
    test_worker()
