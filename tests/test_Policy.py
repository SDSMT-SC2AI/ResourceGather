import context
import os
import shelve
import numpy as np
import tensorflow as tf
from network import Policy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def test_policy():
    tf.reset_default_graph()
    tf.set_random_seed(0)
    policy = Policy('global', policy_spec={
        "input size": 2,
        "hidden layer size": 2,
        "number of actions": 2})

    print("Policy Tests: ")
    print("-------------------------------------------------")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        results = sess.run([policy.action, policy.policy_fn, policy.value_fn],feed_dict={
            policy.input: np.array([[1, 2]]),
            policy.exploration_rate: 1,
        })

    with shelve.open(os.path.join(os.path.dirname(__file__), 'data/network_tests')) as db:
        if 'policy' not in db:
            print(results)
            db['policy'] = results
        elif not np.array([np.any(r == t) for r, t in zip(results, db['policy'])]).all():
            print(results)
            print(db['policy'])
            if input("test_policy: Results didn't match. Update resutls? ") == "yes":
                db['policy'] = results
            else:
                print("test_policy: Test failed!")
                exit()
        else:
            print("test_policy: Test passed!")
    print()


if __name__ == "__main__":
    test_policy()
