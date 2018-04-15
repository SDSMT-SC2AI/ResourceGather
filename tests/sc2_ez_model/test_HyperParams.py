

from hyper_param_search import HyperParams


def main():
    iter = 0
    while iter < 10:
        hp = HyperParams(iter)
        print("\nTesting Hyperparam Randomization:")
        print(hp.learning_rate)
        print(hp.hidden_nodes)
        print(hp.base_xplr_rate)
        print(hp.accuracy_coef)
        print(hp.advantage_coef)
        print(hp.max_grad_norm)
        print(hp.discount)
        print(hp.consist_coef)
        iter += 1


if __name__ == "__main__":
    main()