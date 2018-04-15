import random
from numpy import random as np

class HyperParams: 
    def __init__(self, seed=None):
        """Fills all self variables in self.params. Sets seed member so it can be appended to workerData folder"""        
        random.seed(seed)        
        self.params = { 'learning_rate_min'     : 0.00001,
                        'learning_rate_max'     : 0.01,
                        'h_nodes_layer_min'     : 21,
                        'h_nodes_layer_max'     : 61,
                        'base_explr_min'        : 0.40,
                        'base_explr_max'        : 0.99,
                        'rand_explr_A_min'      : 1.0,
                        'rand_explr_A_max'      : 4.0,
                        'rand_explr_B_min'      : 4.0,
                        'rand_explr_B_max'      : 10.0,
                        'acc_coef_min'          : 1.0,
                        'acc_coef_max'          : 10.0,
                        'advantage_coef_min'    : 1.0,
                        'advantage_coef_max'    : 10.0,
                        'max_grad_norm_min'     : 1.0,
                        'max_grad_norm_max'     : 40.0,
                        'discount_min'          : 0.9,
                        'discount_max'          : 1.0,
                        'consistency_coef_min'  : 0.1,
                        'consistency_coef_max'  : 0.1
                      }

        # Set hyperparams
        self.learning_rate = self.get_uniform_value(self.params['learning_rate_min'], self.params['learning_rate_max'])
        self.hidden_nodes = int(self.get_uniform_value(self.params['h_nodes_layer_min'], self.params['h_nodes_layer_max']))
        if self.hidden_nodes % 2 == 0:
            self.hidden_nodes += 1
        self.base_xplr_rate = self.get_uniform_value(self.params['base_explr_min'], self.params['base_explr_max'])
        self.alpha = 1.25
        self.beta = 5.625
        self.accuracy_coef = self.get_uniform_value(self.params['acc_coef_min'], self.params['acc_coef_max'])
        self.advantage_coef = self.get_uniform_value(self.params['advantage_coef_min'], self.params['advantage_coef_max'])
        self.max_grad_norm = self.get_uniform_value(self.params['max_grad_norm_min'], self.params['max_grad_norm_max'])
        self.discount = self.get_uniform_value(self.params['discount_min'], self.params['discount_max'])
        self.consist_coef = self.get_uniform_value(self.params['consistency_coef_min'], self.params['consistency_coef_max'])

        # Append this to workerData folder so we know which seed obtained which results
        self.seed = seed

    # Available random functions
    def get_uniform_value(self, a, b):
        """Takes a & b and returns a value using uniform distribution between the two"""
        return random.uniform(a, b)

    def get_gammavariate_value(self, alpha, beta):
        """Takes alpha & beta and returns a value using gammavariate distribution between the two"""
        return random.gammavariate(alpha, beta)

    def get_betavariate_value(self, alpha, beta):
        """Takes alpha & beta and returns a value using betavariate distribution between the 0 and 1"""
        return random.betavariate(alpha, beta)

    def get_weibullvariate_value(self, alpha, beta):
        """Takes alpha & beta and returns a value using weibullvariate distribution where alpha is the scale and beta is the shape"""
        return random.betavariate(alpha, beta)


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