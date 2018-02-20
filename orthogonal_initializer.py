def orthogonal_initializer(shape, axis=0, pop_size = 100):
    initial_population = [numpy.random.normal(size=shape) for _ in range(pop_size)]
    def loss(x):
        np.matmul()