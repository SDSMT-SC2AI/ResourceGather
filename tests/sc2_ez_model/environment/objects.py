class Base:
    def __init__(self, parent):
        self.parent = parent
        self.rally_set = False
        self.needs_attention = True
        self.minerals = Minerals(parent)
        self.geyserA = Geyser(parent)
        self.geyserB = Geyser(parent)
        self.larva = 0
        self.unassigned_drones = 0
        self.queens = 0
        self.queens_queued = 0

    def get_drones(self):
        return self.minerals.drones, self.geyserA.drones, self.geyserB.drones

    def production(self):
        return self.minerals.collect(), self.geyserA.collect() + self.geyserB.collect()

    def tick(self):
        if self.unassigned_drones > 0 \
                or (1.05 - self.minerals.drones / (1 + self.minerals.max_drones)) > 0.05\
                or (1.05 - self.geyserA.drones  / (1 + self.geyserA.max_drones))  > 0.05\
                or (1.05 - self.geyserB.drones  / (1 + self.geyserB.max_drones))  > 0.05:
            self.needs_attention = True

        return self.production()


class Resource:
    def __init__(self, parent, max_capacity, max_drones, discount_per_drone, rate_per_drone):
        self.parent = parent
        self._max_capacity = max_capacity
        self.max_drones = max_drones
        self._discount_per_drone = discount_per_drone
        self._rate_per_drone = rate_per_drone
        self.drones = 0
        self._capacity = self._max_capacity

    def collect(self):
        equiv_max = self.max_drones * self._capacity / self._max_capacity
        equiv_tot = min(equiv_max, self.drones) + \
                    Resource.discount(self._discount_per_drone, max(0, self.drones - equiv_max))
        rate = equiv_tot * self._rate_per_drone * self.parent.clock_rate
        self._capacity -= max(rate, self._capacity)
        return max(rate, self._capacity)

    @staticmethod
    def discount(rate, n):
        return (rate - rate**(n+1))/(1 - rate)


class Minerals(Resource):
    def __init__(self, parent):
        super().__init__(
            parent=parent,
            max_capacity=14400,
            max_drones=16,
            discount_per_drone=0.5,
            rate_per_drone=1.25
        )


class Geyser(Resource):
    def __init__(self, parent):
        super().__init__(
            parent=parent,
            max_capacity=2250,
            max_drones=3,
            discount_per_drone=0.5,
            rate_per_drone=0.94
        )
        self.has_extractor = False

    def collect(self):
        if self.has_extractor:
            return super().collect()
        else:
            return 0
