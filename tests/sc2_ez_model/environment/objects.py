class Base:
    def __init__(self, parent):
        self.parent = parent
        self.rally_set = False
        self.needs_attention = True
        self.minerals = Minerals(parent)
        self.geyserA = Geyser(parent)
        self.geyserB = Geyser(parent)
        self.larva = 3
        self.unassigned_drones = 0
        self.queens = 0
        self.queens_queued = 0

    def get_drones(self):
        return self.minerals.drones, self.geyserA.drones, self.geyserB.drones

    def production(self):
        return self.minerals.collect(), self.geyserA.collect() + self.geyserB.collect()

    def tick(self):
        self.larva += self.parent.clock_rate / 11 if self.larva < 3 else 0
        if self.unassigned_drones > 0 \
                or (1.06 - (self.minerals.drones + .1) / (.1 + self.minerals.max_drones)) > 0.05\
                or (1.06 - (self.geyserA.drones + .1) / (.1 + self.geyserA.max_drones)) > 0.05\
                or (1.06 - (self.geyserB.drones + .1) / (.1 + self.geyserB.max_drones)) > 0.05:
            self.needs_attention = True

        return self.production()

    @property
    def resource_collection_rate(self):
        rate = self.minerals.rate
        if self.geyserA.has_extractor:
            rate += self.geyserA.rate
        if self.geyserB.has_extractor:
            rate += self.geyserB.rate

        return rate / self.parent.clock_rate


class Resource:
    def __init__(self, parent, max_capacity, max_drones, discount_per_drone, rate_per_drone):
        self.parent = parent
        self.max_capacity = max_capacity
        self.max_drones = max_drones
        self.discount_per_drone = discount_per_drone
        self.rate_per_drone = rate_per_drone
        self.drones = 0
        self.remaining = self.max_capacity

    def collect(self):
        collected = min(self.rate, self.remaining)
        self.remaining -= collected
        return collected

    @property
    def equiv_max(self):
        return self.max_drones * self.remaining / self.max_capacity

    @property
    def rate(self):
        equiv_tot = min(self.equiv_max, self.drones) + \
            Resource.discount(self.discount_per_drone, max(0, self.drones - self.equiv_max))
        return equiv_tot * self.rate_per_drone * self.parent.clock_rate

    @staticmethod
    def discount(rate, n):
        return (rate - rate**(n+1))/(1 - rate)


class Minerals(Resource):
    def __init__(self, parent):
        super().__init__(
            parent=parent,
            max_capacity=9600,
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
