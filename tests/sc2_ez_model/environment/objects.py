from math import sqrt

class Base:
    larva_gen_rate = 11 # Every 11 seconds
    def __init__(self, parent):
        self.parent = parent
        self.needs_attention = True
        self.minerals = Minerals(parent)
        self.geyserA = Geyser(parent)
        self.geyserB = Geyser(parent)
        self.larva = 3
        self.unassigned_drones = 0
        self.queens = 0
        self.queens_queued = 0
        self.index = parent.base_index
        self.drones_queued = 0
        self.injectable = True

    def get_drones(self):
        return self.minerals.drones, self.geyserA.drones, self.geyserB.drones

    def production(self):
        
        mins = self.minerals.collect()
        gasA = self.geyserA.collect()
        gasB = self.geyserB.collect()
        # print("Equiv_max: ", self.minerals.equiv_max)
        diff_drones = int(self.minerals.drones - self.minerals.equiv_max)
        if diff_drones > 0:
            self.unassigned_drones += diff_drones
            self.minerals.drones -= diff_drones
            # self.assign_drones()

        diff_drones = int(self.geyserA.drones - self.geyserA.equiv_max)
        if diff_drones > 0:
            self.unassigned_drones += diff_drones
            self.geyserA.drones -= diff_drones
            # self.assign_drones()

        diff_drones = int(self.geyserB.drones - self.geyserB.equiv_max)
        if diff_drones > 0:
            self.unassigned_drones += diff_drones
            self.geyserB.drones -= diff_drones
            # self.assign_drones()

        # if self.index == 0:
            # print("Unassigned Workers: ", self.unassigned_drones)
        self.move_workers()
        self.assign_drones()
        return mins, gasA + gasB

    def tick(self):
        self.larva += self.parent.clock_rate / self.larva_gen_rate if self.larva < 3 else 0
        # if self.unassigned_drones > 0:
        self.assign_drones()
        return self.production()


    def assign_drones(self):
        for _ in range(self.unassigned_drones):
            if self.minerals.drones < self.minerals.equiv_max:
                self.unassigned_drones -= 1
                self.minerals.drones += 1
            elif self.geyserA.has_extractor and self.geyserA.drones < self.geyserA.equiv_max:
                self.unassigned_drones -= 1
                self.geyserA.drones += 1
            elif self.geyserB.has_extractor and self.geyserB.drones < self.geyserB.equiv_max:
                self.unassigned_drones -= 1
                self.geyserB.drones += 1
            else:
                # If this base is full on drones need to assign unassigned to another base
                self.minerals.drones += 1
                self.unassigned_drones -= 1 

    # If we have another base to move drones to we push these unassigned_drones down the list, otherwise
    # they sit unassigned until a new base is built
    def move_workers(self):
        next_base_idx = self.index + 1
        if next_base_idx < len(self.parent.bases):
            self.parent.bases[next_base_idx].unassigned_drones += self.unassigned_drones
            self.unassigned_drones = 0


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
        return self.max_drones * (self.remaining / self.max_capacity)**0.3

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
            discount_per_drone=0.75,
            rate_per_drone=1.05
        )


class Geyser(Resource):
    def __init__(self, parent):
        super().__init__(
            parent=parent,
            max_capacity=2250,
            max_drones=3,
            discount_per_drone=0.75,
            rate_per_drone=0.63
        )
        self.has_extractor = False

    def collect(self):
        if self.has_extractor:
            return super().collect()
        else:
            return 0
