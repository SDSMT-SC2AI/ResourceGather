
class Base:
    index = 0
    larva_generation = 11 # Every 11 seconds
    def __init__(self, parent):
        self.index = Base.index
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
        self.last_larva_gen = parent
        Base.index += 1

    def get_drones(self):
        return self.minerals.drones, self.geyserA.drones, self.geyserB.drones

    def generate_larva(self):        
        if self.parent.GetTime() - 

    def production(self):
        mins = self.minerals.collect()
        gasA = self.geyserA.collect()
        gasB = self.geyserB.collect()
        if mins = 0:
            self.unassigned_drones += self.minerals.drones
            self.assign_drones()
        if gasA = 0:
            self.unassigned_drones += self.geyserA.drones
            self.assign_drones()
        if gasB = 0:
            self.unassigned_drones += self.geyserB.drones
            self.assign_drones()

        return mins, gasA + gasB

    def tick(self):
        if self.unassigned_drones > 0 \
                or (1.05 - self.minerals.drones / (1 + self.minerals.max_drones)) > 0.05\
                or (1.05 - self.geyserA.drones  / (1 + self.geyserA.max_drones))  > 0.05\
                or (1.05 - self.geyserB.drones  / (1 + self.geyserB.max_drones))  > 0.05:            
            self.assign_drones()
        return self.production()


    def assign_drones(self):
        for _ in range(self.unassigned_drones):
            if self.minerals.drones < 16:
                self.rally_set = True
                self.unassigned_drones -= 1
                self.minerals.drones += 1
            elif self.geyserA.has_extractor and self.geyserA.drones < 3:
                self.unassigned_drones -= 1
                self.geyserA.drones += 1
            elif self.geyserB.has_extractor and self.geyserB.drones < 3:
                self.unassigned_drones -= 1
                self.geyserB.drones += 1
            else:
                # If this base is full on drones need to assign unassigned to another base
                self.needs_attention = True
                return
        self.needs_attention = False

    # If we have another base to move drones to we push these unassigned_drones down the list, otherwise
    # they sit unassigned until a new base is built
    def move_workers(self):
        next_base_idx = self.index + 1
        if len(self.parent.bases) < next_base_idx:
            self.parent.bases[next_base_idx].unassigned_drones += self.unassigned_drones
            self.parent.bases[next_base_idx].assign_drones()
            self.unassigned_drones = 0
            self.needs_attention = False

    # For resetting the environment
    @staticmethod
    def reset(self):
        index = 0


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
        if not depleted():
            equiv_max = self.max_drones * self._capacity / self._max_capacity
            equiv_tot = min(equiv_max, self.drones) + \
                        Resource.discount(self._discount_per_drone, max(0, self.drones - equiv_max))
            rate = equiv_tot * self._rate_per_drone * self.parent.clock_rate
            self._capacity -= max(rate, self._capacity)
            # Subtract out any negative capacity
            if depleted():
                rate += self._capacity
                self._capacity = 0
            return max(rate, self._capacity)
        else:
            return 0

    def depleted(self):
        return self._capacity <= 0

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
