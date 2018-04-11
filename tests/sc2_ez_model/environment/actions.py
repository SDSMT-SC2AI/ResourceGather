from objects import Base, Resource, Geyser


class ActionError(Exception):
    pass


class Action:
    def __init__(self, parent, time_to_complete):
        self.time_to_complete = time_to_complete
        self.time_remaining = self.time_to_complete
        self.parent = parent
        self.verify()
        self.on_start()

    def tick(self):
        self.time_remaining -= self.parent.clock_rate
        self.on_step()
        if self.time_remaining == 0:
            self.on_complete()
            return True

    def on_start(self):
        pass

    def on_step(self):
        pass

    def on_complete(self):
        pass

    def verify(self):
        pass


class Build(Action):
    def __init__(self, parent, time_to_complete, mineral_cost, gas_cost):
        self.mineral_cost = mineral_cost
        self.gas_cost = gas_cost
        super().__init__(parent, time_to_complete)

    def on_start(self):
        super().on_start()
        self.parent.minerals -= self.mineral_cost
        self.parent.gas -= self.gas_cost

    def verify(self):
        super().verify()
        if self.parent.minerals < self.mineral_cost or self.parent.gas < self.gas_cost:
            raise ActionError("Insufficient Resources")


class BuildDrone(Build):
    def __init__(self, parent):
        super().__init__(parent, time_to_complete=12, mineral_cost=50, gas_cost=0)
        self.base = self.parent.focus

    def on_start(self):
        super().on_start()
        self.base.larva -= 1

    def on_complete(self):
        super().on_complete()
        if self.base.rally_set:
            self.base.minerals.drones += 1
        else:
            self.base.unassigned_drones += 1

    def verify(self):
        super().verify()
        if not isinstance(self.parent.focus, Base):
            raise ActionError("Base Not Selected", type(self.parent.focus))

        if self.parent.focus.larva <= 0:
            raise ActionError("Not Enough Larva", self.parent.focus.larva)


class BuildBase(Build):
    def __init__(self, parent):
        super().__init__(parent, time_to_complete=71, mineral_cost=300, gas_cost=0)

    def on_start(self):
        super().on_start()
        if isinstance(self.parent.focus, Base):
            self.parent.focus.unassigned_drones -= 1

        elif isinstance(self.parent.focus, Resource):
            self.parent.focus.drones -= 1

    def on_complete(self):
        super().on_complete()
        self.parent.bases.append(Base())

    def verify(self):
        super().verify()
        if isinstance(self.parent.focus, Base):
            n_drones = self.parent.focus.unassigned_drones
        elif isinstance(self.parent.focus, Resource):
            n_drones = self.parent.focus.drones
        else:
            raise ActionError("Selection is not a Base or a Resource")

        if n_drones <= 0:
            raise ActionError("No Available Drones from Selection")


class BuildQueen(Build):
    def __init__(self, parent):
        super().__init__(parent, time_to_complete=36, mineral_cost=150, gas_cost=0)
        self.base = self.parent.focus

    def on_start(self):
        super().on_start()
        self.base.queens_queued += 1

    def on_complete(self):
        super().on_complete()
        self.base.queens += 1
        self.base.queens_queued -= 1

    def verify(self):
        super().verify()
        if not self.parent.spawning_pool:
            raise ActionError("Building a Queen Requires a Spawning Pool")

        if not isinstance(self.parent.focus, Base):
            raise ActionError("Base Not Selected", type(self.parent.focus))

        if self.parent.focus.queens_queued >= 5:
            raise ActionError("No More Queens can Be Queued", self.parent.focus.queens_queued)


class BuildOverlord(Build):
    def __init__(self, parent):
        super().__init__(parent, time_to_complete=18, mineral_cost=100, gas_cost=0)
        self.base = self.parent.focus

    def on_start(self):
        super().on_start()
        self.base.larva -= 1

    def on_complete(self):
        super().on_complete()
        self.parent.overlords += 1

    def verify(self):
        super().verify()
        if not isinstance(self.parent.focus, Base):
            raise ActionError("Base Not Selected", type(self.parent.focus))

        if self.parent.focus.larva <= 0:
            raise ActionError("Not Enough Larva", self.parent.focus.larva)


class BuildSpawningPool(Build):
    def __init__(self, parent):
        super().__init__(parent, time_to_complete=46, mineral_cost=200, gas_cost=0)

    def on_start(self):
        super().on_start()
        if isinstance(self.parent.focus, Base):
            self.parent.focus.unassigned_drones -= 1

        elif isinstance(self.parent.focus, Resource):
            self.parent.focus.drones -= 1

    def on_complete(self):
        super().on_complete()
        self.parent.spawning_pool = True

    def verify(self):
        super().verify()
        if isinstance(self.parent.focus, Base):
            n_drones = self.parent.focus.unassigned_drones
        elif isinstance(self.parent.focus, Resource):
            n_drones = self.parent.focus.drones
        else:
            raise ActionError("Selection is not a Base or a Resource")

        if n_drones <= 0:
            raise ActionError("No Available Drones from Selection")


class BuildExtractor(Build):
    def __init__(self, parent, geyser):
        self.geyser = geyser
        super().__init__(parent, time_to_complete=21, mineral_cost=25, gas_cost=0)

    def on_start(self):
        super().on_start()
        if isinstance(self.parent.focus, Base):
            self.parent.focus.unassigned_drones -= 1

        elif isinstance(self.parent.focus, Resource):
            self.parent.focus.drones -= 1

    def on_complete(self):
        super().on_complete()
        self.geyser.has_extractor = True

    def verify(self):
        super().verify()
        if not isinstance(self.geyser, Geyser):
            raise ActionError("Secondary Selection Not a Geyser")
        if self.geyser.has_extractor:
            raise ActionError("Geyser Already has an Extractor")

        if isinstance(self.parent.focus, Base):
            n_drones = self.parent.focus.unassigned_drones
        elif isinstance(self.parent.focus, Resource):
            n_drones = self.parent.focus.drones
        else:
            raise ActionError("Selection is not a Base or a Resource")

        if n_drones <= 0:
            raise ActionError("No Available Drones from Selection")


class Select(Action):
    def __init__(self, parent, thing):
        self.thing = thing
        super().__init__(parent, time_to_complete=0)

    def on_complete(self):
        super().on_complete()
        self.parent.focus = self.thing()


class SetRallyMinerals(Action):
    def __init__(self, parent):
        super().__init__(parent, time_to_complete=0)

    def on_complete(self):
        super().on_complete()
        self.parent.focus.rally_set = True

    def verify(self):
        super().verify()
        if not isinstance(self.parent.focus, Base):
            raise ActionError("Base Not Selected", type(self.parent.focus))


class TransferDrone(Action):
    def __init__(self, parent, home):
        self.home = home
        super().__init__(parent, time_to_complete=10)

    def on_start(self):
        super().on_start()
        if isinstance(self.parent.focus, Base):
            self.parent.focus.unassigned_drones -= 1

        elif isinstance(self.parent.focus, Resource):
            self.parent.focus.drones -= 1

    def on_complete(self):
        super().on_complete()
        if isinstance(self.home, Base):
            if self.home.rally_set:
                self.home.minerals.drones += 1
            else:
                self.home.unassigned_drones += 1

        if isinstance(self.home, Resource):
            self.home.drones += 1

    def verify(self):
        super().verify()
        if isinstance(self.parent.focus, Base):
            n_drones = self.parent.focus.unassigned_drones
        elif isinstance(self.parent.focus, Resource):
            n_drones = self.parent.focus.drones
        else:
            raise ActionError("Selection is not a Base or a Resource")

        if n_drones <= 0:
            raise ActionError("No Available Drones from Selection")


class InjectLarva(Action):
    def __init__(self, parent):
        super().__init__(parent, time_to_complete=29)
        self.base = self.parent.focus

    def on_complete(self):
        super().on_complete()
        self.base.larva = min(19, self.base.larva + 3)

    def verify(self):
        super().verify()
        if not isinstance(self.parent.focus, Base):
            raise ActionError("Base Not Selected", type(self.parent.focus))

        if self.parent.focus.queens <= 0:
            return ActionError("Queen Required to Inject Larva")



