from .objects import Base, Resource, Geyser


class ActionError(Exception):
    pass


class Action:
    time_to_complete = 0

    def __init__(self, parent):
        self.time_remaining = self.time_to_complete
        self.parent = parent
        self.verify(parent)
        self.on_start()

    def tick(self):
        self.time_remaining -= self.parent.clock_rate
        self.on_step()
        if self.time_remaining <= 0:
            self.on_complete()
            return True
        else:
            return False

    def on_start(self):
        pass

    def on_step(self):
        pass

    def on_complete(self):
        pass

    @classmethod
    def verify(cls, env):
        pass


class NoOp(Action):
    pass


class Build(Action):
    mineral_cost = 0
    gas_cost = 0

    def on_start(self):
        super().on_start()
        self.parent.minerals -= type(self).mineral_cost
        self.parent.gas -= type(self).gas_cost

    @classmethod
    def verify(cls, env):
        super().verify(env)
        if env.minerals < cls.mineral_cost:
            raise ActionError("Insufficient Minerals", mineral_cost, env.minerals)

        if env.gas < cls.gas_cost:
            raise ActionError("Insufficient Vespene Gas", gas_cost, env.gas)


class BuildDrone(Build):
    time_to_complete = 12
    mineral_cost = 50

    def __init__(self, parent):
        self.base = parent.focus
        super().__init__(parent)

    def on_start(self):
        super().on_start()
        self.base.larva -= 1

    def on_complete(self):
        super().on_complete()
        self.base.unassigned_drones += 1
        self.base.needs_attention = True

    @classmethod
    def verify(cls, env):
        super().verify(env)
        used, supply = env.supply
        if used + 1 > supply:
            raise ActionError("Insufficient Supply", used, supply)

        if not isinstance(env.focus, Base):
            raise ActionError("Base Not Selected", type(env.focus))

        if env.focus.larva < 1:
            raise ActionError("Not Enough Larva", env.focus.larva)


class BuildBase(Build):
    time_to_complete = 71
    mineral_cost = 300

    def on_start(self):
        super().on_start()
        if isinstance(self.parent.focus, Base):
            self.parent.focus.unassigned_drones -= 1

        elif isinstance(self.parent.focus, Resource):
            self.parent.focus.drones -= 1

    def on_complete(self):
        super().on_complete()
        self.parent.bases.append(Base(self.parent))

    @classmethod
    def verify(cls, env):
        super().verify(env)
        if isinstance(env.focus, Base):
            n_drones = env.focus.unassigned_drones
        elif isinstance(env.focus, Resource):
            n_drones = env.focus.drones
        else:
            raise ActionError("Selection is not a Base or a Resource")

        if n_drones <= 0:
            raise ActionError("No Available Drones from Selection")


class BuildQueen(Build):
    time_to_complete = 36
    mineral_cost = 150

    def __init__(self, parent):
        self.base = parent.focus
        super().__init__(parent)

    def on_start(self):
        super().on_start()
        self.base.queens_queued += 1

    def on_complete(self):
        super().on_complete()
        self.base.queens += 1
        self.base.queens_queued -= 1

    @classmethod
    def verify(cls, env):
        super().verify(env)
        if not env.spawning_pool:
            raise ActionError("Building a Queen Requires a Spawning Pool")

        used, supply = env.supply
        if used + 2 > supply:
            raise ActionError("Insufficient Supply", used, supply)

        if not isinstance(env.focus, Base):
            raise ActionError("Base Not Selected", type(env.focus))

        if env.focus.queens_queued >= 5:
            raise ActionError("No More Queens can Be Queued", env.focus.queens_queued)


class BuildOverlord(Build):
    time_to_complete = 18
    mineral_cost = 100

    def __init__(self, parent):
        self.base = parent.focus
        super().__init__(parent)

    def on_start(self):
        super().on_start()
        self.base.larva -= 1

    def on_complete(self):
        super().on_complete()
        self.parent.overlords += 1

    @classmethod
    def verify(cls, env):
        super().verify(env)
        if not isinstance(env.focus, Base):
            raise ActionError("Base Not Selected", type(env.focus))

        if env.focus.larva < 1:
            raise ActionError("Not Enough Larva", env.focus.larva)


class BuildSpawningPool(Build):
    time_to_complete = 46
    mineral_cost = 200

    def on_start(self):
        super().on_start()
        if isinstance(self.parent.focus, Base):
            self.parent.focus.unassigned_drones -= 1

        elif isinstance(self.parent.focus, Resource):
            self.parent.focus.drones -= 1

    def on_complete(self):
        super().on_complete()
        self.parent.spawning_pool = True

    @classmethod
    def verify(cls, env):
        super().verify(env)
        if isinstance(env.focus, Base):
            n_drones = env.focus.unassigned_drones
        elif isinstance(env.focus, Resource):
            n_drones = env.focus.drones
        else:
            raise ActionError("Selection is not a Base or a Resource")

        if n_drones <= 0:
            raise ActionError("No Available Drones from Selection")


class BuildExtractor(Build):
    time_to_complete = 21
    mineral_cost = 25

    def __init__(self, parent):
        self.geyser = parent.target
        super().__init__(parent)

    def on_start(self):
        super().on_start()
        if isinstance(self.parent.focus, Base):
            self.parent.focus.unassigned_drones -= 1

        elif isinstance(self.parent.focus, Resource):
            self.parent.focus.drones -= 1

    def on_complete(self):
        super().on_complete()
        self.geyser.has_extractor = True

    @classmethod
    def verify(cls, env):
        super().verify(env)
        if not isinstance(env.target, Geyser):
            raise ActionError("Secondary Selection Not a Geyser")
        if env.target.has_extractor:
            raise ActionError("Geyser Already has an Extractor")

        if isinstance(env.focus, Base):
            n_drones = env.focus.unassigned_drones
        elif isinstance(env.focus, Resource):
            n_drones = env.focus.drones
        else:
            raise ActionError("Selection is not a Base or a Resource")

        if n_drones <= 0:
            raise ActionError("No Available Drones from Selection")


class Select(Action):
    def __init__(self, parent, thing):
        self.thing = thing
        super().__init__(parent)

    def on_complete(self):
        super().on_complete()
        self.parent.focus = self.thing


class Target(Action):
    def __init__(self, parent, thing):
        self.thing = thing
        super().__init__(parent)

    def on_complete(self):
        super().on_complete()
        self.parent.target = self.thing


class SetRallyMinerals(Action):
    def on_complete(self):
        super().on_complete()
        self.parent.focus.rally_set = True

    @classmethod
    def verify(cls, env):
        super().verify(env)
        if not isinstance(env.focus, Base):
            raise ActionError("Base Not Selected", type(env.focus))


class TransferDrone(Action):
    time_to_complete = 5

    def __init__(self, parent):
        self.home = parent.target
        super().__init__(parent)

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

    @classmethod
    def verify(cls, env):
        super().verify(env)
        if isinstance(env.focus, Base):
            n_drones = env.focus.unassigned_drones
        elif isinstance(env.focus, Resource):
            n_drones = env.focus.drones
        else:
            raise ActionError("Selection is not a Base or a Resource")

        if n_drones <= 0:
            raise ActionError("No Available Drones from Selection")


class InjectLarva(Action):
    time_to_complete = 29

    def __init__(self, parent):
        self.base = self.parent.focus
        super().__init__(parent)

    def on_complete(self):
        super().on_complete()
        self.base.larva = min(19, self.base.larva + 3)

    @classmethod
    def verify(cls, env):
        super().verify(env)
        if not isinstance(env.focus, Base):
            raise ActionError("Base Not Selected", type(env.focus))

        if env.focus.queens <= 0:
            return ActionError("Queen Required to Inject Larva")


actions = [
    BuildDrone,
    BuildBase,
    BuildQueen,
    BuildOverlord,
    BuildSpawningPool,
    BuildExtractor,
    Select,
    Target,
    SetRallyMinerals,
    TransferDrone,
    InjectLarva,
    NoOp
]
