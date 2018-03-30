import numpy as np
class FakeSC2Env:
    def __init__(self):
        self.episode_step = 0
        self.supply = 16
        self.bases = 1
        self.units

    def reset(self):
        self.episode_step = 0
        return [[0, self.episode_step, False], ]

    def step(self, actions):
        self.episode_step += 1
        reward = actions[0]**2
        return [[reward, self.episode_step, self.episode_step > 100], ]


class Unit:
    def __init__(self):
        # Position
        self.x = 0
        self.y = 0
        self.z = 0

        # Dynamic Attributes
        self.capacity = 0
        self.health = 0
        self.energy = 0
        self.armor = 0
        self.speed = 0
        self.killed_ticker = 0
        self.saved_ticker = 0

        # Static Attributes
        self.strength = 10
        self.dexterity = 10
        self.constitution = 10
        self.intelligence = 10
        self.wisdom = 10
        self.charisma = 10

        # Modifier Collections
        self.weapons = WeaponSet()
        self.spells = SpellSet()
        self.buffs = Upgrades()


    @staticmethod
    def sufficient_tech(units):
        raise NotImplementedError

