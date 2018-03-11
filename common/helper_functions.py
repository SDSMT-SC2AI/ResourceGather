from enum import Enum
from math import sqrt

class Alliance(Enum):
    """Enumerator for alliance"""
    Self = 1
    Ally = 2
    Neutral = 3
    Enemy = 4



def GetUnits(unit_type, obs, alliance=Alliance.Self):
    """Gets all units of unit_type(s), unit_type MUST be a list"""
    ret_val = []
    if (isinstance(unit_type, int)):
        for item in obs[0].observation.raw_data.units:
            if (item.alliance == alliance.value and item.unit_type == unit_type):
                ret_val.append(item)
    else:
        for item in obs[0].observation.raw_data.units:
            if (item.alliance == alliance.value and item.unit_type in unit_type):
                ret_val.append(item)
    return ret_val


def DistSquared(a, b):
    """Distance squared algorithm, requires Point{x: y: z:}"""
    return (a.x - b.x )**2 + (a.y - b.y)**2 + (a.z - b.z)**2


def InDistSqRange(a, b, range):
    """Checks to see if a is in range of b"""
    return DistSquared(a, b) <= range

def InRadius(a, b, radius):
    """Checks to see if a is in range of b"""
    return sqrt(DistSquared(a, b)) <= radius