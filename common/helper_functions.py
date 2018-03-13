from enum import Enum
from math import sqrt
import tensorflow as tf


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


# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder