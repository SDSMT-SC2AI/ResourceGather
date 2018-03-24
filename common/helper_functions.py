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


# Performs bisection on a single variable function f using tensorflow
def bisection(f, x=0.0, epsilon=1e-4):
    xinit = tf.constant(x, name="x_init", dtype=tf.float32)
    eps = tf.constant(epsilon, name="epsilon", dtype=tf.float32)
    greater = lambda x, xmin, xmax: f(xmin) > 0
    lesser = lambda x, xmin, xmax: f(xmax) < 0
    not_converged = lambda x, xmin, xmax: xmax - xmin > eps

    def get_xmin(x, xmin, xmax):
        return [xmin, xmin - tf.abs(xmin/2) - 1, xmin]

    def get_xmax(x, xmin, xmax):
        return [xmax, xmax, xmax + tf.abs(xmax/2) + 1]

    def get_x(x, xmin, xmax):
        return tf.cond(tf.less(0.0, f(x)),
                       lambda: [(x + xmin) / 2, xmin, x],  # too big
                       lambda: [(x + xmax) / 2, x, xmax])  # too small

    x, xmin, xmax = tf.while_loop(greater, get_xmin, [xinit, xinit, xinit], back_prop=False, name="get_xmin")
    x, xmin, xmax = tf.while_loop(lesser, get_xmax, [x, xmin, xmax], back_prop=False, name="get_xmax")
    x, xmin, xmax = tf.while_loop(not_converged, get_x, [x, xmin, xmax], back_prop=False, name="get_x")
    return x


# Performs discounting for the input tensor along the first dimension
def discount(x, gamma, init=0.0):
    gamma = tf.constant(gamma)
    return tf.reverse(tf.scan(lambda prev, curr: gamma*prev + curr, tf.reverse(x, axis=[0]), init), axis=[0])


# For each element x[i], select x[i][indices[i]]
def select_from(x, indices):
    dtype = indices.dtype
    idx = tf.stack([tf.range(tf.shape(indices, out_type=dtype)[0], dtype=dtype), indices], axis=1)
    return tf.gather_nd(x, idx)



