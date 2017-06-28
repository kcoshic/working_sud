from scipy import *
_empty = empty
def empty(*args, **kwargs):
    kwargs.update(dtype=longdouble)
    _empty(*args, **kwargs)
