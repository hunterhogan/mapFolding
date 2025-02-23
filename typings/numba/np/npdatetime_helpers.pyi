from _typeshed import Incomplete

DATETIME_UNITS: Incomplete
NAT: Incomplete

def same_kind(src, dest):
    """
    Whether the *src* and *dest* units are of the same kind.
    """
def can_cast_timedelta_units(src, dest): ...

_factors: Incomplete

def _get_conversion_multiplier(big_unit_code, small_unit_code):
    """
    Return an integer multiplier allowing to convert from *big_unit_code*
    to *small_unit_code*.
    None is returned if the conversion is not possible through a
    simple integer multiplication.
    """
def get_timedelta_conversion_factor(src_unit, dest_unit):
    """
    Return an integer multiplier allowing to convert from timedeltas
    of *src_unit* to *dest_unit*.
    """
def get_datetime_timedelta_conversion(datetime_unit, timedelta_unit):
    """
    Compute a possible conversion for combining *datetime_unit* and
    *timedelta_unit* (presumably for adding or subtracting).
    Return (result unit, integer datetime multiplier, integer timedelta
    multiplier). RuntimeError is raised if the combination is impossible.
    """
def combine_datetime_timedelta_units(datetime_unit, timedelta_unit):
    """
    Return the unit result of combining *datetime_unit* with *timedelta_unit*
    (e.g. by adding or subtracting).  None is returned if combining
    those units is forbidden.
    """
def get_best_unit(unit_a, unit_b):
    """
    Get the best (i.e. finer-grained) of two units.
    """
def datetime_minimum(a, b) -> None: ...
def datetime_maximum(a, b) -> None: ...
