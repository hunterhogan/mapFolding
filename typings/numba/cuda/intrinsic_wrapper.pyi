from .decorators import jit as jit

def all_sync(mask, predicate):
    """
    If for all threads in the masked warp the predicate is true, then
    a non-zero value is returned, otherwise 0 is returned.
    """
def any_sync(mask, predicate):
    """
    If for any thread in the masked warp the predicate is true, then
    a non-zero value is returned, otherwise 0 is returned.
    """
def eq_sync(mask, predicate):
    """
    If for all threads in the masked warp the boolean predicate is the same,
    then a non-zero value is returned, otherwise 0 is returned.
    """
def ballot_sync(mask, predicate):
    """
    Returns a mask of all threads in the warp whose predicate is true,
    and are within the given mask.
    """
def shfl_sync(mask, value, src_lane):
    """
    Shuffles value across the masked warp and returns the value
    from src_lane. If this is outside the warp, then the
    given value is returned.
    """
def shfl_up_sync(mask, value, delta):
    """
    Shuffles value across the masked warp and returns the value
    from (laneid - delta). If this is outside the warp, then the
    given value is returned.
    """
def shfl_down_sync(mask, value, delta):
    """
    Shuffles value across the masked warp and returns the value
    from (laneid + delta). If this is outside the warp, then the
    given value is returned.
    """
def shfl_xor_sync(mask, value, lane_mask):
    """
    Shuffles value across the masked warp and returns the value
    from (laneid ^ lane_mask).
    """
