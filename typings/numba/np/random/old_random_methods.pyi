from numba import uint16 as uint16, uint32 as uint32, uint64 as uint64, uint8 as uint8
from numba.core.extending import register_jitable as register_jitable
from numba.np.random._constants import UINT16_MAX as UINT16_MAX, UINT32_MAX as UINT32_MAX, UINT64_MAX as UINT64_MAX, UINT8_MAX as UINT8_MAX
from numba.np.random.generator_core import next_uint32 as next_uint32, next_uint64 as next_uint64

def gen_mask(max): ...
def buffered_bounded_bool(bitgen, off, rng, bcnt, buf): ...
def buffered_uint8(bitgen, bcnt, buf): ...
def buffered_uint16(bitgen, bcnt, buf): ...
def buffered_bounded_lemire_uint8(bitgen, rng, bcnt, buf):
    """
    Generates a random unsigned 8 bit integer bounded
    within a given interval using Lemire's rejection.

    The buffer acts as storage for a 32 bit integer
    drawn from the associated BitGenerator so that
    multiple integers of smaller bitsize can be generated
    from a single draw of the BitGenerator.
    """
def buffered_bounded_lemire_uint16(bitgen, rng, bcnt, buf):
    """
    Generates a random unsigned 16 bit integer bounded
    within a given interval using Lemire's rejection.

    The buffer acts as storage for a 32 bit integer
    drawn from the associated BitGenerator so that
    multiple integers of smaller bitsize can be generated
    from a single draw of the BitGenerator.
    """
def buffered_bounded_lemire_uint32(bitgen, rng):
    """
    Generates a random unsigned 32 bit integer bounded
    within a given interval using Lemire's rejection.
    """
def bounded_lemire_uint64(bitgen, rng):
    """
    Generates a random unsigned 64 bit integer bounded
    within a given interval using Lemire's rejection.
    """
def random_bounded_uint64_fill(bitgen, low, rng, size, dtype):
    """
    Returns a new array of given size with 64 bit integers
    bounded by given interval.
    """
def random_bounded_uint32_fill(bitgen, low, rng, size, dtype):
    """
    Returns a new array of given size with 32 bit integers
    bounded by given interval.
    """
def random_bounded_uint16_fill(bitgen, low, rng, size, dtype):
    """
    Returns a new array of given size with 16 bit integers
    bounded by given interval.
    """
def random_bounded_uint8_fill(bitgen, low, rng, size, dtype):
    """
    Returns a new array of given size with 8 bit integers
    bounded by given interval.
    """
def random_bounded_bool_fill(bitgen, low, rng, size, dtype):
    """
    Returns a new array of given size with boolean values.
    """
def _randint_arg_check(low, high, endpoint, lower_bound, upper_bound) -> None:
    """
    Check that low and high are within the bounds
    for the given datatype.
    """
def random_interval(bitgen, max_val): ...
