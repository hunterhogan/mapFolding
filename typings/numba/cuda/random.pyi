from _typeshed import Incomplete
from numba import config as config, cuda as cuda, float32 as float32, float64 as float64, from_dtype as from_dtype, int64 as int64, jit as jit, uint32 as uint32, uint64 as uint64

xoroshiro128p_dtype: Incomplete
xoroshiro128p_type: Incomplete
_forceobj: Incomplete
_looplift: Incomplete
_nopython: Incomplete

def init_xoroshiro128p_state(states, index, seed) -> None:
    """Use SplitMix64 to generate an xoroshiro128p state from 64-bit seed.

    This ensures that manually set small seeds don't result in a predictable
    initial sequence from the random number generator.

    :type states: 1D array, dtype=xoroshiro128p_dtype
    :param states: array of RNG states
    :type index: uint64
    :param index: offset in states to update
    :type seed: int64
    :param seed: seed value to use when initializing state
    """
def rotl(x, k):
    """Left rotate x by k bits."""
def xoroshiro128p_next(states, index):
    """Return the next random uint64 and advance the RNG in states[index].

    :type states: 1D array, dtype=xoroshiro128p_dtype
    :param states: array of RNG states
    :type index: int64
    :param index: offset in states to update
    :rtype: uint64
    """
def xoroshiro128p_jump(states, index) -> None:
    """Advance the RNG in ``states[index]`` by 2**64 steps.

    :type states: 1D array, dtype=xoroshiro128p_dtype
    :param states: array of RNG states
    :type index: int64
    :param index: offset in states to update
    """
def uint64_to_unit_float64(x):
    """Convert uint64 to float64 value in the range [0.0, 1.0)"""
def uint64_to_unit_float32(x):
    """Convert uint64 to float32 value in the range [0.0, 1.0)"""
def xoroshiro128p_uniform_float32(states, index):
    """Return a float32 in range [0.0, 1.0) and advance ``states[index]``.

    :type states: 1D array, dtype=xoroshiro128p_dtype
    :param states: array of RNG states
    :type index: int64
    :param index: offset in states to update
    :rtype: float32
    """
def xoroshiro128p_uniform_float64(states, index):
    """Return a float64 in range [0.0, 1.0) and advance ``states[index]``.

    :type states: 1D array, dtype=xoroshiro128p_dtype
    :param states: array of RNG states
    :type index: int64
    :param index: offset in states to update
    :rtype: float64
    """

TWO_PI_FLOAT32: Incomplete
TWO_PI_FLOAT64: Incomplete

def xoroshiro128p_normal_float32(states, index):
    """Return a normally distributed float32 and advance ``states[index]``.

    The return value is drawn from a Gaussian of mean=0 and sigma=1 using the
    Box-Muller transform.  This advances the RNG sequence by two steps.

    :type states: 1D array, dtype=xoroshiro128p_dtype
    :param states: array of RNG states
    :type index: int64
    :param index: offset in states to update
    :rtype: float32
    """
def xoroshiro128p_normal_float64(states, index):
    """Return a normally distributed float32 and advance ``states[index]``.

    The return value is drawn from a Gaussian of mean=0 and sigma=1 using the
    Box-Muller transform.  This advances the RNG sequence by two steps.

    :type states: 1D array, dtype=xoroshiro128p_dtype
    :param states: array of RNG states
    :type index: int64
    :param index: offset in states to update
    :rtype: float64
    """
def init_xoroshiro128p_states_cpu(states, seed, subsequence_start) -> None: ...
def init_xoroshiro128p_states(states, seed, subsequence_start: int = 0, stream: int = 0) -> None:
    """Initialize RNG states on the GPU for parallel generators.

    This initializes the RNG states so that each state in the array corresponds
    subsequences in the separated by 2**64 steps from each other in the main
    sequence.  Therefore, as long no CUDA thread requests more than 2**64
    random numbers, all of the RNG states produced by this function are
    guaranteed to be independent.

    The subsequence_start parameter can be used to advance the first RNG state
    by a multiple of 2**64 steps.

    :type states: 1D DeviceNDArray, dtype=xoroshiro128p_dtype
    :param states: array of RNG states
    :type seed: uint64
    :param seed: starting seed for list of generators
    """
def create_xoroshiro128p_states(n, seed, subsequence_start: int = 0, stream: int = 0):
    """Returns a new device array initialized for n random number generators.

    This initializes the RNG states so that each state in the array corresponds
    subsequences in the separated by 2**64 steps from each other in the main
    sequence.  Therefore, as long no CUDA thread requests more than 2**64
    random numbers, all of the RNG states produced by this function are
    guaranteed to be independent.

    The subsequence_start parameter can be used to advance the first RNG state
    by a multiple of 2**64 steps.

    :type n: int
    :param n: number of RNG states to create
    :type seed: uint64
    :param seed: starting seed for list of generators
    :type subsequence_start: uint64
    :param subsequence_start:
    :type stream: CUDA stream
    :param stream: stream to run initialization kernel on
    """
