from .benchmarking import recordBenchmarks

__all__ = ['recordBenchmarks', 'enableBenchmarks', 'disableBenchmarks']

_measureSpeed = False

def enableBenchmarks():
    """Enable benchmarking measurements."""
    global _measureSpeed
    _measureSpeed = True

def disableBenchmarks():
    """Disable benchmarking measurements."""
    global _measureSpeed
    _measureSpeed = False
