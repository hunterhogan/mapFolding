import pytest
import gc
from mapFolding import oeisSequence_aOFn, settingsOEISsequences

@pytest.mark.benchmark(
    group="OEIS",
    max_time=0.5,
    min_rounds=2,
    disable_gc=False,
    warmup=False
)
@pytest.mark.parametrize("oeisID, n", [
    (oeisID, n)
    for oeisID, settings in settingsOEISsequences.items()
    for n in settings['testValuesSpeed']
])
def test_oeis_sequence_benchmark(benchmark, oeisID, n):
    """
    Benchmark test for OEIS sequences using pytest-benchmark.
    """
    def run_benchmark():
        gc.collect()  # Force garbage collection before each run
        return oeisSequence_aOFn(oeisID, n)
    
    foldingsTotal = benchmark(run_benchmark)