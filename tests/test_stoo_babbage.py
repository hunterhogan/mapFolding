import pytest
import jax
import numpy
from typing import get_args, Dict, Any
from unittest.mock import MagicMock

from mapFolding.oeis import settingsOEISsequences, OEISsequenceID
from mapFolding.lovelaceIndices import leavesTotal as indexLeavesTotal, dimensionsTotal as indexDimensionsTotal

from mapFolding.babbage import foldings as foldingsBabbage
from mapFolding.stoo import foldings as foldingsStoo

@pytest.mark.parametrize("oeisID", get_args(OEISsequenceID))
def test_intermediate_arrays(oeisID: OEISsequenceID, monkeypatch):
    """Test that stoo and babbage create equivalent intermediate arrays.
    
    This test ensures that both implementations:
    1. Generate the same D matrix
    2. Calculate same leavesTotal and dimensionsTotal
    """
    
    parameters_babbage: Dict[str, Any] = {}
    parameters_stoo: Dict[str, Any] = {}
    
    def mock_clientOfBabbage(track: Any, gap: Any, the: Any, D: Any) -> None:
        parameters_babbage.update({
            'track': track,
            'gap': gap,
            'the': the,
            'D': D
        })
    
    def mock_clientOfStoo(taskDivisions: Any, arrayIndicesTask: Any, 
                         leavesTotal: Any, dimensionsTotal: Any, D: Any) -> None:
        parameters_stoo.update({
            'taskDivisions': taskDivisions,
            'arrayIndicesTask': arrayIndicesTask,
            'leavesTotal': leavesTotal,
            'dimensionsTotal': dimensionsTotal,
            'D': D
        })
    
    monkeypatch.setattr('mapFolding.lovelace.countFoldings', mock_clientOfBabbage)
    monkeypatch.setattr('mapFolding.pid.spoon', mock_clientOfStoo)

    for n in range(2, settingsOEISsequences[oeisID]['benchmarkValues'][-1]):
        listDimensions = settingsOEISsequences[oeisID]['dimensions'](n)

        # Clear parameters for each iteration
        parameters_babbage.clear()
        parameters_stoo.clear()

        foldingsBabbage(listDimensions)
        foldingsStoo(listDimensions)

        # Compare D matrices
        Dbabbage = parameters_babbage['D']
        Dstoo = parameters_stoo['D']
        assert jax.numpy.array_equal(jax.numpy.array(Dbabbage), Dstoo)
        assert numpy.array_equal(Dbabbage, numpy.array(Dstoo))

        # Compare leaf and dimension totals
        assert parameters_babbage['the'][indexLeavesTotal] == parameters_stoo['leavesTotal']
        assert parameters_babbage['the'][indexDimensionsTotal] == parameters_stoo['dimensionsTotal']
