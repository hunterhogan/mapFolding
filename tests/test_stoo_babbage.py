import pytest
import jax
import numpy
from typing import get_args, Dict, Any
from unittest.mock import MagicMock

from mapFolding.oeis import settingsOEISsequences
from mapFolding.oeis import OEISsequenceID
from mapFolding.lovelaceIndices import leavesTotal as indexLeavesTotal, dimensionsTotal as indexDimensionsTotal

from mapFolding.babbage import foldings as foldingsBabbage
from mapFolding.stoo import foldings as foldingsStoo
from mapFolding.pid import spoon

@pytest.fixture(params=settingsOEISsequences.keys())
def oeisID(request):
    return request.param

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

# def test_spoon_intermediate_data(oeisID, monkeypatch):
#     """Test spoon function by capturing data before and after 'hubris' is called."""
#     captured_data = {}

#     # We will wrap the original 'spoon' function to capture intermediate data
#     from mapFolding.pid import spoon as original_spoon

#     def mock_spoon(taskDivisions, arrayIndicesTask, leavesTotal, dimensionsTotal, D):
#         # Capture inputs to 'spoon'
#         captured_data['taskDivisions'] = taskDivisions
#         captured_data['arrayIndicesTask'] = arrayIndicesTask
#         captured_data['leavesTotal'] = leavesTotal
#         captured_data['dimensionsTotal'] = dimensionsTotal
#         captured_data['D'] = D

#         # Since 'hubris' is defined inside 'spoon', we cannot directly mock it
#         # Instead, we will monkeypatch 'jax.lax.while_loop' within 'spoon' to capture data

#         original_while_loop = jax.lax.while_loop

#         def mock_while_loop(cond_fun, body_fun, init_val):
#             # Capture data before 'hubris' starts
#             captured_data['hubris_init_val'] = init_val

#             # Run the original while loop
#             result = original_while_loop(cond_fun, body_fun, init_val)

#             # Capture data after 'hubris' finishes
#             captured_data['hubris_output_val'] = result
#             return result

#         # Use monkeypatch to replace 'jax.lax.while_loop' within 'spoon'
#         monkeypatch.setattr(jax.lax, 'while_loop', mock_while_loop)

#         # Call the original 'spoon' function
#         result = original_spoon(taskDivisions, arrayIndicesTask, leavesTotal, dimensionsTotal, D)

#         # Capture the result
#         captured_data['result'] = result
#         return result

#     # Apply the monkeypatch to 'spoon'
#     monkeypatch.setattr('mapFolding.pid.spoon', mock_spoon)

#     for n in settingsOEISsequences[oeisID]['testValuesValidation']:
#         listDimensions = settingsOEISsequences[oeisID]['dimensions'](n)

#         # Clear captured data for each iteration
#         captured_data.clear()

#         # Call 'foldingsStoo' which will call our mocked 'spoon'
#         foldingsStoo(listDimensions)

#         # Verify that data was captured
#         assert 'hubris_init_val' in captured_data
#         assert 'hubris_output_val' in captured_data

#         # Verify intermediate data before 'hubris' starts
#         init_val = captured_data['hubris_init_val']
#         assert init_val['A'].shape == (n + 1,)
#         assert init_val['B'].shape == (n + 1,)
#         assert init_val['foldingsSubtotal'] == 0
#         assert init_val['l'] == 1

#         # Verify intermediate data after 'hubris' finishes
#         output_val = captured_data['hubris_output_val']
#         assert output_val['foldingsSubtotal'] >= 0
#         assert output_val['l'] == 0

#         # Optionally, verify the result
#         result = captured_data['result']
#         # If expected values are known, compare them
#         # expected_result = settingsOEISsequences[oeisID]['valuesKnown'][n]
#         # assert result == expected_result
