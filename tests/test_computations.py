"""
Core Algorithm and Module Generation Testing

This module provides tests for validating algorithm correctness and testing
code generation functionality. It's designed not only to test the package's
functionality but also to serve as a template for users testing their own
custom implementations.

## Key Testing Categories

1. Algorithm Validation Tests
   - `test_algorithmSourceParallel` - Tests the source algorithm in parallel mode
   - `test_algorithmSourceSequential` - Tests the source algorithm in sequential mode
   - `test_aOFn_calculate_value` - Tests OEIS sequence value calculations

2. Synthetic Module Tests
   - `test_syntheticParallel` - Tests generated Numba-optimized code in parallel mode
   - `test_syntheticSequential` - Tests generated Numba-optimized code in sequential mode

3. Job Testing
   - `test_writeJobNumba` - Tests job-specific module generation and execution

## How to Test Your Custom Implementations

### Testing Custom Recipes (RecipeSynthesizeFlow):

1. Copy the `syntheticDispatcherFixture` from conftest.py
2. Modify it to use your custom recipe configuration
3. Copy and adapt `test_syntheticParallel` and `test_syntheticSequential`

Example:

```python
@pytest.fixture
def myCustomRecipeFixture(useThisDispatcher, pathTmpTesting):
    # Create your custom recipe configuration
    myRecipe = RecipeSynthesizeFlow(
        pathPackage=PurePosixPath(pathTmpTesting.absolute()),
        # Add your custom configuration
    )

    # Generate the module
    makeNumbaFlow(myRecipe)

    # Import and patch the dispatcher
    # ... (similar to syntheticDispatcherFixture)

    return customDispatcher

def test_myCustomRecipeParallel(myCustomRecipeFixture, listDimensionsTestParallelization):
    # Test with the standardized validation utility
    standardizedEqualToCallableReturn(
        getFoldsTotalKnown(tuple(listDimensionsTestParallelization)),
        countFolds,
        listDimensionsTestParallelization,
        None,
        'maximum'
    )
```

### Testing Custom Jobs (RecipeJob):

1. Copy and adapt `test_writeJobNumba`
2. Modify it to use your custom job configuration

Example:

```python
def test_myCustomJob(oneTestCuzTestsOverwritingTests, pathFilenameTmpTesting):
    # Create your custom job configuration
    myJob = RecipeJob(
        state=makeInitializedComputationState(validateListDimensions(oneTestCuzTestsOverwritingTests)),
        # Add your custom configuration
    )

    spices = SpicesJobNumba()
    # Customize spices if needed

    # Generate and test the job
    makeJobNumba(myJob, spices)
    # Test execution similar to test_writeJobNumba
```

All tests leverage standardized utilities like `standardizedEqualToCallableReturn`
that provide consistent, informative error messages and simplify test validation.
"""

from mapFolding import countFolds, getFoldsTotalKnown, oeisIDfor_n
from mapFolding.dataBaskets import MapFoldingState
from mapFolding.oeis import settingsOEIS
from mapFolding.someAssemblyRequired.RecipeJob import RecipeJobTheorem2Numba
from mapFolding.syntheticModules.initializeCount import initializeGroupsOfFolds
from pathlib import Path, PurePosixPath
from tests.conftest import standardizedEqualToCallableReturn, registrarRecordsTmpObject
from typing import Literal
import importlib.util
import multiprocessing
import pytest

if __name__ == '__main__':
	multiprocessing.set_start_method('spawn')

# TODO test synthesis

@pytest.mark.parametrize('flow', ['daoOfMapFolding', 'theorem2', 'theorem2Trimmed', 'theorem2numba'])
def test_flowControl(mapShapeTestCountFolds: tuple[int, ...], flow: Literal['daoOfMapFolding'] | Literal['theorem2'] | Literal['theorem2numba']) -> None:
	standardizedEqualToCallableReturn(getFoldsTotalKnown(mapShapeTestCountFolds), countFolds, None, None, None, None, mapShapeTestCountFolds, None, None, flow)

def test_aOFn_calculate_value(oeisID: str) -> None:
	for n in settingsOEIS[oeisID]['valuesTestValidation']:
		standardizedEqualToCallableReturn(settingsOEIS[oeisID]['valuesKnown'][n], oeisIDfor_n, oeisID, n)

@pytest.mark.parametrize('pathFilenameTmpTesting', ['.py'], indirect=True)
def test_writeJobNumba(oneTestCuzTestsOverwritingTests: tuple[int, ...], pathFilenameTmpTesting: Path) -> None:
	from mapFolding.someAssemblyRequired.toolkitNumba import SpicesJobNumba
	from mapFolding.someAssemblyRequired.makeJobTheorem2Numba import makeJobNumba
	mapShape = oneTestCuzTestsOverwritingTests
	state = MapFoldingState(mapShape)
	state = initializeGroupsOfFolds(state)

	pathFilenameModule = pathFilenameTmpTesting.absolute()
	pathFilenameFoldsTotal = pathFilenameModule.with_suffix('.foldsTotalTesting')
	registrarRecordsTmpObject(pathFilenameFoldsTotal)

	jobTest = RecipeJobTheorem2Numba(state
						, pathModule=PurePosixPath(pathFilenameModule.parent)
						, moduleIdentifier=pathFilenameModule.stem
						, pathFilenameFoldsTotal=PurePosixPath(pathFilenameFoldsTotal))
	spices = SpicesJobNumba(useNumbaProgressBar=False)
	makeJobNumba(jobTest, spices)

	Don_Lapre_Road_to_Self_Improvement = importlib.util.spec_from_file_location("__main__", pathFilenameModule)
	if Don_Lapre_Road_to_Self_Improvement is None:
		raise ImportError(f"Failed to create module specification from {pathFilenameModule}")
	if Don_Lapre_Road_to_Self_Improvement.loader is None:
		raise ImportError(f"Failed to get loader for module {pathFilenameModule}")
	module = importlib.util.module_from_spec(Don_Lapre_Road_to_Self_Improvement)

	module.__name__ = "__main__"
	Don_Lapre_Road_to_Self_Improvement.loader.exec_module(module)

	standardizedEqualToCallableReturn(str(getFoldsTotalKnown(oneTestCuzTestsOverwritingTests)), pathFilenameFoldsTotal.read_text().strip)
