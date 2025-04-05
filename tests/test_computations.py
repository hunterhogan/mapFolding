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

from mapFolding import countFolds, getFoldsTotalKnown, oeisIDfor_n, validateListDimensions
from mapFolding.oeis import settingsOEIS
from mapFolding.someAssemblyRequired.transformationTools import makeInitializedComputationState
from pathlib import Path, PurePosixPath
from tests.conftest import standardizedEqualToCallableReturn, registrarRecordsTmpObject
import importlib.util
import multiprocessing
import pytest

if __name__ == '__main__':
	multiprocessing.set_start_method('spawn')

def test_algorithmSourceParallel(listDimensionsTestParallelization: list[int], useAlgorithmSourceDispatcher: None) -> None:
	standardizedEqualToCallableReturn(getFoldsTotalKnown(tuple(listDimensionsTestParallelization)), countFolds, listDimensionsTestParallelization, None, 'maximum', None)

def test_algorithmSourceSequential(listDimensionsTestCountFolds: tuple[int, ...], useAlgorithmSourceDispatcher: None) -> None:
	standardizedEqualToCallableReturn(getFoldsTotalKnown(tuple(listDimensionsTestCountFolds)), countFolds, listDimensionsTestCountFolds)

def test_aOFn_calculate_value(oeisID: str) -> None:
	for n in settingsOEIS[oeisID]['valuesTestValidation']:
		standardizedEqualToCallableReturn(settingsOEIS[oeisID]['valuesKnown'][n], oeisIDfor_n, oeisID, n)

def test_syntheticParallel(syntheticDispatcherFixture: None, listDimensionsTestParallelization: list[int]) -> None:
	standardizedEqualToCallableReturn(getFoldsTotalKnown(tuple(listDimensionsTestParallelization)), countFolds, listDimensionsTestParallelization, None, 'maximum')

def test_syntheticSequential(syntheticDispatcherFixture: None, listDimensionsTestCountFolds: list[int]) -> None:
	standardizedEqualToCallableReturn(getFoldsTotalKnown(tuple(listDimensionsTestCountFolds)), countFolds, listDimensionsTestCountFolds)

@pytest.mark.parametrize('pathFilenameTmpTesting', ['.py'], indirect=True)
def test_writeJobNumba(oneTestCuzTestsOverwritingTests: list[int], pathFilenameTmpTesting: Path) -> None:
	from mapFolding.someAssemblyRequired.toolboxNumba import RecipeJob, SpicesJobNumba
	from mapFolding.someAssemblyRequired.synthesizeNumbaJob import makeJobNumba
	mapShape = validateListDimensions(oneTestCuzTestsOverwritingTests)
	state = makeInitializedComputationState(mapShape)

	pathFilenameModule = pathFilenameTmpTesting.absolute()
	pathFilenameFoldsTotal = pathFilenameModule.with_suffix('.foldsTotalTesting')
	registrarRecordsTmpObject(pathFilenameFoldsTotal)

	jobTest = RecipeJob(state
						, pathModule=PurePosixPath(pathFilenameModule.parent)
						, moduleIdentifier=pathFilenameModule.stem
						, pathFilenameFoldsTotal=PurePosixPath(pathFilenameFoldsTotal))
	spices = SpicesJobNumba()
	makeJobNumba(jobTest, spices)

	Don_Lapre_Road_to_Self_Improvement = importlib.util.spec_from_file_location("__main__", pathFilenameModule)
	if Don_Lapre_Road_to_Self_Improvement is None:
		raise ImportError(f"Failed to create module specification from {pathFilenameModule}")
	if Don_Lapre_Road_to_Self_Improvement.loader is None:
		raise ImportError(f"Failed to get loader for module {pathFilenameModule}")
	module = importlib.util.module_from_spec(Don_Lapre_Road_to_Self_Improvement)

	module.__name__ = "__main__"
	Don_Lapre_Road_to_Self_Improvement.loader.exec_module(module)

	standardizedEqualToCallableReturn(str(getFoldsTotalKnown(mapShape)), pathFilenameFoldsTotal.read_text().strip)
