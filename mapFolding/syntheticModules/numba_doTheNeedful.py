from  import 
from mapFolding.syntheticModules.numbaCount import countInitialize, countParallel, countSequential
from mapFolding.theSSOT import ComputationState
import copy

@(ComputationState(ComputationState), _nrt=True, boundscheck=False, cache=True, error_model='numpy', fastmath=True, forceinline=True, inline='always', looplift=False, no_cfunc_wrapper=False, no_cpython_wrapper=False, nopython=True, parallel=False)
def doTheNeedful(computationStateInitialized: ComputationState) -> ComputationState:
    computationStateInitialized = countInitialize(computationStateInitialized)
    if computationStateInitialized.taskDivisions > 0:
        return countParallel(computationStateInitialized)
    else:
        return countSequential(computationStateInitialized)