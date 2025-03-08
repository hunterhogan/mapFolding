from mapFolding import ComputationState as ComputationState
from mapFolding.syntheticModules.numba_doTheNeedful import doTheNeedful as doTheNeedful
from mapFolding.theSSOT import Array1DElephino as Array1DElephino, Array1DFoldsTotal as Array1DFoldsTotal, Array1DLeavesTotal as Array1DLeavesTotal, Array3D as Array3D, DatatypeElephino as DatatypeElephino, DatatypeFoldsTotal as DatatypeFoldsTotal, DatatypeLeavesTotal as DatatypeLeavesTotal

def flattenData(state: ComputationState) -> ComputationState: ...
