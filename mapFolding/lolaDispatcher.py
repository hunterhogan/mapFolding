from mapFolding.lolaRun import doWhileConcurrent, doWhile
from mapFolding import getTaskDivisions
from typing import Any, Optional, Tuple, Union
import numpy
import numba
import numpy.typing
import pathlib
import os

def saveState(pathState: Union[str, os.PathLike[Any]],
    connectionGraph: numpy.ndarray,
    dimensionsTotal: numpy.uint8,
    leavesTotal: numpy.uint8,
    potentialGaps: numpy.ndarray,
    track: numpy.ndarray):
    pathFilenameState = pathlib.Path(pathState, "stateJob.npz")
    pathFilenameState.parent.mkdir(parents=True, exist_ok=True)
    numpy.savez_compressed(pathFilenameState,
        connectionGraph=connectionGraph,
        dimensionsTotal=dimensionsTotal,
        leavesTotal=leavesTotal,
        potentialGaps=potentialGaps,
        track=track
    )
    return pathFilenameState

def loadState(pathFilenameState: Union[str, os.PathLike[Any]]) -> Tuple:
    tupleLolaState = (
        numpy.load(pathFilenameState)['connectionGraph'],
        numpy.uint8(numpy.load(pathFilenameState)['dimensionsTotal']),
        numpy.uint8(numpy.load(pathFilenameState)['leavesTotal']),
        numpy.load(pathFilenameState)['potentialGaps'],
        numpy.load(pathFilenameState)['track']
    )
    return tupleLolaState

def doJob(pathFilenameState: Union[str, os.PathLike[Any]], computationDivisions: bool = False, CPUlimit: Optional[Union[int, float, bool]] = None):
    taskDivisions = getTaskDivisions(computationDivisions)
    # TODO figure out how to pass state information around. Am I having fun yet?!
    tupleLolaState=loadState(pathFilenameState)
    foldsTotal = lolaDispatcher(taskDivisions, CPUlimit, pathJob=None, tupleLolaState=tupleLolaState)
    print(foldsTotal)
    # TODO SSOT for where I am putting information, ffs!
    filenameFoldsTotal = str(tupleLolaState[2]).replace(' ', '') + ".foldsTotal"
    pathFilenameFoldsTotal = pathlib.Path(pathFilenameState).with_name(filenameFoldsTotal)
    pathFilenameFoldsTotal.write_text(str(foldsTotal))
    return pathFilenameFoldsTotal

@numba.jit(nopython=True, cache=True, fastmath=True)
def lolaDispatcher(computationDivisions, CPUlimit, pathJob, tupleLolaState):
    if pathJob is not None:
        return saveState(pathJob, *tupleLolaState)
    elif computationDivisions:
        # taskDivisions = numpy.uint8(min(numba.get_num_threads(), tupleLolaState[2])) # leavesTotal
        """TODO something is wrong with `taskDivisions = numpy.uint8(min(numba.get_num_threads(), tupleLolaState[2])) # leavesTotal` but setting `taskDivisions = numpy.uint8(tupleLolaState[2]) # leavesTotal` still works as expected. Therefore, while restructuring the entire computationDivisions-to-taskDivisions system, look for the reason(s) why taskDivisions != leavesTotal if leavesTotal < numba.get_num_threads() calculates incorrectly."""
        taskDivisions = numpy.uint8(tupleLolaState[2]) # leavesTotal
        foldsTotal = int(doWhileConcurrent(*tupleLolaState, taskDivisions)) # type: ignore
    else:
        foldsTotal = int(doWhile(*tupleLolaState))
    return foldsTotal
