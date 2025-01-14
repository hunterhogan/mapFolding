from mapFolding import Z0Z_outfitFoldings, indexTrack, indexMy, indexThe, Z0Z_computationState
from typing import List, Callable, Any, Final, Optional, Union, Sequence, Tuple
import numpy
import numba

"""
from Z0Z_tools import dataTabularTOpathFilenameDelimited
from collections import Counter
from functools import wraps
from types import FrameType
import pathlib
import sys
import time
def traceCalls(functionTarget: Callable[..., Any]) -> Callable[..., Any]:
    def decoratorTrace(functionTarget: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(functionTarget)
        def wrapperTrace(*arguments, **keywordArguments):
            pathLog = pathlib.Path("/apps/mapFolding/Z0Z_notes")
            pathFilenameLog = pathLog / 'functionCalls.tab'
            timeStart = time.perf_counter_ns()
            listTraceCalls = []

            pathFilenameHost = pathlib.Path(__file__).resolve()
            def logCall(frame: FrameType, event: str, arg: object):
                if event == 'call':
                    listTraceCalls.append([time.perf_counter_ns(), frame.f_code.co_name, frame.f_code.co_filename])
                return logCall

            oldTrace = sys.gettrace()
            sys.settrace(logCall)
            try:
                return functionTarget(*arguments, **keywordArguments)
            finally:
                sys.settrace(oldTrace)
                # listTraceCalls = [[timeRelativeNS - timeStart, functionName] for timeRelativeNS, functionName, DISCARDpathFilename in listTraceCalls]
                listTraceCalls = [[timeRelativeNS - timeStart, functionName] for timeRelativeNS, functionName, module in listTraceCalls if  pathlib.Path(module).resolve() == pathFilenameHost]
                print(Counter([functionName for timeRelativeNS, functionName in listTraceCalls]))
                dataTabularTOpathFilenameDelimited(pathFilenameLog, listTraceCalls, ['timeRelativeNS', 'functionName'])

        return wrapperTrace
    return decoratorTrace(functionTarget)

"""
# TODO the current tests expect positional `listDimensions, computationDivisions`, so after restructuring you can arrange the parameters however you want.
def countFolds(listDimensions: Sequence[int], computationDivisions = None, CPUlimit: Optional[Union[int, float, bool]] = None):
    stateUniversal = Z0Z_outfitFoldings(listDimensions, computationDivisions=computationDivisions, CPUlimit=CPUlimit)

    connectionGraph: Final[numpy.ndarray] = stateUniversal['connectionGraph']
    foldsTotal = stateUniversal['foldsTotal']
    mapShape: Final[Tuple] = stateUniversal['mapShape']
    my = stateUniversal['my']
    potentialGaps = stateUniversal['potentialGaps']
    the: Final[numpy.ndarray] = stateUniversal['the']
    track = stateUniversal['track']

    # TODO remove after restructuring
    # connectionGraph, foldsTotal, mapShape, my, potentialGaps, the, track = Z0Z_outfitFoldings(listDimensions, computationDivisions=computationDivisions, CPUlimit=CPUlimit)

    my[indexMy.doCountGaps.value] = int(False)
    my[indexMy.leaf1ndex.value] = 1
    from mapFolding.babbage import _countFolds
    return _countFolds(connectionGraph, foldsTotal, mapShape, my, potentialGaps, the, track)
