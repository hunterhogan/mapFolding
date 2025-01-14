from mapFolding import Z0Z_outfitFoldings, indexTrack, indexMy, indexThe, Z0Z_computationState
from typing import List, Callable, Any, Final, Optional, Union, Sequence, Tuple
import numpy
import numba

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
