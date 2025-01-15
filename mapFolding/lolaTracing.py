from mapFolding import Z0Z_outfitFoldings
from typing import Optional, Union, Sequence

# TODO the current tests expect positional `listDimensions, computationDivisions`, so after restructuring you can arrange the parameters however you want.
def countFolds(listDimensions: Sequence[int], computationDivisions: Optional[Union[int, str]] = None, CPUlimit: Optional[Union[int, float, bool]] = None):

    # TODO try different dtypes
    stateUniversal = Z0Z_outfitFoldings(listDimensions, computationDivisions=computationDivisions, CPUlimit=CPUlimit)

    from mapFolding.babbage import _countFolds
    return _countFolds(**stateUniversal)
