from mapFolding import getPathFilenameFoldsTotal, dtypeNumpyDefaults, thisSeemsVeryComplicated
from mapFolding import make_dtype, datatypeLarge, dtypeLarge, datatypeMedium, dtypeMedium, datatypeSmall, dtypeSmall
from mapFolding import outfitCountFolds
from someAssemblyRequired import countInitialize, countSequential
from typing import Any, Optional, Sequence, Type
import more_itertools
import numpy
import pathlib
import pickle
import python_minifier

identifierCallableLaunch = "goGoGadgetAbsurdity"

def makeStateJob(listDimensions: Sequence[int], **keywordArguments: Optional[Type[Any]]):
    stateUniversal = outfitCountFolds(listDimensions, computationDivisions=None, CPUlimit=None, **keywordArguments)
    countInitialize(stateUniversal['connectionGraph'], stateUniversal['gapsWhere'], stateUniversal['my'], stateUniversal['track'])
    pathFilenameChopChop = getPathFilenameFoldsTotal(stateUniversal['mapShape'])
    suffix = pathFilenameChopChop.suffix
    pathJob = pathlib.Path(str(pathFilenameChopChop)[0:-len(suffix)])
    pathJob.mkdir(parents=True, exist_ok=True)
    pathFilenameJob = pathJob / 'stateJob.pkl'

    pathFilenameFoldsTotal = getPathFilenameFoldsTotal(stateUniversal['mapShape'], pathFilenameJob.parent)
    stateJob = {**stateUniversal, 'pathFilenameFoldsTotal': pathFilenameFoldsTotal}

    del stateJob['mapShape']

    pathFilenameJob.write_bytes(pickle.dumps(stateJob))
    return pathFilenameJob

def convertNDArrayToStr(arrayTarget: numpy.ndarray, identifierName: str) -> str:
    def process_nested_array(arraySlice):
        if isinstance(arraySlice, numpy.ndarray) and arraySlice.ndim > 1:
            return [process_nested_array(arraySlice[index]) for index in range(arraySlice.shape[0])]
        elif isinstance(arraySlice, numpy.ndarray) and arraySlice.ndim == 1:
            listWithRanges = []
            for group in more_itertools.consecutive_groups(arraySlice.tolist()):
                ImaSerious = list(group)
                if len(ImaSerious) <= 4:
                    listWithRanges += ImaSerious
                else:
                    ImaRange = [range(ImaSerious[0], ImaSerious[-1] + 1)]
                    listWithRanges += ImaRange
            return listWithRanges
        return arraySlice

    arrayAsNestedLists = process_nested_array(arrayTarget)

    stringMinimized = python_minifier.minify(str(arrayAsNestedLists))
    commaZeroMaximum = arrayTarget.shape[-1] - 1
    stringMinimized = stringMinimized.replace('[0' + ',0'*commaZeroMaximum + ']', '[0]*'+str(commaZeroMaximum+1))
    for countZeros in range(commaZeroMaximum, 2, -1):
        stringMinimized = stringMinimized.replace(',0'*countZeros + ']', ']+[0]*'+str(countZeros))

    stringMinimized = stringMinimized.replace('range', '*range')

    return f"{identifierName} = numpy.array({stringMinimized}, dtype=numpy.{arrayTarget.dtype})"

def writeModuleWithNumba(listDimensions, **keywordArguments: Optional[str]) -> pathlib.Path:
    datatypeLargeAsStr = keywordArguments.get('datatypeLarge', thisSeemsVeryComplicated.datatypeLarge)
    datatypeMediumAsStr = keywordArguments.get('datatypeMedium', thisSeemsVeryComplicated.datatypeMedium)
    datatypeSmallAsStr = keywordArguments.get('datatypeSmall', thisSeemsVeryComplicated.datatypeSmall)

    numpy_dtypeLarge = make_dtype(datatypeLargeAsStr) # type: ignore
    numpy_dtypeMedium = make_dtype(datatypeMediumAsStr) # type: ignore
    numpy_dtypeSmall = make_dtype(datatypeSmallAsStr) # type: ignore

    pathFilenameData = makeStateJob(listDimensions, dtypeLarge = numpy_dtypeLarge, dtypeMedium = numpy_dtypeMedium, dtypeSmall = numpy_dtypeSmall)

    # forceinline=True might actually be useful
    parametersNumba = f"numba.types.{datatypeLargeAsStr}(), \
cache=True, \
nopython=True, \
fastmath=True, \
forceinline=True, \
inline='always', \
looplift=False, \
_nrt=True, \
error_model='numpy', \
parallel=False, \
boundscheck=False, \
no_cfunc_wrapper=False, \
no_cpython_wrapper=False, \
"
# no_cfunc_wrapper=True, \
# no_cpython_wrapper=True, \

    pathFilenameAlgorithm = pathlib.Path('/apps/mapFolding/mapFolding/someAssemblyRequired/countSequentialNoNumba.py')
    pathFilenameDestination = pathFilenameData.with_stem(pathFilenameData.parent.name).with_suffix(".py")

    lineNumba = f"@numba.jit({parametersNumba})"

    linesImport = "\n".join([
                        "import numpy"
                        , "import numba"
                        ])

    stateJob = pickle.loads(pathFilenameData.read_bytes())

    ImaIndent = '    '
    linesDataDynamic = """"""
    linesDataDynamic = "\n".join([linesDataDynamic
            , ImaIndent + f"foldsTotal = numba.types.{datatypeLargeAsStr}(0)"
            , ImaIndent + convertNDArrayToStr(stateJob['my'], 'my')
            , ImaIndent + convertNDArrayToStr(stateJob['foldGroups'], 'foldGroups')
            , ImaIndent + convertNDArrayToStr(stateJob['gapsWhere'], 'gapsWhere')
            , ImaIndent + convertNDArrayToStr(stateJob['track'], 'track')
            ])

    linesDataStatic = """"""
    linesDataStatic = "\n".join([linesDataStatic
            , ImaIndent + convertNDArrayToStr(stateJob['connectionGraph'], 'connectionGraph')
            ])

    pathFilenameFoldsTotal: pathlib.Path = stateJob['pathFilenameFoldsTotal']

    linesAlgorithm = """"""
    for lineSource in pathFilenameAlgorithm.read_text().splitlines():
        if lineSource.startswith('#'):
            continue
        elif not lineSource:
            continue
        elif lineSource.startswith('def '):
            lineSource = "\n".join([lineNumba
                                , f"def {identifierCallableLaunch}():"
                                , linesDataDynamic
                                , linesDataStatic
                                ])
        linesAlgorithm = "\n".join([linesAlgorithm
                            , lineSource
                            ])

    linesLaunch = """"""
    linesLaunch = linesLaunch + f"""
if __name__ == '__main__':
    import time
    timeStart = time.perf_counter()
    {identifierCallableLaunch}()
    print(time.perf_counter() - timeStart)"""

    linesWriteFoldsTotal = """"""
    linesWriteFoldsTotal = "\n".join([linesWriteFoldsTotal
                                    , "    foldsTotal = foldGroups[0:-1].sum() * foldGroups[-1]"
                                    , "    print(foldsTotal)"
                                    , "    with numba.objmode():"
                                    , f"        open('{pathFilenameFoldsTotal.as_posix()}', 'w').write(str(foldsTotal))"
                                    , "    return foldsTotal"
                                    ])

    linesAll = "\n".join([
                        linesImport
                        , linesAlgorithm
                        , linesWriteFoldsTotal
                        , linesLaunch
                        ])

    pathFilenameDestination.write_text(linesAll)

    return pathFilenameDestination

if __name__ == '__main__':
    listDimensions = [6,6]
    datatypeLarge = 'int64'
    datatypeMedium = 'uint8'
    datatypeSmall = datatypeMedium
    writeModuleWithNumba(listDimensions, datatypeLarge=datatypeLarge, datatypeMedium=datatypeMedium, datatypeSmall=datatypeSmall)
