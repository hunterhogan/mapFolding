from mapFolding import getPathFilenameFoldsTotal, indexMy, indexTrack
from mapFolding import make_dtype, datatypeLargeDEFAULT, datatypeMediumDEFAULT, datatypeSmallDEFAULT, datatypeModuleDEFAULT
from someAssemblyRequired import makeStateJob
from typing import Optional
import importlib
import importlib.util
import inspect
import more_itertools
import numpy
import pathlib
import python_minifier

identifierCallableLaunch = "goGoGadgetAbsurdity"

def makeStrRLEcompacted(arrayTarget: numpy.ndarray, identifierName: str) -> str:
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
    datatypeLarge = keywordArguments.get('datatypeLarge', datatypeLargeDEFAULT)
    datatypeMedium = keywordArguments.get('datatypeMedium', datatypeMediumDEFAULT)
    datatypeSmall = keywordArguments.get('datatypeSmall', datatypeSmallDEFAULT)
    datatypeModule = keywordArguments.get('datatypeModule', datatypeModuleDEFAULT)

    dtypeLarge = make_dtype(datatypeLarge, datatypeModule) # type: ignore
    dtypeMedium = make_dtype(datatypeMedium, datatypeModule) # type: ignore
    dtypeSmall = make_dtype(datatypeSmall, datatypeModule) # type: ignore

    stateJob = makeStateJob(listDimensions, writeJob=False, dtypeLarge = dtypeLarge, dtypeMedium = dtypeMedium, dtypeSmall = dtypeSmall)
    pathFilenameFoldsTotal = getPathFilenameFoldsTotal(stateJob['mapShape'])

    from syntheticModules import countSequential
    algorithmSource = countSequential
    codeSource = inspect.getsource(algorithmSource)

    if datatypeLarge:
        lineNumba = f"@numba.jit(numba.types.{datatypeLarge}(), cache=True, nopython=True, fastmath=True, forceinline=True, inline='always', looplift=False, _nrt=True, error_model='numpy', parallel=False, boundscheck=False, no_cfunc_wrapper=True, no_cpython_wrapper=False)"

    linesImport = "\n".join([
                        "import numpy"
                        , "import numba"
                        ])

    ImaIndent = '    '
    linesDataDynamic = """"""
    linesDataDynamic = "\n".join([linesDataDynamic
            # , ImaIndent + f"foldsTotal = numba.types.{datatypeLarge}(0)"
            # , ImaIndent + makeStrRLEcompacted(stateJob['foldGroups'], 'foldGroups')
            , ImaIndent + makeStrRLEcompacted(stateJob['gapsWhere'], 'gapsWhere')
            ])

    linesDataStatic = """"""
    linesDataStatic = "\n".join([linesDataStatic
            , ImaIndent + makeStrRLEcompacted(stateJob['connectionGraph'], 'connectionGraph')
            ])

    my = stateJob['my']
    track = stateJob['track']
    linesAlgorithm = """"""
    for lineSource in codeSource.splitlines():
        if lineSource.startswith(('#', 'import', 'from', '@numba.jit')):
            continue
        elif not lineSource:
            continue
        elif lineSource.startswith('def '):
            lineSource = "\n".join([lineNumba
                                , f"def {identifierCallableLaunch}():"
                                , linesDataDynamic
                                , linesDataStatic
                                ])
        elif 'taskIndex' in lineSource:
            continue
        elif 'my[indexMy.' in lineSource:
            if 'dimensionsTotal' in lineSource:
                continue
            # leaf1ndex = my[indexMy.leaf1ndex.value]
            identifier, statement = lineSource.split('=')
            lineSource = ImaIndent + identifier.strip() + f"=numba.types.{datatypeSmall}({str(eval(statement.strip()))})"
        elif 'track[indexTrack.' in lineSource:
            # leafAbove = track[indexTrack.leafAbove.value]
            identifier, statement = lineSource.split('=')
            lineSource = ImaIndent + makeStrRLEcompacted(eval(statement.strip()), identifier.strip())
        elif 'foldGroups[-1]' in lineSource:
            lineSource = lineSource.replace('foldGroups[-1]', str(stateJob['foldGroups'][-1]))
        elif 'dimensionsTotal' in lineSource:
            lineSource = lineSource.replace('dimensionsTotal', str(stateJob['my'][indexMy.dimensionsTotal]))

        linesAlgorithm = "\n".join([linesAlgorithm
                            , lineSource
                            ])

    linesLaunch = """"""
    linesLaunch = linesLaunch + f"""
if __name__ == '__main__':
    # import time
    # timeStart = time.perf_counter()
    {identifierCallableLaunch}()
    # print(time.perf_counter() - timeStart)
"""

    linesWriteFoldsTotal = """"""
    linesWriteFoldsTotal = "\n".join([linesWriteFoldsTotal
                                    , f"    groupsOfFolds *= {str(stateJob['foldGroups'][-1])}"
                                    , "    print(groupsOfFolds)"
                                    , "    with numba.objmode():"
                                    , f"        open('{pathFilenameFoldsTotal.as_posix()}', 'w').write(str(groupsOfFolds))"
                                    , "    return groupsOfFolds"
                                    ])

    linesAll = "\n".join([
                        linesImport
                        , linesAlgorithm
                        , linesWriteFoldsTotal
                        , linesLaunch
                        ])

    pathFilenameDestination = pathFilenameFoldsTotal.with_suffix(".py")
    pathFilenameDestination.write_text(linesAll)

    return pathFilenameDestination

if __name__ == '__main__':
    listDimensions = [6,6]
    datatypeLarge = 'int64'
    datatypeMedium = 'uint8'
    datatypeSmall = datatypeMedium
    pathFilenameModule = writeModuleWithNumba(listDimensions, datatypeLarge=datatypeLarge, datatypeMedium=datatypeMedium, datatypeSmall=datatypeSmall)

    # Induce numba.jit compilation
    moduleSpec = importlib.util.spec_from_file_location(pathFilenameModule.stem, pathFilenameModule)
    if moduleSpec is None: raise ImportError(f"Could not load module specification from {pathFilenameModule}")
    module = importlib.util.module_from_spec(moduleSpec)
    if moduleSpec.loader is None: raise ImportError(f"Could not load module from {moduleSpec}")
    moduleSpec.loader.exec_module(module)
