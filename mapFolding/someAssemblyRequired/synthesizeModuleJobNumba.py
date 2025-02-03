from mapFolding import getPathFilenameFoldsTotal, indexMy, indexTrack
from mapFolding import setDatatypeElephino, setDatatypeFoldsTotal, setDatatypeLeavesTotal, setDatatypeModule, hackSSOTdatatype
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
    """Converts a NumPy array into a compressed string representation using run-length encoding (RLE).

    This function takes a NumPy array and converts it into an optimized string representation by:
    1. Compressing consecutive sequences of numbers into range objects
    2. Minimizing repeated zeros using array multiplication syntax
    3. Converting the result into a valid Python array initialization statement

    Parameters:
        arrayTarget (numpy.ndarray): The input NumPy array to be converted
        identifierName (str): The variable name to use in the output string

    Returns:
        str: A string containing Python code that recreates the input array in compressed form.
            Format: "{identifierName} = numpy.array({compressed_data}, dtype=numpy.{dtype})"

    Example:
        >>> arr = numpy.array([[0,0,0,1,2,3,4,0,0]])
        >>> print(makeStrRLEcompacted(arr, "myArray"))
        "myArray = numpy.array([[0]*3,*range(1,5),[0]*2], dtype=numpy.int64)"

    Notes:
        - Sequences of 4 or fewer numbers are kept as individual values
        - Sequences longer than 4 numbers are converted to range objects
        - Consecutive zeros are compressed using multiplication syntax
        - The function preserves the original array's dtype
    """

    def compressRangesNDArrayNoFlatten(arraySlice):
        if isinstance(arraySlice, numpy.ndarray) and arraySlice.ndim > 1:
            return [compressRangesNDArrayNoFlatten(arraySlice[index]) for index in range(arraySlice.shape[0])]
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

    arrayAsNestedLists = compressRangesNDArrayNoFlatten(arrayTarget)

    stringMinimized = python_minifier.minify(str(arrayAsNestedLists))
    commaZeroMaximum = arrayTarget.shape[-1] - 1
    stringMinimized = stringMinimized.replace('[0' + ',0'*commaZeroMaximum + ']', '[0]*'+str(commaZeroMaximum+1))
    for countZeros in range(commaZeroMaximum, 2, -1):
        stringMinimized = stringMinimized.replace(',0'*countZeros + ']', ']+[0]*'+str(countZeros))

    stringMinimized = stringMinimized.replace('range', '*range')

    return f"{identifierName} = numpy.array({stringMinimized}, dtype=numpy.{arrayTarget.dtype})"

def writeModuleWithNumba(listDimensions) -> pathlib.Path:
    """
    Writes a Numba-optimized Python module for map folding calculations.

    This function takes map dimensions and generates a specialized Python module with Numba
    optimizations. It processes a sequential counting algorithm, adds Numba decorators and
    necessary data structures, and writes the resulting code to a file.

    Parameters:
        listDimensions: List of integers representing the dimensions of the map to be folded.

    Returns:
        pathlib.Path: Path to the generated Python module file.

    The generated module includes:
    - Numba JIT compilation decorators for performance optimization
    - Required numpy and numba imports
    - Dynamic and static data structures needed for folding calculations
    - Processed algorithm from the original sequential counter
    - Launch code for standalone execution
    - Code to write the final fold count to a file
    The function handles:
    - Translation of original code to Numba-compatible syntax
    - Insertion of pre-calculated values from the state job
    - Management of variable declarations and assignments
    - Setup of proper data types for Numba optimization
    - Organization of the output file structure

    Note:
        The generated module requires Numba and numpy to be installed.
        The output file will be placed in the same directory as the folds total file,
        with a .py extension.
    """
    stateJob = makeStateJob(listDimensions, writeJob=False)
    pathFilenameFoldsTotal = getPathFilenameFoldsTotal(stateJob['mapShape'])

    from syntheticModules import countSequential
    algorithmSource = countSequential
    codeSource = inspect.getsource(algorithmSource)

    lineNumba = f"@numba.jit(numba.types.{hackSSOTdatatype('datatypeFoldsTotal')}(), cache=True, nopython=True, fastmath=True, forceinline=True, inline='always', looplift=False, _nrt=True, error_model='numpy', parallel=False, boundscheck=False, no_cfunc_wrapper=False, no_cpython_wrapper=False)"

    linesImport = "\n".join([
                        "import numpy"
                        , "import numba"
                        ])

    ImaIndent = '    '
    linesDataDynamic = """"""
    linesDataDynamic = "\n".join([linesDataDynamic
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
            # Statements are in the form: leaf1ndex = my[indexMy.leaf1ndex.value]
            identifier, statement = lineSource.split('=')
            lineSource = ImaIndent + identifier.strip() + f"=numba.types.{hackSSOTdatatype(identifier.strip())}({str(eval(statement.strip()))})"
        elif ': int =' in lineSource or ':int=' in lineSource:
            if 'dimensionsTotal' in lineSource:
                continue
            # Statements are in the form: groupsOfFolds: int = 0
            assignment, statement = lineSource.split('=')
            identifier = assignment.split(':')[0].strip()
            lineSource = ImaIndent + identifier.strip() + f"=numba.types.{hackSSOTdatatype(identifier.strip())}({str(eval(statement.strip()))})"
        elif 'track[indexTrack.' in lineSource:
            # Statements are in the form: leafAbove = track[indexTrack.leafAbove.value]
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
    listDimensions = [5,5]
    setDatatypeFoldsTotal('int64', sourGrapes=True)
    setDatatypeElephino('uint8', sourGrapes=True)
    setDatatypeLeavesTotal('int8', sourGrapes=True)
    pathFilenameModule = writeModuleWithNumba(listDimensions)

    # Induce numba.jit compilation
    moduleSpec = importlib.util.spec_from_file_location(pathFilenameModule.stem, pathFilenameModule)
    if moduleSpec is None: raise ImportError(f"Could not load module specification from {pathFilenameModule}")
    module = importlib.util.module_from_spec(moduleSpec)
    if moduleSpec.loader is None: raise ImportError(f"Could not load module from {moduleSpec}")
    moduleSpec.loader.exec_module(module)
