from tkinter import N
from mapFolding import getPathFilenameFoldsTotal, indexMy, indexTrack, ParametersNumba, parametersNumbaDEFAULT
from mapFolding import setDatatypeElephino, setDatatypeFoldsTotal, setDatatypeLeavesTotal, setDatatypeModule, hackSSOTdatatype
from someAssemblyRequired import makeStateJob, decorateCallableWithNumba, Z0Z_UnhandledDecorators
from typing import Optional, Callable, List, Sequence, cast, Dict, Set, Any
import ast
import importlib
import importlib.util
import inspect
import more_itertools
import numpy
import pathlib
import python_minifier
from Z0Z_tools import updateExtendPolishDictionaryLists

dictionaryImportFrom_global: Dict[str, List[str]] = {}

identifierCallableLaunch = "goGoGadgetAbsurdity"

def makeStrRLEcompacted(arrayTarget: numpy.ndarray, identifierName: Optional[str]=None) -> str:
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

    if identifierName:
        return f"{identifierName} = array({stringMinimized}, dtype={arrayTarget.dtype})"
    return f"array({stringMinimized}, dtype={arrayTarget.dtype})"

def makeImports():
    # TODO replace the hardcoded sets with dynamic signals of the necessary imports
    dictionaryImportFrom_local: Dict[str, List[str]] = {}
    dictionaryImportFrom_local['numpy'] = [ "array", hackSSOTdatatype("datatypeElephino"), hackSSOTdatatype("datatypeLeavesTotal"), ]
    dictionaryImportFrom_local['numba'] = [ "jit", "objmode" ]
    setImportFromNumbaTypesRaw = set([ hackSSOTdatatype("datatypeElephino"), hackSSOTdatatype("datatypeLeavesTotal"), hackSSOTdatatype("datatypeFoldsTotal"), ])
    setImportFromNumbaTypes = set([ f"{numbaType} as {numbaType}Numba" for numbaType in setImportFromNumbaTypesRaw ])

    dictionaryImportFrom = updateExtendPolishDictionaryLists(
        dictionaryImportFrom_global
        , dictionaryImportFrom_local
        , destroyDuplicates=True)

    def parseAlias(aliasString: str):
        parts = aliasString.split(" as ")
        if len(parts) == 2:
            return ast.alias(name=parts[0].strip(), asname=parts[1].strip())
        return ast.alias(name=aliasString.strip(), asname=None)

    importStatements = [[
        ast.ImportFrom(module=module, names=[ast.alias(name=identifierName, asname=None)
                                            for identifierName in listIdentifiers], level=0)
                                            for module, listIdentifiers in dictionaryImportFrom.items()]
        + [ast.ImportFrom( module="numba.types", names=[parseAlias(name) for name in setImportFromNumbaTypes], level=0 )]
        ]

    return importStatements

def move_argTo_body(node: ast.FunctionDef, argToMove, argData):
    # print(ast.unparse(argToMove))
    linesData = makeStrRLEcompacted(argData)
    target = ast.Name(id=argToMove.arg, ctx=ast.Store())
    value = ast.parse(linesData).body[0].value # type: ignore
    assignment = ast.Assign(targets=[target], value=value)
    node.body.insert(0, assignment)
    node.args.args.remove(argToMove)
    return node

def makeDecorator(FunctionDefTarget: ast.FunctionDef, parametersNumba: Optional[ParametersNumba]=None):
    if parametersNumba is None:
        parametersNumbaExtracted: Dict[str, Any] = {}
        for decoratorItem in FunctionDefTarget.decorator_list.copy():
            if isinstance(decoratorItem, ast.Call) and isinstance(decoratorItem.func, ast.Attribute):
                if getattr(decoratorItem.func.value, "id", None) == "numba" and decoratorItem.func.attr == "jit":
                    FunctionDefTarget.decorator_list.remove(decoratorItem)
                    for keywordItem in decoratorItem.keywords:
                        if isinstance(keywordItem.value, ast.Constant) and keywordItem.arg is not None:
                            parametersNumbaExtracted[keywordItem.arg] = keywordItem.value.value
        if parametersNumbaExtracted:
            parametersNumba = ParametersNumba(parametersNumbaExtracted)  # type: ignore
    else:
        # TODO code duplication
        for decoratorItem in FunctionDefTarget.decorator_list.copy():
            if isinstance(decoratorItem, ast.Call) and isinstance(decoratorItem.func, ast.Attribute):
                if getattr(decoratorItem.func.value, "id", None) == "numba" and decoratorItem.func.attr == "jit":
                    FunctionDefTarget.decorator_list.remove(decoratorItem)
    FunctionDefTarget = Z0Z_UnhandledDecorators(FunctionDefTarget)
    dictionaryImportFrom_global['numba'] = ['jit']
    FunctionDefTarget = decorateCallableWithNumba(FunctionDefTarget, parametersNumba)
    # make sure the decorator is rendered as `@jit` and not `@numba.jit`
    for decoratorItem in FunctionDefTarget.decorator_list:
        if isinstance(decoratorItem, ast.Call) and isinstance(decoratorItem.func, ast.Attribute) and decoratorItem.func.attr == "jit":
            decoratorItem.func = ast.Name(id="jit", ctx=ast.Load())
    return FunctionDefTarget

def writeJobNumba(listDimensions: Sequence[int], callableSource: Callable, parametersNumba: Optional[ParametersNumba]=None) -> pathlib.Path:
    stateJob = makeStateJob(listDimensions, writeJob=False)
    pathFilenameFoldsTotal = getPathFilenameFoldsTotal(stateJob['mapShape'])

    codeSource = inspect.getsource(callableSource)
    astSource = ast.parse(codeSource)

    FunctionDefTarget = next((node for node in astSource.body if isinstance(node, ast.FunctionDef) and node.name == callableSource.__name__), None)

    if not FunctionDefTarget:
        raise ValueError(f"Could not find function {callableSource.__name__} in source code")

    astImports = makeImports()

    Z0Z_listArgsTarget = ['connectionGraph', 'gapsWhere']
    Z0Z_listArgsRemove = ['foldGroups', 'my', 'track']
    for astArgument in FunctionDefTarget.args.args.copy():
        if astArgument.arg in Z0Z_listArgsTarget:
            FunctionDefTarget = move_argTo_body(FunctionDefTarget, astArgument, stateJob[astArgument.arg])
        elif astArgument.arg in Z0Z_listArgsRemove:
            FunctionDefTarget.args.args.remove(astArgument)

    assert FunctionDefTarget.args.args == [], "FunctionDefTarget.args.args should be empty after moving arguments to body"

    FunctionDefTarget = makeDecorator(FunctionDefTarget, parametersNumba)

    astModule = ast.Module(
        body=cast(List[ast.stmt]
                , astImports
                + [FunctionDefTarget])
        , type_ignores=[]
    )
    ast.fix_missing_locations(astModule)

    codeSource = ast.unparse(astModule)

    # lineNumba = f"@jit({hackSSOTdatatype('datatypeFoldsTotal')}Numba(), cache=True, nopython=True, fastmath=True, forceinline=True, inline='always', looplift=False, _nrt=True, error_model='numpy', parallel=False, boundscheck=False, no_cfunc_wrapper=False, no_cpython_wrapper=False)"
    ImaIndent = '    '

    my = stateJob['my']
    track = stateJob['track']
    linesAlgorithm = """"""
    for lineSource in codeSource.splitlines():
        if lineSource.startswith(('#', '@numba')):
            continue
        elif not lineSource:
            continue
        elif lineSource.startswith('def '):
            lineSource = "\n".join([
                # lineNumba,
                                f"def {identifierCallableLaunch}():"
                                ])
        elif 'taskIndex' in lineSource:
            continue
        elif 'my[indexMy.' in lineSource:
            if 'dimensionsTotal' in lineSource:
                continue
            # Statements are in the form: leaf1ndex = my[indexMy.leaf1ndex.value]
            identifier, statement = lineSource.split('=')
            lineSource = ImaIndent + identifier.strip() + f"={hackSSOTdatatype(identifier.strip())}Numba({str(eval(statement.strip()))})"
        elif ': int =' in lineSource or ':int=' in lineSource:
            if 'dimensionsTotal' in lineSource:
                continue
            # Statements are in the form: groupsOfFolds: int = 0
            assignment, statement = lineSource.split('=')
            identifier = assignment.split(':')[0].strip()
            lineSource = ImaIndent + identifier.strip() + f"={hackSSOTdatatype(identifier.strip())}Numba({str(eval(statement.strip()))})"
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
                                    , "    with objmode():"
                                    , f"        open('{pathFilenameFoldsTotal.as_posix()}', 'w').write(str(groupsOfFolds))"
                                    , "    return groupsOfFolds"
                                    ])

    linesAll = "\n".join([
                        # linesImport
                        linesAlgorithm
                        , linesWriteFoldsTotal
                        , linesLaunch
                        ])

    pathFilenameDestination = pathFilenameFoldsTotal.with_suffix(".py")
    pathFilenameDestination.write_text(linesAll)

    return pathFilenameDestination

if __name__ == '__main__':
    listDimensions = [4,4]
    setDatatypeFoldsTotal('int64', sourGrapes=True)
    setDatatypeElephino('uint8', sourGrapes=True)
    setDatatypeLeavesTotal('uint8', sourGrapes=True)
    from mapFolding.syntheticModules.numba_countSequential import countSequential
    callableSource = countSequential
    pathFilenameModule = writeJobNumba(listDimensions, callableSource, parametersNumbaDEFAULT)

    # Induce numba.jit compilation
    moduleSpec = importlib.util.spec_from_file_location(pathFilenameModule.stem, pathFilenameModule)
    if moduleSpec is None: raise ImportError(f"Could not load module specification from {pathFilenameModule}")
    module = importlib.util.module_from_spec(moduleSpec)
    if moduleSpec.loader is None: raise ImportError(f"Could not load module from {moduleSpec}")
    moduleSpec.loader.exec_module(module)

    # from mapFolding.someAssemblyRequired.getLLVMforNoReason import writeModuleLLVM
    # pathFilenameLLVM = writeModuleLLVM(pathFilenameModule, identifierCallableLaunch)
