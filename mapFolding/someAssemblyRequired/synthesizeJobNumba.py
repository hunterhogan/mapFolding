"""
It's rough, but it works. Actually, the modules it produces are lightening fast.

Specific issues:
- When trying to run the synthesized file, `ModuleNotFoundError: No module named '<dynamic>'` unless I first re-save the file in the IDE.
- ast interprets the signature as `def countSequential() -> None:` even though there is a return statement.
- Similarly, but possibly my fault, `decorateCallableWithNumba` doesn't add the return type to the signature.

General issues:
- an ironic dearth of abstract functionality in this module based on ast.
    - I don't have much experience with ast.
    - ast is one of the few cases that absolutely benefits from an OOP paradigm, and I am comically inept at OOP.
- (almost) Everything prefixed with `Z0Z_` is something I want to substantially improve.
- convergence with other synthesize modules and functions would be good.
- while management of datatypes seems to be pretty good, managing pathFilenames could be better.
- as of this writing, there are zero direct tests for `someAssemblyRequired`.
"""
from mapFolding import indexMy, indexTrack, ParametersNumba, parametersNumbaDEFAULT, getFilenameFoldsTotal, getPathJobRootDEFAULT, getPathFilenameFoldsTotal
from mapFolding import setDatatypeElephino, setDatatypeFoldsTotal, setDatatypeLeavesTotal, setDatatypeModule, hackSSOTdatatype, computationState
from mapFolding.someAssemblyRequired import makeStateJob, decorateCallableWithNumba, Z0Z_UnhandledDecorators
from typing import Optional, Callable, List, Sequence, cast, Dict, Set, Any, Union
from Z0Z_tools import updateExtendPolishDictionaryLists
import ast
import collections
import importlib
import importlib.util
import inspect
import more_itertools
import numpy
import os
import pathlib
import python_minifier

dictionaryImportFrom: Dict[str, List[str]] = collections.defaultdict(list)
datatypeModuleScalar = 'numba'

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
    return stringMinimized

def makeImports() -> List[List[ast.ImportFrom]]:
    global dictionaryImportFrom
    dictionaryImportFrom = updateExtendPolishDictionaryLists(dictionaryImportFrom, destroyDuplicates=True)

    def parseAlias(aliasString: str):
        parts = aliasString.split(" as ")
        if len(parts) == 2:
            return ast.alias(name=parts[0].strip(), asname=parts[1].strip())
        return ast.alias(name=aliasString.strip(), asname=None)

    importStatements = [[
        ast.ImportFrom(module=module, names=[ast.alias(name=identifierName, asname=None)
                                            for identifierName in listIdentifiers], level=0)
                                            for module, listIdentifiers in dictionaryImportFrom.items()]
        ]

    return importStatements

def evaluateArrayIn_body(node: ast.FunctionDef, astArg: ast.arg, argData: numpy.ndarray) -> ast.FunctionDef:
    global dictionaryImportFrom
    arrayType = type(argData)
    moduleConstructor = arrayType.__module__
    constructorName = arrayType.__name__
    # NOTE hack
    constructorName = constructorName.replace('ndarray', 'array')
    dictionaryImportFrom[moduleConstructor].append(constructorName)

    for stmt in node.body.copy():
        if isinstance(stmt, ast.Assign):
            if isinstance(stmt.targets[0], ast.Name) and isinstance(stmt.value, ast.Subscript):
                astAssignee: ast.Name = stmt.targets[0]
                argData_dtypeName = hackSSOTdatatype(astAssignee.id)
                dictionaryImportFrom[moduleConstructor].append(argData_dtypeName)
                astSubscript: ast.Subscript = stmt.value
                if isinstance(astSubscript.value, ast.Name) and astSubscript.value.id == astArg.arg and isinstance(astSubscript.slice, ast.Attribute):
                    indexAs_astAttribute: ast.Attribute = astSubscript.slice
                    indexAsStr = ast.unparse(indexAs_astAttribute)
                    argDataSlice = argData[eval(indexAsStr)]

                    onlyDataRLE = makeStrRLEcompacted(argDataSlice)
                    astStatement = cast(ast.Expr, ast.parse(onlyDataRLE).body[0])
                    dataAst = astStatement.value

                    arrayCall = ast.Call(
                        func=ast.Name(id=constructorName, ctx=ast.Load()) , args=[dataAst]
                        , keywords=[ast.keyword(arg='dtype', value=ast.Name(id=argData_dtypeName, ctx=ast.Load()) ) ] )

                    assignment = ast.Assign( targets=[astAssignee], value=arrayCall )
                    node.body.insert(0, assignment)
                    node.body.remove(stmt)

    node.args.args.remove(astArg)
    return node

def evaluate_argIn_body(node: ast.FunctionDef, astArg: ast.arg, argData: numpy.ndarray, Z0Z_listChaff: List[str]) -> ast.FunctionDef:
    global dictionaryImportFrom
    moduleConstructor = datatypeModuleScalar
    for stmt in node.body.copy():
        if isinstance(stmt, ast.Assign):
            if isinstance(stmt.targets[0], ast.Name) and isinstance(stmt.value, ast.Subscript):
                astAssignee: ast.Name = stmt.targets[0]
                argData_dtypeName = hackSSOTdatatype(astAssignee.id)
                dictionaryImportFrom[moduleConstructor].append(argData_dtypeName)
                astSubscript: ast.Subscript = stmt.value
                if isinstance(astSubscript.value, ast.Name) and astSubscript.value.id == astArg.arg and isinstance(astSubscript.slice, ast.Attribute):
                    indexAs_astAttribute: ast.Attribute = astSubscript.slice
                    indexAsStr = ast.unparse(indexAs_astAttribute)
                    argDataSlice: int = argData[eval(indexAsStr)].item()
                    astCall = ast.Call(func=ast.Name(id=argData_dtypeName, ctx=ast.Load()) , args=[ast.Constant(value=argDataSlice)], keywords=[])
                    assignment = ast.Assign(targets=[astAssignee], value=astCall)
                    if astAssignee.id not in Z0Z_listChaff:
                        node.body.insert(0, assignment)
                    node.body.remove(stmt)
    node.args.args.remove(astArg)
    return node

def evaluateAnnAssignIn_body(node: ast.FunctionDef) -> ast.FunctionDef:
    global dictionaryImportFrom
    moduleConstructor = datatypeModuleScalar
    for stmt in node.body.copy():
        if isinstance(stmt, ast.AnnAssign):
            if isinstance(stmt.target, ast.Name) and isinstance(stmt.value, ast.Constant):
                astAssignee: ast.Name = stmt.target
                argData_dtypeName = hackSSOTdatatype(astAssignee.id)
                dictionaryImportFrom[moduleConstructor].append(argData_dtypeName)
                astCall = ast.Call(func=ast.Name(id=argData_dtypeName, ctx=ast.Load()) , args=[stmt.value], keywords=[])
                assignment = ast.Assign(targets=[astAssignee], value=astCall)
                node.body.insert(0, assignment)
                node.body.remove(stmt)
    return node

def move_argTo_body(node: ast.FunctionDef, astArg: ast.arg, argData: numpy.ndarray) -> ast.FunctionDef:
    arrayType = type(argData)
    moduleConstructor = arrayType.__module__
    constructorName = arrayType.__name__
    # NOTE hack
    constructorName = constructorName.replace('ndarray', 'array')
    argData_dtype: numpy.dtype = argData.dtype
    argData_dtypeName = argData.dtype.name

    global dictionaryImportFrom
    dictionaryImportFrom[moduleConstructor].append(constructorName)
    dictionaryImportFrom[moduleConstructor].append(argData_dtypeName)

    onlyDataRLE = makeStrRLEcompacted(argData)
    astStatement = cast(ast.Expr, ast.parse(onlyDataRLE).body[0])
    dataAst = astStatement.value

    arrayCall = ast.Call(
        func=ast.Name(id=constructorName, ctx=ast.Load())
        , args=[dataAst]
        , keywords=[ast.keyword(arg='dtype' , value=ast.Name(id=argData_dtypeName , ctx=ast.Load()) ) ] )

    assignment = ast.Assign( targets=[ast.Name(id=astArg.arg, ctx=ast.Store())], value=arrayCall )
    node.body.insert(0, assignment)
    node.args.args.remove(astArg)

    return node

def makeDecorator(FunctionDefTarget: ast.FunctionDef, parametersNumba: Optional[ParametersNumba]=None) -> ast.FunctionDef:
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
    global dictionaryImportFrom
    dictionaryImportFrom['numba'].append('jit')
    FunctionDefTarget = decorateCallableWithNumba(FunctionDefTarget, parametersNumba)
    # make sure the decorator is rendered as `@jit` and not `@numba.jit`
    for decoratorItem in FunctionDefTarget.decorator_list:
        if isinstance(decoratorItem, ast.Call) and isinstance(decoratorItem.func, ast.Attribute) and decoratorItem.func.attr == "jit":
            decoratorItem.func = ast.Name(id="jit", ctx=ast.Load())
    return FunctionDefTarget

def makeLauncher(identifierCallable: str) -> ast.Module:
    linesLaunch = f"""
if __name__ == '__main__':
    import time
    timeStart = time.perf_counter()
    {identifierCallable}()
    print(time.perf_counter() - timeStart)
"""
    astLaunch = ast.parse(linesLaunch)
    return astLaunch

def make_writeFoldsTotal(stateJob: computationState, pathFilenameFoldsTotal: pathlib.Path) -> ast.Module:
    global dictionaryImportFrom
    dictionaryImportFrom['numba'].append("objmode")
    linesWriteFoldsTotal = f"""
groupsOfFolds *= {str(stateJob['foldGroups'][-1])}
print(groupsOfFolds)
with objmode():
    open('{pathFilenameFoldsTotal.as_posix()}', 'w').write(str(groupsOfFolds))
return groupsOfFolds
    """
    return ast.parse(linesWriteFoldsTotal)

def removeIdentifierFrom_body(node: ast.FunctionDef, astArg: ast.arg) -> ast.FunctionDef:
    for stmt in node.body.copy():
        if isinstance(stmt, ast.Assign):
            if isinstance(stmt.targets[0], ast.Subscript) and isinstance(stmt.targets[0].value, ast.Name):
                if stmt.targets[0].value.id == astArg.arg:
                    node.body.remove(stmt)
    node.args.args.remove(astArg)
    return node

def astObjectToAstConstant(astFunction: ast.FunctionDef, object: str, value: int) -> ast.FunctionDef:
    """
    Replaces nodes in astFunction matching the AST of the string `object`
    with a constant node holding the provided value.
    """
    targetExpression = ast.parse(object, mode='eval').body
    targetDump = ast.dump(targetExpression, annotate_fields=False)

    class ReplaceObjectWithConstant(ast.NodeTransformer):
        def __init__(self, targetDump: str, constantValue: int) -> None:
            self.targetDump = targetDump
            self.constantValue = constantValue

        def generic_visit(self, node: ast.AST) -> ast.AST:
            currentDump = ast.dump(node, annotate_fields=False)
            if currentDump == self.targetDump:
                return ast.copy_location(ast.Constant(value=self.constantValue), node)
            return super().generic_visit(node)

    transformer = ReplaceObjectWithConstant(targetDump, value)
    newFunction = transformer.visit(astFunction)
    ast.fix_missing_locations(newFunction)
    return newFunction

def astNameToAstConstant(astFunction: ast.FunctionDef, name: str, value: int) -> ast.FunctionDef:
    class ReplaceNameWithConstant(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            if node.id == name:
                return ast.copy_location(ast.Constant(value=value), node)
            return node
    return ReplaceNameWithConstant().visit(astFunction)

def writeJobNumba(listDimensions: Sequence[int], callableSource: Callable, parametersNumba: Optional[ParametersNumba]=None, pathFilenameWriteJob: Optional[Union[str, os.PathLike[str]]] = None) -> pathlib.Path:
    stateJob = makeStateJob(listDimensions, writeJob=False)
    codeSource = inspect.getsource(callableSource)
    astSource = ast.parse(codeSource)

    pathFilenameFoldsTotal = getPathFilenameFoldsTotal(stateJob['mapShape'])

    FunctionDefTarget = next((node for node in astSource.body if isinstance(node, ast.FunctionDef) and node.name == callableSource.__name__), None)

    if not FunctionDefTarget:
        raise ValueError(f"Could not find function {callableSource.__name__} in source code")

    Z0Z_listArgsTarget = ['connectionGraph', 'gapsWhere']
    Z0Z_listArraysEvaluate = ['track']
    Z0Z_listArgsEvaluate = ['my']
    Z0Z_listChaff = ['taskIndex', 'dimensionsTotal']
    Z0Z_listArgsRemove = ['foldGroups']
    for astArgument in FunctionDefTarget.args.args.copy():
        if astArgument.arg in Z0Z_listArgsTarget:
            FunctionDefTarget = move_argTo_body(FunctionDefTarget, astArgument, stateJob[astArgument.arg])
        elif astArgument.arg in Z0Z_listArraysEvaluate:
            FunctionDefTarget = evaluateArrayIn_body(FunctionDefTarget, astArgument, stateJob[astArgument.arg])
        elif astArgument.arg in Z0Z_listArgsEvaluate:
            FunctionDefTarget = evaluate_argIn_body(FunctionDefTarget, astArgument, stateJob[astArgument.arg], Z0Z_listChaff)
        elif astArgument.arg in Z0Z_listArgsRemove:
            FunctionDefTarget = removeIdentifierFrom_body(FunctionDefTarget, astArgument)

    FunctionDefTarget = evaluateAnnAssignIn_body(FunctionDefTarget)
    FunctionDefTarget = astNameToAstConstant(FunctionDefTarget, 'dimensionsTotal', int(stateJob['my'][indexMy.dimensionsTotal]))
    FunctionDefTarget = astObjectToAstConstant(FunctionDefTarget, 'foldGroups[-1]', int(stateJob['foldGroups'][-1]))

    global identifierCallableLaunch
    identifierCallableLaunch = FunctionDefTarget.name

    FunctionDefTarget = makeDecorator(FunctionDefTarget, parametersNumba)

    astWriteFoldsTotal = make_writeFoldsTotal(stateJob, pathFilenameFoldsTotal)
    FunctionDefTarget.body += astWriteFoldsTotal.body

    astLauncher = makeLauncher(FunctionDefTarget.name)

    astImports = makeImports()

    astModule = ast.Module(
        body=cast(List[ast.stmt]
                , astImports
                + [FunctionDefTarget]
                + [astLauncher])
        , type_ignores=[]
    )
    ast.fix_missing_locations(astModule)

    codeSource = ast.unparse(astModule)

    if pathFilenameWriteJob is None:
        filename = getFilenameFoldsTotal(stateJob['mapShape'])
        pathRoot = getPathJobRootDEFAULT()
        pathFilenameWriteJob = pathlib.Path(pathRoot, pathlib.Path(filename).stem, pathlib.Path(filename).with_suffix('.py'))
    else:
        pathFilenameWriteJob = pathlib.Path(pathFilenameWriteJob)
    pathFilenameWriteJob.parent.mkdir(parents=True, exist_ok=True)

    pathFilenameWriteJob.write_text(codeSource)
    return pathFilenameWriteJob

if __name__ == '__main__':
    listDimensions = [2,15]
    setDatatypeFoldsTotal('int64', sourGrapes=True)
    setDatatypeElephino('uint8', sourGrapes=True)
    setDatatypeLeavesTotal('uint8', sourGrapes=True)
    from mapFolding.syntheticModules.numba_countSequential import countSequential
    callableSource = countSequential
    pathFilenameModule = writeJobNumba(listDimensions, callableSource, parametersNumbaDEFAULT)

    # Induce numba.jit compilation
    # TODO Inducing compilation might be causing the `ModuleNotFoundError: No module named '<dynamic>'` error

    # moduleSpec = importlib.util.spec_from_file_location(pathFilenameModule.stem, pathFilenameModule)
    # if moduleSpec is None: raise ImportError(f"Could not load module specification from {pathFilenameModule}")
    # module = importlib.util.module_from_spec(moduleSpec)
    # if moduleSpec.loader is None: raise ImportError(f"Could not load module from {moduleSpec}")
    # moduleSpec.loader.exec_module(module)

    # from mapFolding.someAssemblyRequired.getLLVMforNoReason import writeModuleLLVM
    # pathFilenameLLVM = writeModuleLLVM(pathFilenameModule, identifierCallableLaunch)
