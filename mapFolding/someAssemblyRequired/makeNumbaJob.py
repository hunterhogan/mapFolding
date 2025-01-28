"""Create a python module hardcoded to compute a map's foldsTotal.
- NumPy ndarray.
- Numba optimized.
- Absolutely no other imports.

Can create LLVM IR from the module: of unknown utility.
"""
# from mapFolding import dtypeDefault, dtypeSmall
from mapFolding import make_dtype, datatypeLarge, dtypeLarge
# from mapFolding.someAssemblyRequired.inlineAfunction import Z0Z_inlineMapFolding
# from someAssemblyRequired.makeComputationState import Z0Z_makeJob
import importlib
import llvmlite.binding
import numpy
import pathlib
import pickle
import python_minifier
import itertools
import more_itertools
from mapFolding import indexMy, indexTrack, dtypeDefault, dtypeLarge
import ast
import copy
import pathlib
from typing import Any, Optional, Sequence, Type

def Z0Z_makeJob(listDimensions: Sequence[int], **keywordArguments: Optional[Type[Any]]):
    from mapFolding import outfitCountFolds
    stateUniversal = outfitCountFolds(listDimensions, computationDivisions=None, CPUlimit=None, **keywordArguments)
    from mapFolding.someAssemblyRequired.countInitializeNoNumba import countInitialize
    countInitialize(stateUniversal['connectionGraph'], stateUniversal['gapsWhere'], stateUniversal['my'], stateUniversal['track'])
    from mapFolding import getPathFilenameFoldsTotal
    pathFilenameChopChop = getPathFilenameFoldsTotal(stateUniversal['mapShape'])
    import pathlib
    suffix = pathFilenameChopChop.suffix
    pathJob = pathlib.Path(str(pathFilenameChopChop)[0:-len(suffix)])
    pathJob.mkdir(parents=True, exist_ok=True)
    pathFilenameJob = pathJob / 'stateJob.pkl'

    pathFilenameFoldsTotal = getPathFilenameFoldsTotal(stateUniversal['mapShape'], pathFilenameJob.parent)
    stateJob = {**stateUniversal, 'pathFilenameFoldsTotal': pathFilenameFoldsTotal}

    del stateJob['mapShape']

    import pickle
    pathFilenameJob.write_bytes(pickle.dumps(stateJob))
    return pathFilenameJob

def getDictionaryEnumValues():
    dictionaryEnumValues = {}
    for enumIndex in [indexMy, indexTrack]:
        for memberName, memberValue in enumIndex._member_map_.items():
            dictionaryEnumValues[f"{enumIndex.__name__}.{memberName}.value"] = memberValue.value
    return dictionaryEnumValues

def getHardcodedShapes():
    """Temporary function to provide array shapes until proper SSOT is implemented."""
    return {
        'my': '(numba.int64[::1])',
        'track': '(numba.int64[:,::1])',
        'gapsWhere': '(numba.int64[::1])',
        'connectionGraph': '(numba.int64[:,:,::1])',
        'foldGroups': '(numba.int64[::1])',
    }

def generateDecorator(functionName: str, isParallel: bool = False) -> str:
    """Generate a Numba decorator string based on function signature."""
    shapes = getHardcodedShapes()
    parameters = []

    # Map function parameters to their shapes
    parameterMapping = {
        'my': 'numba.int64[::1]',
        'track': 'numba.int64[:,::1]',
        'gapsWhere': 'numba.int64[::1]',
        'connectionGraph': 'numba.int64[:,:,::1]',
        'foldGroups': 'numba.int64[::1]',
    }

    # Get function parameters from theDao
    import inspect
    from mapFolding import theDao
    functionObj = getattr(theDao, functionName)
    signature = inspect.signature(functionObj)

    # Debug info
    print(f"Creating decorator for function: {functionName}")
    print(f"Function parameters: {list(signature.parameters.keys())}")

    # Build parameter list in correct order based on function signature
    for param in signature.parameters:
        if param in parameterMapping:
            parameters.append(parameterMapping[param])

    # Format into numba decorator string
    otherParams = "cache=True, nopython=True, fastmath=True, looplift=False, error_model='numpy', parallel=False, boundscheck=False"
    decoratorStr = f"@numba.jit(({', '.join(parameters)}), {otherParams})"

    print(f"Generated decorator: {decoratorStr}")
    return decoratorStr

class RecursiveInlinerWithEnum(ast.NodeTransformer):
    def __init__(self, dictionaryFunctions, dictionaryEnumValues):
        self.dictionaryFunctions = dictionaryFunctions
        self.dictionaryEnumValues = dictionaryEnumValues
        self.processed = set()  # Track processed functions to avoid infinite recursion

    def inlineFunctionBody(self, functionName):
        if functionName in self.processed:
            return None

        self.processed.add(functionName)
        inlineDefinition = self.dictionaryFunctions[functionName]
        # Recursively process the function body
        for node in ast.walk(inlineDefinition):
            self.visit(node)
        return inlineDefinition

    def visit_Attribute(self, node):
        # Substitute enum identifiers (e.g., indexMy.leaf1ndex.value)
        if isinstance(node.value, ast.Attribute) and isinstance(node.value.value, ast.Name):
            enumPath = f"{node.value.value.id}.{node.value.attr}.{node.attr}"
            if enumPath in self.dictionaryEnumValues:
                return ast.Constant(value=self.dictionaryEnumValues[enumPath])
        return self.generic_visit(node)

    def visit_Call(self, node):
        callNode = self.generic_visit(node)
        if isinstance(callNode, ast.Call) and isinstance(callNode.func, ast.Name) and callNode.func.id in self.dictionaryFunctions:
            inlineDefinition = self.inlineFunctionBody(callNode.func.id)
            if (inlineDefinition and inlineDefinition.body):
                lastStmt = inlineDefinition.body[-1]
                if isinstance(lastStmt, ast.Return) and lastStmt.value is not None:
                    return self.visit(lastStmt.value)
                elif isinstance(lastStmt, ast.Expr) and lastStmt.value is not None:
                    return self.visit(lastStmt.value)
                return None
        return callNode

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Name) and node.value.func.id in self.dictionaryFunctions:
                inlineDefinition = self.inlineFunctionBody(node.value.func.id)
                if inlineDefinition:
                    return [self.visit(stmt) for stmt in inlineDefinition.body]
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        """Add decorator if function doesn't have one."""
        # Skip decorator addition since they're already in theDao.py
        return self.generic_visit(node)

def findRequiredImports(node):
    """Find all modules that need to be imported based on AST analysis.
    NOTE: due to hardcoding, this is a glorified regex. No, wait, this is less versatile than regex."""
    requiredImports = set()

    class ImportFinder(ast.NodeVisitor):
        def visit_Name(self, node):
            if node.id in {'numba'}:
                requiredImports.add(node.id)
            self.generic_visit(node)

        def visitDecorator(self, node):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == 'jit':
                    requiredImports.add('numba')
            self.generic_visit(node)

    ImportFinder().visit(node)
    return requiredImports

def generateImports(requiredImports):
    """Generate import statements based on required modules."""
    importStatements = []

    # Map of module names to their import statements
    importMapping = {
        'numba': 'import numba',
    }

    for moduleName in sorted(requiredImports):
        if moduleName in importMapping:
            importStatements.append(importMapping[moduleName])

    return '\n'.join(importStatements)

def inlineFunctions(sourceCode, targetFunctionName, dictionaryEnumValues):
    dictionaryParsed = ast.parse(sourceCode)
    dictionaryFunctions = {
        element.name: element
        for element in dictionaryParsed.body
        if isinstance(element, ast.FunctionDef)
    }
    nodeTarget = dictionaryFunctions[targetFunctionName]
    nodeInliner = RecursiveInlinerWithEnum(dictionaryFunctions, dictionaryEnumValues)
    nodeInlined = nodeInliner.visit(nodeTarget)
    ast.fix_missing_locations(nodeInlined)

    # Generate imports
    requiredImports = findRequiredImports(nodeInlined)
    importStatements = generateImports(requiredImports)

    # Combine imports with inlined code
    inlinedCode = importStatements + '\n\n' + ast.unparse(ast.Module(body=[nodeInlined], type_ignores=[]))
    return inlinedCode

def Z0Z_inlineMapFolding():
    dictionaryEnumValues = getDictionaryEnumValues()

    pathFilenameSource = pathlib.Path("/apps/mapFolding/mapFolding/theDao.py")
    codeSource = pathFilenameSource.read_text()

    listCallables = [
        'countInitialize',
        'countParallel',
        'countSequential',
    ]

    listPathFilenamesDestination: list[pathlib.Path] = []
    for callableTarget in listCallables:
        pathFilenameDestination = pathFilenameSource.parent / "someAssemblyRequired" / pathFilenameSource.with_stem(callableTarget).name
        codeInlined = inlineFunctions(codeSource, callableTarget, dictionaryEnumValues)
        pathFilenameDestination.write_text(codeInlined)
        listPathFilenamesDestination.append(pathFilenameDestination)

    listNoNumba = [
        'countInitialize',
        'countSequential',
    ]

    listPathFilenamesNoNumba = []
    for pathFilename in listPathFilenamesDestination:
        if pathFilename.stem in listNoNumba:
            pathFilenameNoNumba = pathFilename.with_name(pathFilename.stem + 'NoNumba' + pathFilename.suffix)
        else:
            continue
        codeNoNumba = pathFilename.read_text()
        for codeLine in copy.copy(codeNoNumba.splitlines()):
            if 'numba' in codeLine:
                codeNoNumba = codeNoNumba.replace(codeLine, '')
        pathFilenameNoNumba.write_text(codeNoNumba)
        listPathFilenamesNoNumba.append(pathFilenameNoNumba)

identifierCallableLaunch = "goGoGadgetAbsurdity"

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

def writeModuleWithNumba(listDimensions, datatypeDefault: str = 'uint8'):
    numpy_dtypeLarge = dtypeLarge
    #, datatypeDefault: str = 'uint8'
    # datatypeDefault = 'uint8'
    numpy_dtypeDefault = make_dtype(datatypeDefault)
    numpy_dtypeSmall = numpy_dtypeDefault
    # forceinline=True might actually be useful
    parametersNumba = f"numba.types.{datatypeLarge}(), \
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

    pathFilenameData = Z0Z_makeJob(listDimensions, datatypeDefault=numpy_dtypeDefault, datatypeLarge=numpy_dtypeLarge, datatypeSmall=numpy_dtypeSmall)

    pathFilenameAlgorithm = pathlib.Path('/apps/mapFolding/mapFolding/someAssemblyRequired/countSequentialNoNumba.py')  # Switch back to generated module
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
            , ImaIndent + f"foldsTotal = numba.types.{datatypeLarge}(0)"
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

def writeModuleLLVM(pathFilenamePythonFile: pathlib.Path) -> pathlib.Path:
    pathRootPackage = pathlib.Path('c:/apps/mapFolding')
    relativePathModule = pathFilenamePythonFile.relative_to(pathRootPackage)
    moduleTarget = '.'.join(relativePathModule.parts)[0:-len(relativePathModule.suffix)]
    moduleTargetImported = importlib.import_module(moduleTarget)
    linesLLVM = moduleTargetImported.__dict__[identifierCallableLaunch].inspect_llvm()[()]
    moduleLLVM = llvmlite.binding.module.parse_assembly(linesLLVM)
    pathFilenameLLVM = pathFilenamePythonFile.with_suffix(".ll")
    pathFilenameLLVM.write_text(str(moduleLLVM))
    return pathFilenameLLVM

def doIt(listDimensions, datatypeDefault: str = 'uint8'):
    # NOTE this overwrites files
    Z0Z_inlineMapFolding()
    pathFilenamePythonFile = writeModuleWithNumba(listDimensions, datatypeDefault=datatypeDefault)
    pathFilenameLLVM = writeModuleLLVM(pathFilenamePythonFile)  # noqa: F841
    return pathFilenamePythonFile

if __name__ == '__main__':
    doIt([2, 8])
    # doIt([2]*2, datatypeDefault='int64')
