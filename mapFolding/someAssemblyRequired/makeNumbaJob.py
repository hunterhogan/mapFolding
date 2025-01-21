"""Create a python module hardcoded to compute a map's foldsTotal.
NumPy ndarray.
Numba optimized.
Absolutely no other imports.
"""
from mapFolding import datatypeLarge, dtypeLarge, dtypeDefault
from mapFolding.someAssemblyRequired.inlineAfunction import Z0Z_inlineMapFolding
from mapFolding.someAssemblyRequired.jobsAndTasks import Z0Z_makeJob
import importlib
import llvmlite.binding
import numpy
import pathlib
import pickle

listDimensions = [3,7]

# NOTE this overwrites files
Z0Z_inlineMapFolding()

identifierCallableLaunch = "goGoGadgetAbsurdity"

def archivistFormatsArrayToCode(arrayTarget: numpy.ndarray, identifierName: str) -> str:
    """Format numpy array into a code string that recreates the array."""
    arrayAsTypeStr = numpy.array2string(arrayTarget, threshold=10000, max_line_width=200, separator=',')
    return f"{identifierName} = numpy.array({arrayAsTypeStr}, dtype=numpy.{arrayTarget.dtype})"

def writeModuleWithNumba(listDimensions):
    numpy_dtypeLarge = dtypeLarge
    numpy_dtypeDefault = dtypeDefault

    parametersNumba = f"numba.types.{datatypeLarge}(), cache=True, parallel=False, boundscheck=False, \
        error_model='numpy', fastmath=True, nogil=True, nopython=True, _nrt=True, forceinline=True, \
            inline=True, looplift=True, no_cfunc_wrapper=False, no_cpython_wrapper=False"

    pathFilenameData = Z0Z_makeJob(listDimensions, datatypeDefault=numpy_dtypeDefault, datatypeLarge=numpy_dtypeLarge)

    pathFilenameAlgorithm = pathlib.Path('/apps/mapFolding/mapFolding/countSequentialNoNumba.py')
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
            , ImaIndent + archivistFormatsArrayToCode(stateJob['my'], 'my')
            , ImaIndent + archivistFormatsArrayToCode(stateJob['foldsSubTotals'], 'foldsSubTotals')
            , ImaIndent + archivistFormatsArrayToCode(stateJob['gapsWhere'], 'gapsWhere')
            , ImaIndent + archivistFormatsArrayToCode(stateJob['track'], 'track')
            ])

    linesDataStatic = """"""
    linesDataStatic = "\n".join([linesDataStatic
            , ImaIndent + archivistFormatsArrayToCode(stateJob['the'], 'the')
            , ImaIndent + archivistFormatsArrayToCode(stateJob['connectionGraph'], 'connectionGraph')
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

    lineReturn = f"{ImaIndent}return foldsSubTotals.sum().item()"

    linesLaunch = """"""
    linesLaunch = linesLaunch + f"""
if __name__ == '__main__':
    foldsTotal = {identifierCallableLaunch}()"""

    linesWriteFoldsTotal = """"""
    linesWriteFoldsTotal = "\n".join([linesWriteFoldsTotal
                                    , "    print(foldsTotal)"
                                    , f"    open('{pathFilenameFoldsTotal.as_posix()}', 'w').write(str(foldsTotal))"
                                    ])

    linesAll = "\n".join([
                        linesImport
                        , linesAlgorithm
                        , f"{ImaIndent}print(foldsSubTotals.sum().item())"
                        , lineReturn
                        , linesLaunch
                        , linesWriteFoldsTotal
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

if __name__ == '__main__':
    pathFilenamePythonFile = writeModuleWithNumba(listDimensions)
    pathFilenameLLVM = writeModuleLLVM(pathFilenamePythonFile)
