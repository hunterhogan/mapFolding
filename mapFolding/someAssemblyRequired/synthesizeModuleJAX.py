from mapFolding import getAlgorithmSource, relativePathSyntheticModules
from mapFolding import setDatatypeModule, setDatatypeFoldsTotal, setDatatypeElephino, setDatatypeLeavesTotal
from typing import Optional
import ast
import inspect
import pathlib
import sys

def transformPythonToJAX(codePython: str):
    astPython = ast.parse(codePython)

def writeJax(*, codeSource: Optional[str] = None, pathFilenameAlgorithm: Optional[pathlib.Path] = None, pathFilenameDestination: Optional[pathlib.Path] = None) -> None:
    if codeSource is None and pathFilenameAlgorithm is None:
        algorithmSource = getAlgorithmSource()
        codeSource = inspect.getsource(algorithmSource)
        transformedText = transformPythonToJAX(codeSource)
        pathFilenameAlgorithm = pathlib.Path(inspect.getfile(algorithmSource))
    else:
        raise NotImplementedError("You haven't written this part yet.")
    if pathFilenameDestination is None:
        pathFilenameDestination = pathFilenameAlgorithm.parent / relativePathSyntheticModules / "countJax.py"
    # pathFilenameDestination.write_text(transformedText)

if __name__ == '__main__':
    setDatatypeModule('jax.numpy', sourGrapes=True)
    setDatatypeFoldsTotal('int64', sourGrapes=True)
    setDatatypeElephino('uint8', sourGrapes=True)
    setDatatypeLeavesTotal('uint8', sourGrapes=True)
    writeJax()
