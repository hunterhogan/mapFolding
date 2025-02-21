from mapFolding import getAlgorithmSource, getPathSyntheticModules
from mapFolding import setDatatypeModule, setDatatypeFoldsTotal, setDatatypeElephino, setDatatypeLeavesTotal
import ast
import inspect
import pathlib

def transformPythonToJAX(codePython: str) -> None:
	astPython = ast.parse(codePython)

def writeJax(*, codeSource: str | None = None, pathFilenameAlgorithm: pathlib.Path | None = None, pathFilenameDestination: pathlib.Path | None = None) -> None:
	if codeSource is None and pathFilenameAlgorithm is None:
		algorithmSource = getAlgorithmSource()
		codeSource = inspect.getsource(algorithmSource)
		transformedText = transformPythonToJAX(codeSource)
		pathFilenameAlgorithm = pathlib.Path(inspect.getfile(algorithmSource))
	else:
		raise NotImplementedError("You haven't written this part yet.")
	if pathFilenameDestination is None:
		pathFilenameDestination = getPathSyntheticModules() / "countJax.py"
	# pathFilenameDestination.write_text(transformedText)

if __name__ == '__main__':
	setDatatypeModule('jax.numpy', sourGrapes=True)
	setDatatypeFoldsTotal('int64', sourGrapes=True)
	setDatatypeElephino('uint8', sourGrapes=True)
	setDatatypeLeavesTotal('uint8', sourGrapes=True)
	writeJax()
