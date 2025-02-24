import pathlib

def writeModuleLLVM(pathFilename: pathlib.Path, identifierCallable: str) -> pathlib.Path:
    """Import the generated module directly and get its LLVM IR."""
