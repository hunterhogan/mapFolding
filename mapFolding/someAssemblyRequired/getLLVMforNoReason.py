import importlib
import importlib.util
import llvmlite.binding
import pathlib

def writeModuleLLVM(pathFilename: pathlib.Path, identifierCallable: str) -> pathlib.Path:
    """Import the generated module directly and get its LLVM IR."""
    specTarget = importlib.util.spec_from_file_location("generatedModule", pathFilename)
    if specTarget is None or specTarget.loader is None:
        raise ImportError(f"Could not create module spec or loader for {pathFilename}")
    moduleTarget = importlib.util.module_from_spec(specTarget)
    specTarget.loader.exec_module(moduleTarget)

    # Get LLVM IR and write to file
    linesLLVM = moduleTarget.__dict__[identifierCallable].inspect_llvm()[()]
    moduleLLVM = llvmlite.binding.module.parse_assembly(linesLLVM)
    pathFilenameLLVM = pathFilename.with_suffix(".ll")
    pathFilenameLLVM.write_text(str(moduleLLVM))
    return pathFilenameLLVM
