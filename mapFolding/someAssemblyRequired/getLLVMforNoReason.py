"""
Utility for extracting LLVM IR from compiled Python modules.

This module provides functionality to extract and save the LLVM Intermediate Representation (IR)
generated when Numba compiles Python functions. It implements a simple interface that:

1. Imports a specified Python module from its file path
2. Extracts the LLVM IR from a specified function within that module
3. Writes the IR to a file with the same base name but with the .ll extension

The extracted LLVM IR can be valuable for debugging, optimization analysis, or educational
purposes, as it provides a view into how high-level Python code is translated into
lower-level representations for machine execution.

While originally part of a tighter integration with the code generation pipeline,
this module now operates as a standalone utility that can be applied to any module
containing Numba-compiled functions.
"""
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
import importlib.util
import llvmlite.binding

def writeModuleLLVM(pathFilename: Path, identifierCallable: str) -> Path:
    """Import the generated module directly and get its LLVM IR."""
    specTarget: ModuleSpec | None = importlib.util.spec_from_file_location("generatedModule", pathFilename)
    if specTarget is None or specTarget.loader is None:
        raise ImportError(f"Could not create module spec or loader for {pathFilename}")
    moduleTarget: ModuleType = importlib.util.module_from_spec(specTarget)
    specTarget.loader.exec_module(moduleTarget)

    # Get LLVM IR and write to file
    linesLLVM = moduleTarget.__dict__[identifierCallable].inspect_llvm()[()]
    moduleLLVM: llvmlite.binding.ModuleRef = llvmlite.binding.module.parse_assembly(linesLLVM)
    pathFilenameLLVM: Path = pathFilename.with_suffix(".ll")
    pathFilenameLLVM.write_text(str(moduleLLVM))
    return pathFilenameLLVM
