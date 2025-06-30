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

Example of successful use:
The LLVM IR for the groundbreaking 2x19 map calculation can be found at:
mapFolding/reference/jobsCompleted/[2x19]/[2x19].ll

This file demonstrates the low-level optimizations that made this previously
intractable calculation possible. The IR reveals how the abstract algorithm was
transformed into efficient machine code through Numba's compilation assembly-line.

While originally part of a tighter integration with the code generation assembly-line,
this module now operates as a standalone utility that can be applied to any module
containing Numba-compiled functions.
"""
from pathlib import Path
from typing import TYPE_CHECKING
import importlib.util
import llvmlite.binding

if TYPE_CHECKING:
	from importlib.machinery import ModuleSpec
	from types import ModuleType

def writeModuleLLVM(pathFilename: Path, identifierCallable: str) -> Path:
	"""Import the generated module directly and get its LLVM IR.

	Parameters
	----------
	pathFilename : Path
		Path to the Python module file containing the Numba-compiled function
	identifierCallable : str
		Name of the function within the module to extract LLVM IR from

	Returns
	-------
	pathFilenameLLVM : Path
		Path to the generated .ll file containing the extracted LLVM IR

	For an example of the output, see reference/jobsCompleted/[2x19]/[2x19].ll,
	which contains the IR for the historically significant 2x19 map calculation.
	"""
	specTarget: ModuleSpec | None = importlib.util.spec_from_file_location("generatedModule", pathFilename)
	if specTarget is None or specTarget.loader is None:
		message = f"Could not create module spec or loader for {pathFilename}"
		raise ImportError(message)
	moduleTarget: ModuleType = importlib.util.module_from_spec(specTarget)
	specTarget.loader.exec_module(moduleTarget)

	# Get LLVM IR and write to file
	linesLLVM = moduleTarget.__dict__[identifierCallable].inspect_llvm()[()]
	moduleLLVM: llvmlite.binding.ModuleRef = llvmlite.binding.module.parse_assembly(linesLLVM)
	pathFilenameLLVM: Path = pathFilename.with_suffix(".ll")
	pathFilenameLLVM.write_text(str(moduleLLVM))
	return pathFilenameLLVM
