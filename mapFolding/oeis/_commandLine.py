from __future__ import annotations

from mapFolding.oeis import _theSSOT
from mapFolding.oeis._metadata import dictionaryOEISMapFolding
from mapFolding.oeis._needsAHome import oeisIDfor_n
import argparse
import sys
import time

def _CLIHelpText() -> str:
	"""Format comprehensive help text for both command-line and interactive use.

	(AI generated docstring)

	This function generates standardized help documentation that includes all available OEIS sequences
	with their descriptions and provides usage examples for both command-line and programmatic
	interfaces.

	Returns
	-------
	helpText : str
		A formatted string containing complete usage information and examples.
	"""
	exampleOEISid: str = 'A001415'
	exampleN: int = 6

	return (
		"\nAvailable OEIS sequences:\n"
		f"{_getOEISDescriptions()}\n"
		"\nUsage examples:\n"
		"  Command line:\n"
		f"\tOEIS_for_n {exampleOEISid} {exampleN}\n"
		"  Python:\n"
		"\tfrom mapFolding.oeis import oeisIDfor_n\n"
		f"\tfoldsTotal = oeisIDfor_n('{exampleOEISid}', {exampleN})"
	)

def _getOEISDescriptions() -> str:
	"""Format information about available OEIS sequences for display in help messages and error output.

	(AI generated docstring)

	This function creates a standardized listing of all implemented OEIS sequences with their
	mathematical descriptions, suitable for inclusion in help text and error messages.

	Returns
	-------
	sequenceInfo : str
		A formatted string listing each OEIS sequence ID with its description.
	"""
	return "\n".join(
		f"  {oeisID}: {dictionaryOEISMapFolding[oeisID]['description']}"
		for oeisID in _theSSOT.oeisIDsImplementedMapFolding
	)

def OEIS_for_n() -> None:
	"""Command-line interface for calculating OEIS sequence values.

	(AI generated docstring)

	This function provides a command-line interface to the `oeisIDfor_n` function, enabling users to
	calculate specific values of implemented OEIS sequences from the terminal. It includes argument
	parsing, error handling, and performance timing to provide a complete user experience.

	The function accepts two command-line arguments: an OEIS sequence identifier and an integer index,
	then outputs the calculated sequence value along with execution time. Error messages are directed
	to stderr with appropriate exit codes for shell scripting integration.

	Usage
	-----
	python -m mapFolding.oeis OEIS_for_n A001415 10
	"""
	parserCLI: argparse.ArgumentParser = argparse.ArgumentParser(
		description="Calculate a(n) for an OEIS sequence.",
		epilog=_CLIHelpText(),
		formatter_class=argparse.RawDescriptionHelpFormatter
	)
	parserCLI.add_argument('oeisID', help="OEIS sequence identifier")
	parserCLI.add_argument('n', type=int, help="Calculate a(n) for this n")

	argumentsCLI: argparse.Namespace = parserCLI.parse_args()

	timeStart: float = time.perf_counter()

	try:
		sys.stdout.write(f"{oeisIDfor_n(argumentsCLI.oeisID, argumentsCLI.n)} distinct folding patterns.\n")
	except (KeyError, ValueError, ArithmeticError) as ERRORmessage:
		sys.stderr.write(f"Error: {ERRORmessage}\n")
		sys.exit(1)

	timeElapsed: float = time.perf_counter() - timeStart
	sys.stdout.write(f"Time elapsed: {timeElapsed:.3f} seconds\n")

def getOEISids() -> None:
	"""Display comprehensive information about all implemented OEIS sequences.

	(AI generated docstring)

	This function serves as the primary help interface for the module, displaying detailed information
	about all directly implemented OEIS sequences along with usage examples for both command-line and
	programmatic interfaces. It provides users with a complete overview of available sequences and
	their mathematical meanings.

	The output includes sequence identifiers, mathematical descriptions, and practical usage examples
	to help users understand how to access and utilize the OEIS interface functionality.

	"""
	sys.stdout.write(_CLIHelpText())
