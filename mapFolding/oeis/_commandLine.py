from __future__ import annotations

from mapFolding.oeis._byID import oeisIDfor_n
from mapFolding.oeis._metadata import dictionaryOEIS
import argparse
import sys
import time

def _CLIHelpText() -> str:
	"""I use this to share sequence listings and usage examples between console commands.

	(AI generated docstring)

	This function combines the registered sequence descriptions with examples of the `OEIS_for_n`
	command and `mapFolding.oeis.oeisIDfor_n` [1].

	Returns
	-------
	helpText : str
		A formatted string containing complete usage information and examples.

	References
	----------
	[1] `mapFolding.oeis.oeisIDfor_n`

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
		f"\ttotalFolds = oeisIDfor_n('{exampleOEISid}', {exampleN})"
	)

def _getOEISDescriptions() -> str:
	"""I use this to insert registered sequence descriptions into shared help text.

	(AI generated docstring)

	This function formats identifiers and descriptions from `mapFolding.oeis._metadata.dictionaryOEIS`
	[1] in registry insertion order.

	Returns
	-------
	sequenceInfo : str
		A formatted string listing each OEIS sequence ID with its description.

	References
	----------
	[1] `mapFolding.oeis._metadata.dictionaryOEIS`

	"""
	return "\n".join(
		f"  {oeisID}: {dictionaryOEIS[oeisID]['description']}"
		for oeisID in dictionaryOEIS
	)

def OEIS_for_n() -> None:
	"""Calculate a sequence term from a command-line identifier and index.

	(AI generated docstring)

	You can run the `OEIS_for_n` console command to evaluate one term with
	`mapFolding.oeis.oeisIDfor_n` [1]. This function reads a sequence identifier and integer index
	from command-line arguments, writes the result to standard output with the literal label
	`distinct folding patterns.`, and writes elapsed time on the next line.

	Command Arguments
	-----------------
	oeisID : str
		Identifier of the sequence to evaluate.
	n : int
		Index of the sequence term to evaluate.

	Exit Status
	-----------
	Calculation errors of type `KeyError`, `ValueError`, or `ArithmeticError` produce an error
	message on standard error and exit status one. `argparse` [2] handles invalid arguments and
	`--help` before calculation; invalid arguments exit with status two, and help exits with zero.

	Examples
	--------
	The shared console help uses this calculation command.

		```console
		OEIS_for_n A001415 6
		```

	References
	----------
	[1] `mapFolding.oeis.oeisIDfor_n`

	[2] argparse - Python standard library.
		https://docs.python.org/3/library/argparse.html
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
	"""Display registered OEIS sequence descriptions and usage examples.

	(AI generated docstring)

	You can call this function or run the `getOEISids` console command to inspect the sequences
	registered in `mapFolding.oeis._metadata.dictionaryOEIS` [1]. This function writes identifiers,
	descriptions, and command-line and Python usage examples to standard output.

	References
	----------
	[1] `mapFolding.oeis._metadata.dictionaryOEIS`


	"""
	sys.stdout.write(_CLIHelpText())
