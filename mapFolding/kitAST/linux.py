"""Compile generated computation jobs and strip their Linux binaries.

(AI generated docstring)

You can use this module to prepare generated source for Codon [1], build an executable,
and rewrite the executable with LIEF [2]. Source preparation and binary stripping replace
the supplied files. Compilation requires Linux and an installed `codon` command.

Contents
--------
Functions
	binaryStrip
		Strip a supported binary in place and return the original path.
	toCodon
		Rewrite generated source, compile on Linux, and print launch commands.

References
----------
[1] Codon compiler.
	https://github.com/exaloop/codon
[2] LIEF binary rewriting API.
	https://lief.re/doc/latest/formats/elf/python.html
"""
from __future__ import annotations

from hunterMakesPy import raiseIfNone
from typing import TYPE_CHECKING
import anyascii
import lief
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys

if TYPE_CHECKING:
	from pathlib import Path

def binaryStrip(pathFilename: Path) -> Path:
	"""Strip the binary at `pathFilename` in place and return the original path.

	(AI generated docstring)

	This function asks LIEF [1] to parse and strip the supplied binary, then writes the
	modified binary over `pathFilename`. The function does not create a backup.

	Parser Failure
	--------------
	If the binary parser returns `None`, the required-value check raises `ValueError`
	before the function attempts to strip or write the binary.

	Parameters
	----------
	pathFilename : Path
		Existing binary that LIEF can parse, strip, and rewrite [1]. The compilation flow
		uses this function for its Linux executable.

	Returns
	-------
	pathFilename : Path
		The same path object supplied by the caller after rewriting the binary.

	Examples
	--------
	`toCodon` [2] strips the output whose filename is the source path without its suffix.

		```python
		pathFilenameBinary: Path = binaryStrip(pathFilenamePython.with_suffix(''))
		```

	References
	----------
	[1] LIEF binary parsing, stripping, and rewriting API.
		https://lief.re/doc/latest/formats/elf/python.html
	[2] `toCodon`

	"""
	binary: lief.OAT.Binary | lief.ELF.Binary = raiseIfNone(lief.parse(pathFilename))
	binary.strip()
	binary.write(pathFilename)
	return pathFilename

def toCodon(pathFilenamePython: Path) -> Path:
	"""Rewrite generated source at `pathFilenamePython` and compile a Linux executable.

	(AI generated docstring)

	This function prepares a generated computation job for Codon [1], compiles the job,
	and strips the executable with `binaryStrip` [2]. The function returns the executable
	path and prints two suggested launch commands without executing either command.

	Source Rewriting
	----------------
	The function removes the first 36 decoded characters without checking their contents,
	transliterates the entire remainder with `anyascii` [3], and overwrites the source using
	ASCII encoding. Repeating the call removes another 36 characters. The rewrite occurs
	before the platform check, and no backup or rollback is provided.

	Compilation and Launch Output
	-----------------------------
	The `codon` command must be available through the process search path. The build requests
	an executable in release mode, targets the native CPU, enables fast and unsafe floating
	point optimization, and disables exceptions [1]. `subprocess.run` uses `check=False` [4].
	A nonzero compiler exit status therefore does not stop the subsequent attempt to strip
	the output path; an existing executable at that path can be stripped after a failed build.

	Parameters
	----------
	pathFilenamePython : Path
		Writable UTF-8 source file with a disposable 36-character prefix. The remaining
		source must be suitable for compilation after transliteration. The executable
		uses the same path with the final suffix removed.

	Returns
	-------
	pathFilenameBinary : Path
		Path of the executable after stripping. Returning does not independently verify
		that this invocation of the compiler produced the executable.

	Raises
	------
	OSError
		If the platform is not Linux, after the source file has already been rewritten.
		Process startup and file access failures also propagate.

	Examples
	--------
	The job writer calls this function after writing its source and checking for Linux [5].

		```python
		toCodon(Path(job.pathFilenameModule))
		```

	References
	----------
	[1] Codon compiler and executable builds.
		https://github.com/exaloop/codon
	[2] `binaryStrip`

	[3] AnyAscii transliteration.
		https://github.com/anyascii/anyascii
	[4] Python `subprocess.run` and exit status handling.
		https://docs.python.org/3/library/subprocess.html#subprocess.run
	[5] `mapFolding.kitAST.codon.makeJob.makeJob`

	"""
	pathFilenamePython.write_text(anyascii.anyascii(pathFilenamePython.read_text(encoding='utf-8')[36:None]), encoding='ascii')
	if sys.platform == 'linux':
		commandBuild: list[str] = ['codon', 'build', '--exe', '--release', '--mcpu=native'
			, '--fast-math', '--enable-unsafe-fp-math', '--disable-exceptions'
			, '-o', str(pathFilenamePython.with_suffix(''))
			, str(pathFilenamePython)
		]

		subprocess.run(commandBuild, check=False)
		pathFilenameBinary: Path = binaryStrip(pathFilenamePython.with_suffix(''))

		sys.stdout.write(f"sudo systemd-run --unit={pathFilenameBinary.name} --nice=-10 {pathFilenameBinary}\n")
		sys.stdout.write(f"sudo nice -n -10 {pathFilenameBinary}\n")

	else:
		message: str = f"Python says {sys.platform = }, and I need 'linux'."
		raise OSError(message)

	return pathFilenameBinary
