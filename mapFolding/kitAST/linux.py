# ruff: ignore[undocumented-public-module]
# DOCUMENT  #
from __future__ import annotations

from hunterMakesPy import raiseIfNone
from typing import TYPE_CHECKING
import anyascii
import lief
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys

if TYPE_CHECKING:
	from pathlib import Path

def binaryStrip(pathFilename: Path) -> Path:  # ruff: ignore[undocumented-public-function]
	# DOCUMENT
	binary: lief.OAT.Binary | lief.ELF.Binary = raiseIfNone(lief.parse(pathFilename))
	binary.strip()
	binary.write(pathFilename)
	return pathFilename

def toCodon(pathFilenamePython: Path) -> Path:  # ruff: ignore[undocumented-public-function]
	# DOCUMENT
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
