# TODO Implement this module while on linux.
from __future__ import annotations

from typing import TYPE_CHECKING
import lief
import sys

if TYPE_CHECKING:
	from pathlib import Path

def binaryStrip(pathFilename: Path) -> Path:
	binary = lief.parse(pathFilename)
	binary.strip()
	binary.write(pathFilename)
	return pathFilename


def toCodon(pathFilenamePython: Path) -> Path:
	if sys.platform == 'linux':
		commandBuild: list[str] = ['codon', 'build', '--exe', '--release', '--mcpu=native'
			, '--fast-math', '--enable-unsafe-fp-math', '--disable-exceptions'
			, '-o', str(pathFilenamePython.with_suffix(''))
			, str(pathFilenamePython)
		]

		subprocess.run(commandBuild, check=False)
		pathFilenameBinary = binaryStrip(pathFilenamePython.with_suffix(''))

		sys.stdout.write(f"sudo systemd-run --unit={pathFilenameBinary.parent.name} --nice=-10 --property=CPUAffinity=0 {pathFilenameBinary}\n")

	else:
		message: str = f"Python says {sys.platform = }, and I need 'linux'."
		raise OSError(message)

	return pathFilenameBinary
