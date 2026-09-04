"""A semi-automated process for finding and computing values not published on OEIS.

It's a pain to update more than a few sequences, so increasing the automation would be wise. There is
a Pytest function that compares uses "unknown value" field from the metadata to try to generate an
Exception by asking for the "unknown value." (The test has a mark to prevent it from running on
GitHub.) The lack of exception indicates a mismatch between the OEIS b-file in oeis/.cache and the
putative unknown value in the metadata.

The manual process includes running, updating this script to select which values to compute and
copy-pasting the computed values in the b-file.
"""
from __future__ import annotations

from hunterMakesPy import CallableFunction, errorL33T
from mapFolding import ansiColorReset, ansiColors
from mapFolding.oeis import getValuesKnown
from mapFolding.oeis._byFormulaLookup import (
	A000682, A060206, A077014, A077460, A085973, A208357, A217310, A217318, A223093, A223094, A223095, A259702, A333971, A334615, A337581)
from mapFolding.oeis._theSSOT import pathCache
from typing import TYPE_CHECKING
import sys

if TYPE_CHECKING:
	from pathlib import Path
	from typing import LiteralString

if __name__ == '__main__':
	def _write() -> None:
		sys.stdout.write(
			f"{(match := countTotal == getValuesKnown(oeisID).get(n, -errorL33T))}\t"
			f"{(ansiColors.YellowOnRed, ansiColors.GreenOnBlack)[match]}"
			f"{n}\t"
			f"{countTotal}\t"
			f"{ansiColorReset}\n"
		)

	oeisID = 'A223094'

	sys.stdout.write(f"{ansiColors[int(oeisID, 36) % len(ansiColors)]}{oeisID} ")
	sys.stdout.write(ansiColorReset + '\n')

	fList: list[LiteralString] = []

	n: int = 46

	for f in fList:
		try:
			print('\n', f)
			countTotal = A223094(n, f)
			_write()
		except KeyError as error:
			print(type(error).__name__)

	qq: list[tuple[CallableFunction[..., int], int, LiteralString]] = [
	# *((A077014, n, 'A005316') for n in range(56, 57)),
	# *((A077014, n, 'A000682 and A223093') for n in range(46, 47)),
	# *((A077014, n, 'A223095, A000136, and A000682') for n in range(46, 47)),
	# *((A077460, n, 'A005315, A005316, and A060206') for n in range(29, 30)),
	# *((A085973, n, 'A077054 and A005315') for n in range(28, 29)),
	# *((A208357, n, 'A005315') for n in range(28, 29)),
	# *((A259702, n, 'A000682') for n in range(46, 47)),
	# *((A333971, n, 'A000682') for n in range(47, 48)),
	# *((A334615, n, 'A000682') for n in range(46, 47)),
	# *((A334615, n, 'A301620') for n in range(46, 47)),
	# *((A223093, n, 'A000682 and A077014') for n in range(45, 46)),
	# *((A223095, n, 'A000136, A077014, and A000682') for n in range(45, 46)),
	# *((A217318, n, 'A223095 and A000034') for n in range(45, 50)),
	# *((A217310, n, 'A223093') for n in range(45, 50)),
	# *((A077460, n, 'A005316, A005315, and A060206') for n in range(30, 60)),
	# *((A000682, n, 'A077460, A005316, and A000560') for n in range(47, 60)),
	*((A000682, n, 'A259689') for n in range(28, 60)),
	# *( for n in range(20, 60)),
]

	for callableA, n, f in qq:
		print(callableA.__name__, n, f)
		pathFilenameB: Path = pathCache / f"b{callableA.__name__[1:]}.txt"
		aOFn = str(callableA(n, f))
		append: str = str(n) + " " + aOFn + "\n"
		with pathFilenameB.open('a', encoding='utf-8') as streamAppend:
			streamAppend.write(append)
