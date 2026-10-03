from __future__ import annotations

from contextlib import suppress
from hunterMakesPy.filesystemToolkit import appendStringToHere
from inspect import getmembers, isfunction
from mapFolding.oeis import _byFormulaLookup, getMetadata, getValuesKnown, oeisIDsImplemented
from mapFolding.oeis._theSSOT import pathCache
from typing import get_args, get_type_hints

if __name__ == '__main__':
	keepGoing: bool = True
	while keepGoing:
		keepGoing = False
		#=SIN= For loop: attempt every callable on each pass.
		for oeisID, callableA in getmembers(_byFormulaLookup, isfunction):
			if oeisID in oeisIDsImplemented:
				valuesKnown: dict[int, int] = getValuesKnown(oeisID)
				n: int = getMetadata(oeisID)['valueUnknown']
				#=SIN= For loop: every alternative formula must be attempted to find computable values.
				for f in get_args(get_args(get_type_hints(callableA)['f'])[0]):
					#=SIN= KeyError suppression: missing prerequisites must allow the remaining callables to run.
					with suppress(KeyError):
						countTotal: int = callableA(n, f)
						#=SIN= Filesystem operations outside kitFilesystem.
						appendStringToHere(f'{n} {countTotal}\n', pathCache / f'b{oeisID[1:]}.txt')
						valuesKnown[n] = countTotal
						print(oeisID, n, f, countTotal)
						n += 1
						# keepGoing = True
