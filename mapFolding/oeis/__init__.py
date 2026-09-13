"""OEIS."""
from __future__ import annotations

from mapFolding.oeis._beDRY import formatBFile as formatBFile
from mapFolding.oeis._theSSOT import oeisIDsImplemented as oeisIDsImplemented, oeisIDsMapFoldingImplemented as oeisIDsMapFoldingImplemented

# isort: split
from mapFolding.oeis._metadata import getMetadata as getMetadata, getValuesKnown as getValuesKnown

# isort: split
from mapFolding.oeis.__main__ import (
	getTotalFoldsKnown as getTotalFoldsKnown, getTriangleDiagonal as getTriangleDiagonal, getTriangleRows as getTriangleRows,
	makeMapShape as makeMapShape, printEasyRunBenchmark as printEasyRunBenchmark, printEasyRunHeader as printEasyRunHeader,
	readBFileDiagonal as readBFileDiagonal, readBFileTriangle as readBFileTriangle)

# isort: split
from mapFolding.oeis._byID import oeisIDfor_n as oeisIDfor_n

# isort: split
from mapFolding.oeis._commandLine import getOEISids as getOEISids, OEIS_for_n as OEIS_for_n
