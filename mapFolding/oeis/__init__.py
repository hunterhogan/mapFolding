"""OEIS."""

from __future__ import annotations

from mapFolding.oeis._commandLine import getOEISids as getOEISids, OEIS_for_n as OEIS_for_n
from mapFolding.oeis._metadata import dictionaryOEIS as dictionaryOEIS, dictionaryOEISMapFolding as dictionaryOEISMapFolding
from mapFolding.oeis._needsAHome import oeisIDfor_n as oeisIDfor_n
from mapFolding.oeis._probablyNotOEIS import (
	countingMeanders as countingMeanders, getFoldsTotalKnown as getFoldsTotalKnown,
	makeDictionaryFoldsTotalKnown as makeDictionaryFoldsTotalKnown)
