"""OEIS."""
from __future__ import annotations

from mapFolding.oeis._byID import oeisIDfor_n as oeisIDfor_n
from mapFolding.oeis._commandLine import getOEISids as getOEISids, OEIS_for_n as OEIS_for_n
from mapFolding.oeis._meanders import countMeanders as countMeanders
from mapFolding.oeis._metadata import makeDictionaryFoldsTotalKnown as makeDictionaryFoldsTotalKnown
