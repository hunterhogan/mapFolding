"""OEIS."""
from __future__ import annotations

from mapFolding.oeis._theSSOT import oeisIDsImplemented as oeisIDsImplemented, oeisIDsMapFoldingImplemented as oeisIDsMapFoldingImplemented

# isort: split
from mapFolding.oeis._metadata import getMetadata as getMetadata, getValuesKnown as getValuesKnown

# isort: split
from mapFolding.oeis.__main__ import makeDictionaryFoldsTotalKnown as makeDictionaryFoldsTotalKnown, makeMapShape as makeMapShape

# isort: split
from mapFolding.oeis._byID import oeisIDfor_n as oeisIDfor_n

# isort: split
from mapFolding.oeis._commandLine import getOEISids as getOEISids, OEIS_for_n as OEIS_for_n
