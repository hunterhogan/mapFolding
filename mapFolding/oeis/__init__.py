"""OEIS."""
from __future__ import annotations

from mapFolding.oeis._metadata import getValuesKnown as getValuesKnown

# isort: split
from mapFolding.oeis._needsAHome import getMapShape as getMapShape, makeDictionaryFoldsTotalKnown as makeDictionaryFoldsTotalKnown

# isort: split
from mapFolding.oeis._byID import oeisIDfor_n as oeisIDfor_n

# isort: split
from mapFolding.oeis._commandLine import getOEISids as getOEISids, OEIS_for_n as OEIS_for_n
