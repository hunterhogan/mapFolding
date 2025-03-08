"""Map/stamp folding pattern enumeration."""

from mapFolding.theSSOT import ComputationState as ComputationState

from mapFolding.basecamp import countFolds
from mapFolding.oeis import clearOEIScache, getOEISids, OEIS_for_n

__all__ = [
    'clearOEIScache',
	'countFolds',
    'getOEISids',
    'OEIS_for_n',
]
