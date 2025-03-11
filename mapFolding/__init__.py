"""Map/stamp folding pattern enumeration."""

from mapFolding.basecamp import countFolds
from mapFolding.oeis import clearOEIScache, getOEISids, OEIS_for_n

__all__ = [
    'clearOEIScache',
	'countFolds',
    'getOEISids',
    'OEIS_for_n',
]
