"""
phuck Python
"""

from .oeis import OEISsequenceID
from .oeis import _formatFilenameCache
from .oeis import _getOEISidValues
from .oeis import _parseBFileOEIS
from .oeis import _validateOEISid
from .oeis import getOEISids
from .oeis import oeisSequence_aOFn
from .oeis import settingsOEISsequences

__all__ = [
    'OEISsequenceID',
    '_formatFilenameCache',
    '_getOEISidValues',
    '_parseBFileOEIS',
    '_validateOEISid',
    'getOEISids',
    'oeisSequence_aOFn',
    'settingsOEISsequences',
]
