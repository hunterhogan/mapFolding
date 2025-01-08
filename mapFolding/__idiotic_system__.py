"""
phuck Python
"""

from .oeis import OEIS_for_n
from .oeis import _formatFilenameCache
from .oeis import _getOEISidValues
from .oeis import _parseBFileOEIS
from .oeis import _validateOEISid
from .oeis import getOEISids
from .oeis import oeisIDfor_n
from .oeis import oeisIDsImplemented
from .oeis import settingsOEIS

__all__ = [
    'OEIS_for_n',
    'oeisIDsImplemented',
    '_formatFilenameCache',
    '_getOEISidValues',
    '_parseBFileOEIS',
    '_validateOEISid',
    'getOEISids',
    'oeisIDfor_n',
    'settingsOEIS',
]
