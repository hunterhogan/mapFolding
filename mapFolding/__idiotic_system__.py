"""
Phyck Python. A module to cope with incomplete imports: primarily used in the pytest modules.
I hope there is a better way to do this, and I want to learn it. Whatever the case, phyck you, Python:
phyck your horrible documentation, your dogma, your outdated style conventions, and most especially,
your PyConceit.
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
    '_formatFilenameCache',
    '_getOEISidValues',
    '_parseBFileOEIS',
    '_validateOEISid',
    'getOEISids',
    'oeisIDfor_n',
    'oeisIDsImplemented',
    'settingsOEIS',
]
