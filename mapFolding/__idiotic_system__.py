"""
phuck Python
"""

from .oeis import settingsOEISsequences, _validateOEISid, oeisSequence_aOFn, _getOEISidValues, _parseBFileOEIS, getOEISids, _formatFilenameCache, OEISsequenceID

__all__ = [
    'settingsOEISsequences', '_validateOEISid','oeisSequence_aOFn', '_getOEISidValues',
    '_parseBFileOEIS','getOEISids', '_formatFilenameCache',
    'OEISsequenceID'
]
