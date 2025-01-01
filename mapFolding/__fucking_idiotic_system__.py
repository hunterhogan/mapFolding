"""
phuck Python
"""

from .beDRY import getLeavesTotal, validateListDimensions, outfitFoldings, validateTaskDivisions
from .oeis import settingsOEISsequences, _validateOEISid, oeisSequence_aOFn, _getOEISidValues, _parseBFileOEIS, getOEISids, _formatFilenameCache, OEISsequenceID

__all__ = [
    'getLeavesTotal', 
    'validateListDimensions', 
    'outfitFoldings', 
    'validateTaskDivisions',
    'settingsOEISsequences', '_validateOEISid','oeisSequence_aOFn', '_getOEISidValues',
    '_parseBFileOEIS','getOEISids', '_formatFilenameCache',
    'OEISsequenceID'
]
