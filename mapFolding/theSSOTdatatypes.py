"""module for prototyping/developing a new system for datatype management."""
from numpy import integer
from typing import Any, Callable, Dict, Final, Optional, Tuple, Type, TypedDict
from types import ModuleType
from collections import defaultdict
import enum
import numba
import numpy
import numpy.typing
import pathlib
import sys

"delay realization/instantiation until a concrete value is desired"
"moment of truth: when the value is needed, not when the value is defined"

"""What is a (not too complicated, integer) datatype?
    - ecosystem/module
        - must apathy|value|list of values
        - mustn't apathy|value|list of values
    - bit width
        - bits maximum apathy|value
        - bits minimum apathy|value
        - magnitude maximum apathy|value
        - ?magnitude minimum apathy|value
    - signedness apathy|non-negative|non-positive|both
    """

_datatype = defaultdict(str)
_datatypeDefault: Final[Dict[str, str]] = {
    'elephino': 'uint8',
    'foldsTotal': 'int64',
    'leavesTotal': 'uint8',
}
_datatypeModule = ''
_datatypeModuleDEFAULT: Final[str] = 'numpy'

def reportDatatypeLimit(identifier: str, datatype: str, sourGrapes: Optional[bool] = False) -> str:
    global _datatype
    if not _datatype[identifier]:
        _datatype[identifier] = datatype
    elif _datatype[identifier] == datatype:
        pass
    elif sourGrapes:
        raise Exception(f"Datatype is '{_datatype[identifier]}' not '{datatype}', so you can take your ball and go home.")
    return _datatype[identifier]

def setDatatypeModule(datatypeModule: str, sourGrapes: Optional[bool] = False):
    global _datatypeModule
    if not _datatypeModule:
        _datatypeModule = datatypeModule
    elif _datatypeModule == datatypeModule:
        pass
    elif sourGrapes:
        raise Exception(f"Datatype module is '{_datatypeModule}' not '{datatypeModule}', so you can take your ball and go home.")
    return _datatypeModule

def setDatatypeElephino(datatype: str, sourGrapes: Optional[bool] = False):
    identifier = 'elephino'
    return reportDatatypeLimit(identifier, datatype, sourGrapes)

def setDatatypeFoldsTotal(datatype: str, sourGrapes: Optional[bool] = False):
    identifier = 'foldsTotal'
    return reportDatatypeLimit(identifier, datatype, sourGrapes)

def setDatatypeLeavesTotal(datatype: str, sourGrapes: Optional[bool] = False):
    identifier = 'leavesTotal'
    return reportDatatypeLimit(identifier, datatype, sourGrapes)

def _get_datatype(identifier: str) -> str:
    global _datatype
    if not _datatype[identifier]:
        _datatype[identifier] = _datatypeDefault.get(identifier) or _datatypeDefault['foldsTotal']
    return _datatype[identifier]

def _getDatatypeModule():
    global _datatypeModule
    if not _datatypeModule:
        _datatypeModule = _datatypeModuleDEFAULT
    return _datatypeModule

def setInStone(identifier: str):
    # not quite right.
    # for example, 'myPARALLEL' needs to freeze every identifier in indexMy and probably elephino
    # But, hackSSOTdatatype and hackSSOTdtype mediate access, so I might be able to configure those to cascade the freeze
    # Better than that. The freeze will cascade, but there can be sort of two default values for the singleton identifiers
    return eval(f"{_getDatatypeModule()}.{_get_datatype(identifier)}")

def _make_dtypeElephinoYouLazyBum():
    return eval(f"{_getDatatypeModule()}.{_get_datatype('elephino')}")

def _make_dtypeFoldsTotalYouLazyBum():
    return eval(f"{_getDatatypeModule()}.{_get_datatype('foldsTotal')}")

def _make_dtypeLeavesTotalYouLazyBum():
    return eval(f"{_getDatatypeModule()}.{_get_datatype('leavesTotal')}")

def hackSSOTdtype(identifier: str) -> Type[Any]:
    _hackSSOTdtype={
    'connectionGraph': 'dtypeLeavesTotal',
    'dtypeElephino': 'dtypeElephino',
    'dtypeFoldsTotal': 'dtypeFoldsTotal',
    'dtypeLeavesTotal': 'dtypeLeavesTotal',
    'foldGroups': 'dtypeFoldsTotal',
    'gapsWhere': 'dtypeLeavesTotal',
    'gapsWherePARALLEL': 'dtypeLeavesTotal',
    'my': 'dtypeElephino',
    'myPARALLEL': 'dtypeElephino',
    'track': 'dtypeElephino',
    'trackPARALLEL': 'dtypeElephino',
    }
    Rube = _hackSSOTdtype[identifier]
    if Rube == 'dtypeElephino':
        GoldBerg = _make_dtypeElephinoYouLazyBum()
    elif Rube == 'dtypeFoldsTotal':
        GoldBerg = _make_dtypeFoldsTotalYouLazyBum()
    elif Rube == 'dtypeLeavesTotal':
        GoldBerg = _make_dtypeLeavesTotalYouLazyBum()
    return GoldBerg

def hackSSOTdatatype(identifier: str) -> str:
    _hackSSOTdatatype={
    'connectionGraph': 'datatypeLeavesTotal',
    'datatypeElephino': 'datatypeElephino',
    'datatypeFoldsTotal': 'datatypeFoldsTotal',
    'datatypeLeavesTotal': 'datatypeLeavesTotal',
    'dimensionsTotal': 'datatypeLeavesTotal',
    'dimensionsUnconstrained': 'datatypeLeavesTotal',
    'foldGroups': 'datatypeFoldsTotal',
    'gap1ndex': 'datatypeLeavesTotal',
    'gap1ndexCeiling': 'datatypeElephino',
    'gapsWhere': 'datatypeLeavesTotal',
    'gapsWherePARALLEL': 'datatypeLeavesTotal',
    'groupsOfFolds': 'datatypeFoldsTotal',
    'indexDimension': 'datatypeLeavesTotal',
    'indexLeaf': 'datatypeLeavesTotal',
    'indexMiniGap': 'datatypeElephino',
    'leaf1ndex': 'datatypeLeavesTotal',
    'leafConnectee': 'datatypeLeavesTotal',
    'my': 'datatypeElephino',
    'myPARALLEL': 'datatypeElephino',
    'taskDivisions': 'datatypeLeavesTotal',
    'taskIndex': 'datatypeLeavesTotal',
    'track': 'datatypeElephino',
    'trackPARALLEL': 'datatypeElephino',
    }
    Rube = _hackSSOTdatatype[identifier]
    if Rube == 'datatypeElephino':
        GoldBerg = _get_datatype('elephino')
    elif Rube == 'datatypeFoldsTotal':
        GoldBerg = _get_datatype('foldsTotal')
    elif Rube == 'datatypeLeavesTotal':
        GoldBerg = _get_datatype('leavesTotal')
    return GoldBerg
