"""module for prototyping/developing a new system for datatype management."""
from numpy import integer
from typing import Any, Callable, Final, Optional, Tuple, Type, TypedDict
from types import ModuleType
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

_datatypeElephino = ''
_datatypeFoldsTotal = ''
_datatypeLeavesTotal = ''
_datatypeModule = ''
_datatypeFoldsTotalDEFAULT: Final[str] = 'int64'
_datatypeElephinoDEFAULT: Final[str] = 'uint8'
_datatypeLeavesTotalDEFAULT: Final[str] = 'uint8'
_datatypeModuleDEFAULT: Final[str] = 'numpy'

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
    global _datatypeElephino
    if not _datatypeElephino:
        _datatypeElephino = datatype
    elif _datatypeElephino == datatype:
        pass
    elif sourGrapes:
        raise Exception(f"Datatype is '{_datatypeElephino}' not '{datatype}', so you can take your ball and go home.")
    return _datatypeElephino

def setDatatypeFoldsTotal(datatype: str, sourGrapes: Optional[bool] = False):
    global _datatypeFoldsTotal
    if not _datatypeFoldsTotal:
        _datatypeFoldsTotal = datatype
    elif _datatypeFoldsTotal == datatype:
        pass
    elif sourGrapes:
        raise Exception(f"Datatype is '{_datatypeFoldsTotal}' not '{datatype}', so you can take your ball and go home.")
    return _datatypeFoldsTotal

def setDatatypeLeavesTotal(datatype: str, sourGrapes: Optional[bool] = False):
    global _datatypeLeavesTotal
    if not _datatypeLeavesTotal:
        _datatypeLeavesTotal = datatype
    elif _datatypeLeavesTotal == datatype:
        pass
    elif sourGrapes:
        raise Exception(f"Datatype is '{_datatypeLeavesTotal}' not '{datatype}', so you can take your ball and go home.")
    return _datatypeLeavesTotal

def _get_datatypeElephino():
    global _datatypeElephino
    if not _datatypeElephino:
        _datatypeElephino = _datatypeElephinoDEFAULT
    return _datatypeElephino

def _get_datatypeFoldsTotal():
    global _datatypeFoldsTotal
    if not _datatypeFoldsTotal:
        _datatypeFoldsTotal = _datatypeFoldsTotalDEFAULT
    return _datatypeFoldsTotal

def _get_datatypeLeavesTotal():
    global _datatypeLeavesTotal
    if not _datatypeLeavesTotal:
        _datatypeLeavesTotal = _datatypeLeavesTotalDEFAULT
    return _datatypeLeavesTotal

def _getDatatypeModule():
    global _datatypeModule
    if not _datatypeModule:
        _datatypeModule = _datatypeModuleDEFAULT
    return _datatypeModule

def _make_dtypeElephinoYouLazyBum():
    return eval(f"{_getDatatypeModule()}.{_get_datatypeElephino()}")

def _make_dtypeFoldsTotalYouLazyBum():
    return eval(f"{_getDatatypeModule()}.{_get_datatypeFoldsTotal()}")

def _make_dtypeLeavesTotalYouLazyBum():
    return eval(f"{_getDatatypeModule()}.{_get_datatypeLeavesTotal()}")

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
        GoldBerg = _get_datatypeElephino()
    elif Rube == 'datatypeFoldsTotal':
        GoldBerg = _get_datatypeFoldsTotal()
    elif Rube == 'datatypeLeavesTotal':
        GoldBerg = _get_datatypeLeavesTotal()
    return GoldBerg
