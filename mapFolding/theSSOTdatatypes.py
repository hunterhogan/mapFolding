"""module for prototyping/developing a new system for datatype management."""
from numpy import integer
# import lazy_object_proxy
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

# datatypeLargeDEFAULT: Final[str] = 'int64'
# datatypeMediumDEFAULT: Final[str] = 'uint8'
# datatypeSmallDEFAULT: Final[str] = 'uint8'
# datatypeModuleDEFAULT: Final[str] = 'numpy'

# def make_dtype(datatype: str, datatypeModule: Optional[str] = None) -> Type[Any]:
#     if datatypeModule is None:
#         datatypeModule = datatypeModuleDEFAULT
#     return eval(f"{datatypeModule}.{datatype}")

# dtypeLargeDEFAULT = make_dtype(datatypeLargeDEFAULT)
# dtypeMediumDEFAULT = make_dtype(datatypeMediumDEFAULT)
# dtypeSmallDEFAULT = make_dtype(datatypeSmallDEFAULT)

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

# dtypeElephino = lazy_object_proxy.Proxy(_make_dtypeElephinoYouLazyBum)
# dtypeFoldsTotal = lazy_object_proxy.Proxy(_make_dtypeFoldsTotalYouLazyBum)
# dtypeLeavesTotal = lazy_object_proxy.Proxy(_make_dtypeLeavesTotalYouLazyBum)

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
    # if Rube == 'dtypeElephino':
    #     GoldBerg = dtypeElephino
    # elif Rube == 'dtypeFoldsTotal':
    #     GoldBerg = dtypeFoldsTotal
    # elif Rube == 'dtypeLeavesTotal':
    #     GoldBerg = dtypeLeavesTotal
    if Rube == 'dtypeElephino':
        GoldBerg = _make_dtypeElephinoYouLazyBum()
    elif Rube == 'dtypeFoldsTotal':
        GoldBerg = _make_dtypeFoldsTotalYouLazyBum()
    elif Rube == 'dtypeLeavesTotal':
        GoldBerg = _make_dtypeLeavesTotalYouLazyBum()
    return GoldBerg
