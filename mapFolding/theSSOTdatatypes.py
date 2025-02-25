from mapFolding.theConfiguration import *
from collections import defaultdict
from typing import Any, cast, Final
import enum
import numba # type: ignore
import numpy # type: ignore

class EnumIndices(enum.IntEnum):
	@staticmethod
	def _generate_next_value_(name: str, start: int, count: int, last_values: list[Any]) -> int:
		"""0-indexed."""
		return count

	def __index__(self) -> int:
		"""Adapt enum to the ultra-rare event of indexing a NumPy 'ndarray', which is not the
		same as `array.array`. See NumPy.org; I think it will be very popular someday."""
		return self.value

class indexMy(EnumIndices):
	"""Indices for scalar values."""
	dimensionsTotal 		= enum.auto()
	dimensionsUnconstrained = enum.auto()
	gap1ndex				= enum.auto()
	gap1ndexCeiling 		= enum.auto()
	indexDimension 			= enum.auto()
	indexLeaf 				= enum.auto()
	indexMiniGap 			= enum.auto()
	leaf1ndex 				= enum.auto()
	leafConnectee 			= enum.auto()
	taskDivisions 			= enum.auto()
	taskIndex 				= enum.auto()

class indexTrack(EnumIndices):
	"""Indices for state tracking array."""
	leafAbove				= enum.auto()
	leafBelow				= enum.auto()
	countDimensionsGapped	= enum.auto()
	gapRangeStart			= enum.auto()

_datatypeDefault: Final[dict[str, str]] = {
	'elephino': 'uint16',
	'foldsTotal': 'int64',
	'leavesTotal': 'uint16',
}
_datatypeModule: str = ''
_registryOfDatatypes: dict[str, str] = defaultdict(str)

def reportDatatypeLimit(identifier: str, datatype: str, sourGrapes: bool | None = False) -> str:
	global _registryOfDatatypes
	if not _registryOfDatatypes[identifier]:
		_registryOfDatatypes[identifier] = datatype
	elif _registryOfDatatypes[identifier] == datatype:
		pass
	elif sourGrapes:
		raise Exception(f"Datatype is '{_registryOfDatatypes[identifier]}' not '{datatype}', so you can take your ball and go home.")
	return _registryOfDatatypes[identifier]

def setDatatypeModule(datatypeModule: str, sourGrapes: bool | None = False) -> str:
	global _datatypeModule
	if not _datatypeModule:
		_datatypeModule = datatypeModule
	elif _datatypeModule == datatypeModule:
		pass
	elif sourGrapes:
		raise Exception(f"Datatype module is '{_datatypeModule}' not '{datatypeModule}', so you can take your ball and go home.")
	return _datatypeModule

def setDatatypeElephino(datatype: str, sourGrapes: bool | None = False) -> str:
	return reportDatatypeLimit('elephino', datatype, sourGrapes)

def setDatatypeFoldsTotal(datatype: str, sourGrapes: bool | None = False) -> str:
	return reportDatatypeLimit('foldsTotal', datatype, sourGrapes)

def setDatatypeLeavesTotal(datatype: str, sourGrapes: bool | None = False) -> str:
	return reportDatatypeLimit('leavesTotal', datatype, sourGrapes)

def _get_datatype(identifier: str) -> str:
	global _registryOfDatatypes
	if not _registryOfDatatypes[identifier]:
		if identifier in indexMy._member_names_:
			_registryOfDatatypes[identifier] = _datatypeDefault.get(identifier) or _get_datatype('elephino')
		elif identifier in indexTrack._member_names_:
			_registryOfDatatypes[identifier] = _datatypeDefault.get(identifier) or _get_datatype('elephino')
		else:
			_registryOfDatatypes[identifier] = _datatypeDefault.get(identifier) or _get_datatype('foldsTotal')
	return _registryOfDatatypes[identifier]

def getDatatypeModule() -> str:
	global _datatypeModule
	if not _datatypeModule:
		_datatypeModule = datatypeModulePACKAGING
	return _datatypeModule

def setInStone(identifier: str) -> type[Any]:
	datatypeModule: str = getDatatypeModule()
	datatypeStr: str = _get_datatype(identifier)
	return cast(type[Any], getattr(eval(datatypeModule), datatypeStr))

def hackSSOTdtype(identifier: str) -> type[Any]:
	_hackSSOTdtype: dict[str, str]={
	'connectionGraph': 	'dtypeLeavesTotal',
	'dtypeElephino': 	'dtypeElephino',
	'dtypeFoldsTotal': 	'dtypeFoldsTotal',
	'dtypeLeavesTotal': 'dtypeLeavesTotal',
	'foldGroups': 		'dtypeFoldsTotal',
	'gapsWhere': 		'dtypeLeavesTotal',
	'mapShape': 		'dtypeLeavesTotal',
	'my': 				'dtypeElephino',
	'track': 			'dtypeElephino',
	}
	RubeGoldBerg: str = _hackSSOTdtype[identifier]
	if RubeGoldBerg == 'dtypeElephino':
		return setInStone('elephino')
	elif RubeGoldBerg == 'dtypeFoldsTotal':
		return setInStone('foldsTotal')
	elif RubeGoldBerg == 'dtypeLeavesTotal':
		return setInStone('leavesTotal')
	raise Exception("Dude, you forgot to set a value in `hackSSOTdtype`.")

def hackSSOTdatatype(identifier: str) -> str:
	_hackSSOTdatatype: dict[str, str]={
	'connectionGraph':	 		'datatypeLeavesTotal',
	'countDimensionsGapped': 	'datatypeLeavesTotal',
	'datatypeElephino': 		'datatypeElephino',
	'datatypeFoldsTotal': 		'datatypeFoldsTotal',
	'datatypeLeavesTotal': 		'datatypeLeavesTotal',
	'dimensionsTotal': 			'datatypeLeavesTotal',
	'dimensionsUnconstrained':	'datatypeLeavesTotal',
	'foldGroups': 				'datatypeFoldsTotal',
	'gap1ndex': 				'datatypeLeavesTotal',
	'gap1ndexCeiling': 			'datatypeElephino',
	'gapRangeStart': 			'datatypeElephino',
	'gapsWhere': 				'datatypeLeavesTotal',
	'groupsOfFolds': 			'datatypeFoldsTotal',
	'indexDimension': 			'datatypeLeavesTotal',
	'indexLeaf': 				'datatypeLeavesTotal',
	'indexMiniGap': 			'datatypeElephino',
	'leaf1ndex': 				'datatypeLeavesTotal',
	'leafAbove': 				'datatypeLeavesTotal',
	'leafBelow': 				'datatypeLeavesTotal',
	'leafConnectee': 			'datatypeLeavesTotal',
	'mapShape': 				'datatypeLeavesTotal',
	'my':	 					'datatypeElephino',
	'taskDivisions': 			'datatypeLeavesTotal',
	'taskIndex': 				'datatypeLeavesTotal',
	'track':	 				'datatypeElephino',
	}
	RubeGoldBerg: str = _hackSSOTdatatype[identifier]
	if RubeGoldBerg == 'datatypeElephino':
		return _get_datatype('elephino')
	elif RubeGoldBerg == 'datatypeFoldsTotal':
		return _get_datatype('foldsTotal')
	elif RubeGoldBerg == 'datatypeLeavesTotal':
		return _get_datatype('leavesTotal')
	raise Exception("Dude, you forgot to set a value in `hackSSOTdatatype`.")
