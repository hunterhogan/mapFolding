from mapFolding.beDRY import getConnectionGraph, getLeavesTotal, makeDataContainer
from mapFolding.datatypes import Array3D, Array1DElephino, Array1DLeavesTotal, DatatypeElephino, DatatypeFoldsTotal, DatatypeLeavesTotal
import dataclasses

@dataclasses.dataclass
class MapFoldingState:
	mapShape: tuple[DatatypeLeavesTotal, ...] = dataclasses.field(init=True, metadata={'elementConstructor': 'DatatypeLeavesTotal'})

	groupsOfFolds: DatatypeFoldsTotal = dataclasses.field(default=DatatypeFoldsTotal(0), metadata={'theCountingIdentifier': True})

	gap1ndex: DatatypeElephino = DatatypeElephino(0)
	gap1ndexCeiling: DatatypeElephino = DatatypeElephino(0)
	indexDimension: DatatypeLeavesTotal = DatatypeLeavesTotal(0)
	indexLeaf: DatatypeLeavesTotal = DatatypeLeavesTotal(0)
	indexMiniGap: DatatypeElephino = DatatypeElephino(0)
	leaf1ndex: DatatypeLeavesTotal = DatatypeLeavesTotal(1)
	leafConnectee: DatatypeLeavesTotal = DatatypeLeavesTotal(0)

	dimensionsUnconstrained: DatatypeLeavesTotal = dataclasses.field(default=None, init=True) # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue, reportUnknownMemberType]

	countDimensionsGapped: Array1DLeavesTotal = dataclasses.field(default=None, init=True, metadata={'dtype': Array1DLeavesTotal.__args__[1].__args__[0]}) # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue, reportUnknownMemberType]
	gapRangeStart: Array1DElephino = dataclasses.field(default=None, init=True, metadata={'dtype': Array1DElephino.__args__[1].__args__[0]}) # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue, reportUnknownMemberType]
	gapsWhere: Array1DLeavesTotal = dataclasses.field(default=None, init=True, metadata={'dtype': Array1DLeavesTotal.__args__[1].__args__[0]}) # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue, reportUnknownMemberType]
	leafAbove: Array1DLeavesTotal = dataclasses.field(default=None, init=True, metadata={'dtype': Array1DLeavesTotal.__args__[1].__args__[0]}) # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue, reportUnknownMemberType]
	leafBelow: Array1DLeavesTotal = dataclasses.field(default=None, init=True, metadata={'dtype': Array1DLeavesTotal.__args__[1].__args__[0]}) # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue, reportUnknownMemberType]

	connectionGraph: Array3D = dataclasses.field(init=False, metadata={'dtype': Array3D.__args__[1].__args__[0]}) # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
	dimensionsTotal: DatatypeLeavesTotal = dataclasses.field(init=False)
	leavesTotal: DatatypeLeavesTotal = dataclasses.field(init=False)

	@property
	def foldsTotal(self) -> DatatypeFoldsTotal:
		_foldsTotal = DatatypeFoldsTotal(self.leavesTotal) * self.groupsOfFolds
		return _foldsTotal

	def __post_init__(self) -> None:
		self.dimensionsTotal = DatatypeLeavesTotal(len(self.mapShape))
		self.leavesTotal = DatatypeLeavesTotal(getLeavesTotal(self.mapShape))

		leavesTotalAsInt = int(self.leavesTotal)

		self.connectionGraph = getConnectionGraph(self.mapShape, leavesTotalAsInt, self.__dataclass_fields__['connectionGraph'].metadata['dtype'])

		if self.dimensionsUnconstrained is None: self.dimensionsUnconstrained = DatatypeLeavesTotal(int(self.dimensionsTotal)) # pyright: ignore[reportUnnecessaryComparison]
		if self.gapsWhere is None: self.gapsWhere = makeDataContainer(leavesTotalAsInt * leavesTotalAsInt + 1, self.__dataclass_fields__['gapsWhere'].metadata['dtype']) # pyright: ignore[reportUnnecessaryComparison]
		if self.countDimensionsGapped is None: self.countDimensionsGapped = makeDataContainer(leavesTotalAsInt + 1, self.__dataclass_fields__['countDimensionsGapped'].metadata['dtype']) # pyright: ignore[reportUnnecessaryComparison]
		if self.gapRangeStart is None: self.gapRangeStart = makeDataContainer(leavesTotalAsInt + 1, self.__dataclass_fields__['gapRangeStart'].metadata['dtype']) # pyright: ignore[reportUnnecessaryComparison]
		if self.leafAbove is None: self.leafAbove = makeDataContainer(leavesTotalAsInt + 1, self.__dataclass_fields__['leafAbove'].metadata['dtype']) # pyright: ignore[reportUnnecessaryComparison]
		if self.leafBelow is None: self.leafBelow = makeDataContainer(leavesTotalAsInt + 1, self.__dataclass_fields__['leafBelow'].metadata['dtype']) # pyright: ignore[reportUnnecessaryComparison]
