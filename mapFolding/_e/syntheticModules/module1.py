from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict, deque
from concurrent.futures import as_completed, ProcessPoolExecutor
from functools import cache, partial, reduce
from gmpy2 import (
	bit_clear, bit_flip, bit_mask, bit_scan1, bit_set, bit_test as isBit1吗, f_mod_2exp, is_even as isEven吗, is_odd as isOdd吗, mpz, xmpz)
from humpy_cytoolz import (
	assoc as associateKeyValue, compose, concat, curry as syntacticCurry, dissoc as dissociatePile, first, get, groupby as toolz_groupby,
	itemfilter, keyfilter as filterPile, merge, unique, valfilter as filterLeaf, valfilter as filterLeafOptions, valfilter as filterValue)
from hunterMakesPy import decreasing, errorL33T, inclusive, raiseIfNone, zeroIndexed
from hunterMakesPy.parseParameters import defineConcurrencyLimit, intInnit
from itertools import accumulate, chain, combinations, filterfalse, product as CartesianProduct
from math import factorial, log, prod
from more_itertools import all_unique as allUnique吗, iter_index, last, loops, one, pairwise, partition, triplewise
from operator import add, attrgetter, getitem, itemgetter, methodcaller, mul, neg, sub
from sys import maxsize as sysMaxsize
from tqdm import tqdm
from typing import cast, overload, TYPE_CHECKING, TypeAlias
from Z0Z_tools import between吗, consecutive吗, DOTitems, DOTkeys, DOTvalues, exclude, reverseLookup, thisHasThat吗, thisNotHaveThat吗
import dataclasses

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence
    from concurrent.futures import Future
    from hunterMakesPy import CallableFunction
    from hunterMakesPy.theTypes import Limitation
    from typing import Self, TypeIs

@cache
def getLeavesTotal(mapShape: tuple[int, ...]) -> int:
    productDimensions = 1
    for dimension in mapShape:
        if dimension > sysMaxsize // productDimensions:
            message: str = f'I received `dimension = {dimension!r}` in `mapShape = {mapShape!r}`, but the product of the dimensions exceeds the maximum size of an integer on this system.'
            raise OverflowError(message)
        productDimensions *= dimension
    return productDimensions

def defineProcessorLimit(CPUlimit: Limitation, concurrencyPackage: str | None = None) -> int:
    if concurrencyPackage == 'numba':
        from numba import get_num_threads, set_num_threads
        concurrencyLimit: int = defineConcurrencyLimit(limit=CPUlimit, cpuTotal=get_num_threads())
        set_num_threads(concurrencyLimit)
        concurrencyLimit = get_num_threads()
    elif concurrencyPackage in {'multiprocessing', None}:
        concurrencyLimit = defineConcurrencyLimit(limit=CPUlimit)
    else:
        concurrencyLimit = defineConcurrencyLimit(limit=CPUlimit)
    return concurrencyLimit
type DimensionIndex = int
Leaf: TypeAlias = int
LeafOptions: TypeAlias = mpz
type LeafSpace = Leaf | LeafOptions
type Pile = int
type Folding = tuple[Leaf, ...]
type PinnedLeaves = dict[Pile, Leaf]
type UndeterminedPiles = dict[Pile, LeafOptions]
leafOrigin: Leaf = 0
pileOrigin: Pile = 0

def getLeafDomain(state: EliminationState, leaf: Leaf) -> range:
    return _getLeafDomain(leaf, state.dimensionsTotal, state.mapShape, state.leavesTotal)

def getLeafOptions(state: EliminationState, pile: Pile) -> LeafOptions:
    return _getLeafOptions(pile, state.dimensionsTotal, state.mapShape, state.leavesTotal)

def getDictionaryLeafOptions(state: EliminationState) -> UndeterminedPiles:
    return {pile: getLeafOptions(state, pile) for pile in range(state.leavesTotal)}

def getIteratorOfLeaves(leafOptions: LeafOptions) -> Iterator[Leaf]:
    iteratorOfLeaves: xmpz = xmpz(leafOptions)
    iteratorOfLeaves[-1] = 0
    return iteratorOfLeaves.iter_set()

def makeLeafAntiOptions(leavesTotal: int, leaves: Iterable[Leaf]) -> LeafOptions:
    return reduce(bit_clear, leaves, bit_mask(leavesTotal + inclusive))

def makeLeafOptions(leavesTotal: int, leaves: Iterable[Leaf]) -> LeafOptions:
    return reduce(bit_set, leaves, bit_set(0, leavesTotal))

def howManyLeavesInLeafOptions(leafOptions: LeafOptions) -> int:
    return leafOptions.bit_count() - 1

def leafOptionsLeafNone(leafOptions: LeafOptions, /) -> LeafOptions | Leaf | None:
    whoAmI: LeafOptions | Leaf | None = leafOptions
    if isLeafOptions吗(leafOptions):
        if leafOptions.bit_count() == 2:
            whoAmI = raiseIfNone(leafOptions.bit_scan1())
        elif leafOptions.bit_count() == 1:
            whoAmI = None
    return whoAmI

@syntacticCurry
def leafOptionsAND(leafOptionsDISPOSABLE: LeafOptions, leafOptions: LeafOptions) -> LeafOptions:
    return leafOptions & leafOptionsDISPOSABLE

def getProductsOfDimensions(mapShape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(accumulate(mapShape, mul, initial=1))

def getSumsOfProductsOfDimensions(mapShape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(accumulate(getProductsOfDimensions(mapShape), add, initial=0))

def getSumsOfProductsOfDimensionsNearest首(productsOfDimensions: tuple[int, ...], dimensionsTotal: int | None = None, dimensionFrom首: int | None = None) -> tuple[int, ...]:
    dimensionsTotal = dimensionsTotal or len(productsOfDimensions) - 1
    if dimensionFrom首 is None:
        dimensionFrom首 = dimensionsTotal
    productsOfDimensionsTruncator: int = dimensionFrom首 - (dimensionsTotal + zeroIndexed)
    productsOfDimensionsFrom首: tuple[int, ...] = productsOfDimensions[0:productsOfDimensionsTruncator][::-1]
    sumsOfProductsOfDimensionsNearest首: tuple[int, ...] = tuple(accumulate(productsOfDimensionsFrom首, add, initial=0))
    return sumsOfProductsOfDimensionsNearest首

def indicesMapShapeDimensionLengthsAreEqual(mapShape: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    return filter(lambda indices: 1 < len(indices), map(tuple, map(partial(iter_index, mapShape), unique(filter(1 .__lt__, mapShape)))))

class PermutationSpace(dict[Pile, LeafSpace]):

    def addMissingPileLeafSpace(self, missing: PermutationSpace | UndeterminedPiles | PinnedLeaves) -> PermutationSpace:
        self = PermutationSpace(sorted(DOTitems(merge(missing, self, factory=PermutationSpace))))
        return self.copy()

    def atPilePinLeaf(self, pile: Pile, leaf: Leaf) -> PermutationSpace:
        return PermutationSpace(associateKeyValue(self, pile, leaf, PermutationSpace))

    def atPilePinLeafSafetyFilter(self, pile: Pile, leaf: Leaf) -> bool:
        return self.leafPinnedAtPile吗(leaf, pile) or (self.pileUndetermined吗(pile) and self.leafNotPinned吗(leaf))

    def bifurcate(self) -> tuple[PinnedLeaves, UndeterminedPiles]:
        leavesPinned: PinnedLeaves = self.extractPinnedLeaves()
        return (leavesPinned, cast('UndeterminedPiles', dissociatePile(self, *DOTkeys(leavesPinned))))

    def copy(self) -> PermutationSpace:
        return PermutationSpace(self)

    def deconstructAtPile(self, pile: Pile | None = None, leavesToPin: Iterable[Leaf] = ()) -> Iterable[PermutationSpace]:
        if pile is None:
            pile = first(filterLeaf(isLeafOptions吗, self))
        if (leafOptions := self.getLeafOptions(pile)) is None:
            deconstructed: Iterable[PermutationSpace] = deque([self])
        else:
            leavesToPin = leavesToPin or getIteratorOfLeaves(leafOptions)
            deconstructed = map(partial(self.atPilePinLeaf, pile), filter(self.leafNotPinned吗, leavesToPin))
        return deconstructed

    def deconstructByDomainOfLeaf(self, leaf: Leaf, leafDomain: Iterable[Pile]) -> deque[PermutationSpace]:
        deconstructedPermutationSpace: deque[PermutationSpace] = deque()
        if self.leafNotPinned吗(leaf):
            leafInPileRange: Callable[[int], bool] = compose(leafInLeafOptions吗(leaf), partial(self.getLeafOptions, default=bit_mask(len(self))))
            pinLeafAt: Callable[[int], PermutationSpace] = partial(self.atPilePinLeaf, leaf=leaf)
            deconstructedPermutationSpace.extend(map(pinLeafAt, filter(leafInPileRange, filter(self.pileUndetermined吗, leafDomain))))
        else:
            deconstructedPermutationSpace.append(self)
        return deconstructedPermutationSpace

    def deconstructByDomainsCombined(self, leaves: Sequence[Leaf], leavesDomain: Iterable[Sequence[Pile]]) -> deque[PermutationSpace]:
        deconstructedPermutationSpace: deque[PermutationSpace] = deque()

        def pileOpenByIndex(index: int) -> CallableFunction[[Sequence[Pile]], bool]:

            def workhorse(domain: Sequence[Pile]) -> bool:
                return self.pileUndetermined吗(domain[index])
            return workhorse

        def leafInPileRangeByIndex(index: int) -> CallableFunction[[Sequence[Pile]], bool]:

            def workhorse(domain: Sequence[Pile]) -> bool:
                leafOptions: LeafOptions = raiseIfNone(self.getLeafOptions(domain[index], default=bit_mask(len(self))))
                return leafInLeafOptions吗(leaves[index], leafOptions)
            return workhorse

        def isPinnedAtPileByIndex(leaf: Leaf, index: int) -> CallableFunction[[Sequence[Pile]], bool]:

            def workhorse(domain: Sequence[Pile]) -> bool:
                return self.leafPinnedAtPile吗(leaf, domain[index])
            return workhorse
        if any(map(self.leafNotPinned吗, leaves)):
            for index in range(len(leaves)):
                if self.leafNotPinned吗(leaves[index]):
                    leavesDomain = filter(pileOpenByIndex(index), leavesDomain)
                    leavesDomain = filter(leafInPileRangeByIndex(index), leavesDomain)
                else:
                    leavesDomain = filter(isPinnedAtPileByIndex(leaves[index], index), leavesDomain)
            for listOfPiles in leavesDomain:
                permutationSpaceForListOfPiles: PermutationSpace = self.copy()
                for index in range(len(leaves)):
                    permutationSpaceForListOfPiles = permutationSpaceForListOfPiles.atPilePinLeaf(listOfPiles[index], leaves[index])
                deconstructedPermutationSpace.append(permutationSpaceForListOfPiles)
        else:
            deconstructedPermutationSpace.append(self)
        return deconstructedPermutationSpace

    def extractPinnedLeaves(self) -> PinnedLeaves:
        return dict(sorted(DOTitems(filterLeaf(isLeaf吗, self))))

    def extractUndeterminedPiles(self) -> UndeterminedPiles:
        return dict(sorted(DOTitems(filterLeaf(isLeafOptions吗, self))))

    @overload
    def getLeaf(self, pile: Pile, default: None = None) -> Leaf | None:
        ...

    @overload
    def getLeaf(self, pile: Pile, default: Leaf) -> Leaf:
        ...

    @overload
    def getLeaf[个](self, pile: Pile, default: 个) -> Leaf | 个:
        ...

    def getLeaf[个](self, pile: Pile, default: Leaf | 个 | None = None) -> Leaf | 个 | None:
        ImaLeaf: LeafSpace | None = self.get(pile)
        if isLeaf吗(ImaLeaf):
            return ImaLeaf
        return default

    @overload
    def getLeafOptions(self, pile: Pile, default: None = None) -> LeafOptions | None:
        ...

    @overload
    def getLeafOptions(self, pile: Pile, default: LeafOptions) -> LeafOptions:
        ...

    @overload
    def getLeafOptions[个](self, pile: Pile, default: 个) -> LeafOptions | 个:
        ...

    def getLeafOptions[个](self, pile: Pile, default: LeafOptions | 个 | None = None) -> LeafOptions | 个 | None:
        ImaLeafOptions: LeafSpace | None = self.get(pile)
        if isLeafOptions吗(ImaLeafOptions):
            return ImaLeafOptions
        return default

    def leafNotPinned吗(self, leaf: Leaf) -> bool:
        return leaf not in self.values()

    @property
    def leafCount(self) -> int:
        return sum(map(isLeaf吗, self.values()))

    def leafPinned吗(self, leaf: Leaf) -> bool:
        return leaf in self.values()

    def leafPinnedAtPile吗(self, leaf: Leaf, pile: Pile) -> bool:
        return leaf == self.get(pile)

    def makeFolding(self, leavesToInsert: Sequence[Leaf] = ()) -> Folding:
        pilesToInsert: Iterator[Pile] = DOTkeys(self.extractUndeterminedPiles())
        return tuple(DOTvalues(dict(sorted(DOTitems(cast('PinnedLeaves', merge(self, dict(zip(pilesToInsert, leavesToInsert, strict=True)), factory=PermutationSpace)))))))

    def pilePinned吗(self, pile: Pile) -> bool:
        return isLeaf吗(self[pile])

    def pileUndetermined吗(self, pile: Pile) -> bool:
        return not isLeaf吗(self[pile])

@dataclasses.dataclass(slots=True)
class EliminationState:
    mapShape: tuple[int, ...] = dataclasses.field(init=True)
    groupsOfFolds: int = 0
    listFolding: deque[Folding] = dataclasses.field(default_factory=deque[Folding], init=True)
    listPermutationSpace: deque[PermutationSpace] = dataclasses.field(default_factory=deque[PermutationSpace], init=True)
    pile: Pile = -1
    permutationSpace: PermutationSpace = dataclasses.field(default_factory=PermutationSpace, init=True)
    Theorem2aMultiplier: int = 1
    Theorem2Multiplier: int = 1
    Theorem3Multiplier: int = 1
    Theorem4Multiplier: int = 1
    dimensionsTotal: int = dataclasses.field(init=False)
    foldingCheckSum: int = dataclasses.field(init=False)
    leafLast: Leaf = dataclasses.field(init=False)
    leavesTotal: int = dataclasses.field(init=False)
    pileLast: Pile = dataclasses.field(init=False)
    pilesTotal: int = dataclasses.field(init=False)
    productsOfDimensions: tuple[int, ...] = dataclasses.field(init=False)
    sumsOfProductsOfDimensions: tuple[int, ...] = dataclasses.field(init=False)
    sumsOfProductsOfDimensionsNearest首: tuple[int, ...] = dataclasses.field(init=False)
    首: int = dataclasses.field(init=False)

    @property
    def foldsTotal(self) -> int:
        return prod((self.groupsOfFolds, self.Theorem2aMultiplier, self.Theorem2Multiplier, self.Theorem3Multiplier, self.Theorem4Multiplier))

    def __post_init__(self) -> None:
        self.dimensionsTotal = len(self.mapShape)
        self.leavesTotal = getLeavesTotal(self.mapShape)
        if 0 < self.leavesTotal:
            self.Theorem2aMultiplier = self.leavesTotal
        self.leafLast = self.leavesTotal - 1
        self.foldingCheckSum = self.leafLast * self.leavesTotal // 2
        self.pilesTotal = self.leavesTotal
        self.pileLast = self.pilesTotal - 1
        self.首 = self.leavesTotal
        self.productsOfDimensions = getProductsOfDimensions(self.mapShape)
        self.sumsOfProductsOfDimensions = getSumsOfProductsOfDimensions(self.mapShape)
        self.sumsOfProductsOfDimensionsNearest首 = getSumsOfProductsOfDimensionsNearest首(self.productsOfDimensions, self.dimensionsTotal, self.dimensionsTotal)

    def moveToListFolding(self) -> Self:
        foldingGroup吗: dict[bool, list[PermutationSpace]] = toolz_groupby(compose(self.leavesTotal.__eq__, attrgetter('leafCount')), self.listPermutationSpace)
        self.listPermutationSpace = deque(foldingGroup吗.get(False, ()))
        self.listFolding.extend(map(methodcaller('makeFolding'), foldingGroup吗.get(True, ())))
        return self

    def permutationSpaceCreaseViolation吗(self, permutationSpace: PermutationSpace) -> bool:
        leafToPile: dict[Leaf, Pile] = {leafValue: pileKey for pileKey, leafValue in DOTitems(permutationSpace.extractPinnedLeaves())}
        for dimension in range(self.dimensionsTotal):
            listPileCreaseByParity: list[list[tuple[Pile, Pile]]] = [[], []]
            for pile, leaf in permutationSpace.extractPinnedLeaves().items():
                crease: int | None = getCreasePost(self.mapShape, leaf, dimension)
                if crease:
                    pileCrease: int | None = leafToPile.get(crease)
                    if pileCrease:
                        listPileCreaseByParity[oddLeaf吗(self.mapShape, leaf, dimension)].append((pile, pileCrease))
            for groupedParity in listPileCreaseByParity:
                if any((creaseViolation吗(pile, pileComparand, pileCrease, pileComparandCrease) for (pile, pileCrease), (pileComparand, pileComparandCrease) in combinations(sorted(groupedParity), 2))):
                    return True
        return False

    def pinAt_pile吗(self, leaf: Leaf) -> bool:
        return all((self.permutationSpace.leafNotPinned吗(leaf), self.permutationSpace.pileUndetermined吗(self.pile), self.pile in getLeafDomain(self, leaf)))

    def reduceAllPermutationSpace(self, listFunctionsReduction: Sequence[Callable[[EliminationState, PermutationSpace], PermutationSpace | None]]) -> Self:
        listPermutationSpace: deque[PermutationSpace] = self.listPermutationSpace
        self.listPermutationSpace = deque()
        listPermutationSpaceIrreducible: deque[PermutationSpace] = deque()
        while listPermutationSpace:
            permutationSpace: PermutationSpace | None = listPermutationSpace.pop()
            sumPermutationSpace: Leaf | LeafOptions = sum(permutationSpace.values())
            functionsReduction: deque[Callable[[EliminationState, PermutationSpace], PermutationSpace | None]] = deque(listFunctionsReduction)
            keepGoing: bool = True
            while keepGoing:
                reducePermutationSpace: Callable[[EliminationState, PermutationSpace], PermutationSpace | None] = functionsReduction.popleft()
                permutationSpace = reducePermutationSpace(self, raiseIfNone(permutationSpace))
                if not permutationSpace:
                    keepGoing = False
                elif sumPermutationSpace != sum(permutationSpace.values()):
                    functionsReduction = deque(listFunctionsReduction)
                    sumPermutationSpace = sum(permutationSpace.values())
                elif not functionsReduction:
                    listPermutationSpaceIrreducible.append(permutationSpace)
                    keepGoing = False
        else:
            self.listPermutationSpace.extend(listPermutationSpaceIrreducible)
        return self

    def removeCreaseViolations(self) -> Self:
        listPermutationSpace: deque[PermutationSpace] = self.listPermutationSpace.copy()
        self.listPermutationSpace = deque()
        self.listPermutationSpace.extend(filterfalse(self.permutationSpaceCreaseViolation吗, listPermutationSpace))
        return self

@syntacticCurry
def leafInLeafOptions吗(leaf: Leaf, leafOptions: LeafOptions) -> bool:
    return leafOptions.bit_test(leaf)

@syntacticCurry
def leafPinned吗(leavesPinned: PinnedLeaves, leaf: Leaf) -> bool:
    return leaf in leavesPinned.values()

@syntacticCurry
def notPileLast(pileLast: Pile, pile: Pile) -> bool:
    return pileLast != pile

def isLeaf吗(leafSpace: LeafSpace | None) -> TypeIs[Leaf]:
    return isinstance(leafSpace, Leaf)

def isLeafOptions吗(leafSpace: LeafSpace | None) -> TypeIs[LeafOptions]:
    return isinstance(leafSpace, LeafOptions)

def segregateLeafPinnedAtPile(listPermutationSpace: Iterable[PermutationSpace], leaf: Leaf, pile: Pile) -> tuple[list[PermutationSpace], list[PermutationSpace]]:
    isPinned: Callable[[PermutationSpace], bool] = partial(PermutationSpace.leafPinnedAtPile吗, leaf=leaf, pile=pile)
    grouped: dict[bool, list[PermutationSpace]] = toolz_groupby(isPinned, listPermutationSpace)
    return (grouped.get(False, []), grouped.get(True, []))

def excludeLeaf_rBeforeLeaf_k(state: EliminationState, leaf_k: Leaf, leaf_r: Leaf, domain_k: Iterable[Pile] | None = None, domain_r: Iterable[Pile] | None = None) -> EliminationState:
    if domain_k is None:
        domain_k = getLeafDomain(state, leaf_k)
    for pile_k in reversed(tuple(domain_k)):
        state = excludeLeaf_rBeforeLeaf_kAtPile_k(state, leaf_k, leaf_r, pile_k, domainOf_leaf_r=domain_r)
    return state

def excludeLeaf_rBeforeLeaf_kAtPile_k(state: EliminationState, leaf_k: Leaf, leaf_r: Leaf, pile_k: Pile, domainOf_leaf_r: Iterable[Pile] | None = None) -> EliminationState:
    listPermutationSpace: deque[PermutationSpace] = state.listPermutationSpace
    state.listPermutationSpace = deque()
    listPermutationSpaceUnchanged: deque[PermutationSpace] = deque()
    listExcludeLeaf_r: Iterable[PermutationSpace] = []
    for permutationSpace in listPermutationSpace:
        if permutationSpace.leafPinnedAtPile吗(leaf_k, pile_k):
            listExcludeLeaf_r.append(permutationSpace)
        elif leafInLeafOptions吗(leaf_k, permutationSpace.getLeafOptions(pile_k, LeafOptions(0))):
            permutationSpaceCopy = permutationSpace.copy()
            permutationSpaceCopy[pile_k] = bit_clear(permutationSpaceCopy[pile_k], leaf_k)
            state.listPermutationSpace.append(permutationSpaceCopy)
            listExcludeLeaf_r.append(permutationSpace.atPilePinLeaf(pile_k, leaf_k))
        else:
            listPermutationSpaceUnchanged.append(permutationSpace)
    if domainOf_leaf_r is None:
        domainOf_leaf_r = getLeafDomain(state, leaf_r)
    for pile_r in filter(between吗(0, pile_k - inclusive), domainOf_leaf_r):
        listExcludeLeaf_r = excludeLeafAtPile(listExcludeLeaf_r, leaf_r, pile_r)
    state.listPermutationSpace.extend(listExcludeLeaf_r)
    state.reduceAllPermutationSpace(listFunctionsReduction).removeCreaseViolations()
    state.listPermutationSpace.extend(listPermutationSpaceUnchanged)
    return state

def excludeLeafAtPile(listPermutationSpace: Iterable[PermutationSpace], leaf: Leaf, pile: Pile) -> Iterator[PermutationSpace]:
    listPermutationSpace, _pinnedAtPile = segregateLeafPinnedAtPile(listPermutationSpace, leaf, pile)
    pilePinned: dict[bool, list[PermutationSpace]] = toolz_groupby(methodcaller('pilePinned吗', pile), listPermutationSpace)
    yield from pilePinned.get(True, [])
    for permutationSpace in pilePinned.get(False, []):
        permutationSpace[pile] = bit_clear(permutationSpace[pile], leaf)
        yield permutationSpace

def reduceLeafSpace(permutationSpace: PermutationSpace, pilesToUpdate: Iterable[tuple[Pile, LeafOptions]], leafAntiOptions: LeafOptions) -> PermutationSpace:
    for pile, leafOptions in pilesToUpdate:
        leafSpace: LeafSpace | None = leafOptionsLeafNone(leafOptionsAND(leafAntiOptions, leafOptions))
        if leafSpace is None:
            permutationSpace.clear()
        else:
            permutationSpace[pile] = leafSpace
    return permutationSpace

def reducePermutationSpace_LeafIsPinned(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
    permutationSpaceHasNewLeaf: bool = True
    while permutationSpaceHasNewLeaf:
        permutationSpaceHasNewLeaf = False
        leavesPinned, pilesUndetermined = permutationSpace.bifurcate()
        if not (permutationSpace := reduceLeafSpace(permutationSpace, DOTitems(pilesUndetermined), makeLeafAntiOptions(state.leavesTotal, DOTvalues(leavesPinned)))):
            return None
        if len(leavesPinned) < permutationSpace.leafCount:
            permutationSpaceHasNewLeaf = True
    return permutationSpace

def reducePermutationSpace_nakedSubset(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
    permutationSpaceHasNewLeaf: bool = True
    leafOptionsKey: int = 0
    piles: int = 1
    while permutationSpaceHasNewLeaf:
        permutationSpaceHasNewLeaf = False
        leafCount: int = permutationSpace.leafCount
        pilesUndetermined: UndeterminedPiles = permutationSpace.extractUndeterminedPiles()
        groupByLeafOptions: dict[LeafOptions, set[Pile]] = {}
        for pile, leafOptions in DOTitems(filterLeafOptions(thisNotHaveThat吗(unique(pilesUndetermined.values())), pilesUndetermined)):
            groupByLeafOptions.setdefault(leafOptions, set()).add(pile)
        for leafOptions, setPiles in DOTitems(itemfilter(lambda groupBy: howManyLeavesInLeafOptions(groupBy[leafOptionsKey]) == len(groupBy[piles]), groupByLeafOptions)):
            if not (permutationSpace := reduceLeafSpace(permutationSpace, DOTitems(filterPile(thisNotHaveThat吗(setPiles), pilesUndetermined)), makeLeafAntiOptions(state.leavesTotal, getIteratorOfLeaves(leafOptions)))):
                return None
        if permutationSpace.leafCount < leafCount:
            permutationSpaceHasNewLeaf = True
    return permutationSpace

def reducePermutationSpace_leafDomainOf1(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
    permutationSpaceHasNewLeaf: bool = True
    while permutationSpaceHasNewLeaf:
        permutationSpaceHasNewLeaf = False
        leavesPinned, pilesUndetermined = permutationSpace.bifurcate()
        counterLeafDomainSize: Counter[Leaf] = Counter(chain(chain.from_iterable(map(getIteratorOfLeaves, DOTvalues(pilesUndetermined))), DOTvalues(leavesPinned)))
        if set(range(state.leavesTotal)).difference(counterLeafDomainSize.keys()):
            return None
        leavesWithDomainOf1: set[Leaf] = set(DOTkeys(filterValue(1 .__eq__, counterLeafDomainSize))).difference(leavesPinned.values()).difference([state.leavesTotal])
        if leavesWithDomainOf1:
            leaf: Leaf = leavesWithDomainOf1.pop()
            sherpa: PermutationSpace | None = reducePermutationSpace_LeafIsPinned(state, permutationSpace.atPilePinLeaf(one(DOTkeys(filterLeaf(leafInLeafOptions吗(leaf), pilesUndetermined))), leaf))
            if sherpa is None or not sherpa:
                return None
            else:
                permutationSpace = sherpa
            permutationSpaceHasNewLeaf = True
    return permutationSpace
listFunctionsReduction: Sequence[Callable[[EliminationState, PermutationSpace], PermutationSpace | None]] = (reducePermutationSpace_LeafIsPinned, reducePermutationSpace_leafDomainOf1, reducePermutationSpace_nakedSubset)

def _pinPiles(state: EliminationState, maximumSizeListPermutationSpace: int, pileProcessingOrder: deque[Pile], *, CPUlimit: Limitation = None) -> EliminationState:
    workersMaximum: int = defineProcessorLimit(CPUlimit)
    while pileProcessingOrder and len(state.listPermutationSpace) < maximumSizeListPermutationSpace:
        pile: Pile = pileProcessingOrder.popleft()
        thesePilesAreOpen: tuple[Iterator[PermutationSpace], Iterator[PermutationSpace]] = partition(partial(PermutationSpace.pileUndetermined吗, pile=pile), state.listPermutationSpace)
        state.listPermutationSpace = deque(thesePilesAreOpen[False])
        with ProcessPoolExecutor(workersMaximum) as concurrencyManager:
            listClaimTickets: list[Future[EliminationState]] = [concurrencyManager.submit(_pinPilesConcurrentTask, EliminationState(mapShape=state.mapShape, permutationSpace=permutationSpace, pile=pile)) for permutationSpace in thesePilesAreOpen[True]]
            for claimTicket in tqdm(as_completed(listClaimTickets), total=len(listClaimTickets), desc=f'Pinning pile {pile:3d} of {state.pileLast:3d}', disable=False):
                state.listPermutationSpace.extend(claimTicket.result().listPermutationSpace)
                state.listFolding.extend(claimTicket.result().listFolding)
    return state

def _pinPilesConcurrentTask(state: EliminationState) -> EliminationState:
    state.listPermutationSpace.extend(state.permutationSpace.deconstructAtPile(state.pile, filter(state.pinAt_pile吗, _getLeavesAtPile(state))))
    return state.reduceAllPermutationSpace(listFunctionsReduction2上nDimensional).removeCreaseViolations().moveToListFolding()

def _getLeavesAtPile(state: EliminationState) -> Iterable[Leaf]:
    leavesToPin: Iterable[Leaf] = frozenset()
    if state.pile == pileOrigin:
        leavesToPin = frozenset([leafOrigin])
    elif state.pile == 零:
        leavesToPin = frozenset([零])
    elif state.pile == neg(零) + state.首:
        leavesToPin = frozenset([首零(state.dimensionsTotal)])
    elif state.pile == 一:
        leavesToPin = pinPile一ByCrease(state)
    elif state.pile == neg(一) + state.首:
        leavesToPin = pinPile一Ante首ByCrease(state)
    elif state.pile == 一 + 零:
        leavesToPin = pinPile一零ByCrease(state)
    elif state.pile == neg(零 + 一) + state.首:
        leavesToPin = pinPile零一Ante首ByCrease(state)
    elif state.pile == 二:
        leavesToPin = pinPile二ByCrease(state)
    elif state.pile == neg(二) + state.首:
        leavesToPin = pinPile二Ante首ByCrease(state)
    elif state.pile == neg(零) + 首零(state.dimensionsTotal):
        leavesToPin = pinPile零Ante首零AfterDepth4(state)
    return leavesToPin

def pinPilesAtEnds(state: EliminationState, pileDepth: int = 4, maximumSizeListPermutationSpace: int = 2 ** 14, *, CPUlimit: Limitation = None) -> EliminationState:
    if not mapShapeIs2上nDimensions(state.mapShape):
        return state
    if not state.listPermutationSpace:
        state.listPermutationSpace.append(PermutationSpace().addMissingPileLeafSpace(getDictionaryLeafOptions(state)))
    depth: int = getitem(intInnit((pileDepth,), 'pileDepth', int), 0)
    if depth < 0:
        message: str = f'I received `pileDepth = {pileDepth!r}`, but I need a value greater than or equal to 0.'
        raise ValueError(message)
    pileProcessingOrder: deque[Pile] = deque()
    if 0 < depth:
        pileProcessingOrder.extend([pileOrigin])
    if 1 <= depth:
        pileProcessingOrder.extend([零, neg(零) + state.首])
    if 2 <= depth:
        pileProcessingOrder.extend([一, neg(一) + state.首])
    if 3 <= depth:
        pileProcessingOrder.extend([一 + 零, neg(零 + 一) + state.首])
    if 4 <= depth:
        youMustBeDimensionsTallToPinThis = 4
        if youMustBeDimensionsTallToPinThis < state.dimensionsTotal:
            pileProcessingOrder.extend([二])
        youMustBeDimensionsTallToPinThis = 5
        if youMustBeDimensionsTallToPinThis < state.dimensionsTotal:
            pileProcessingOrder.extend([neg(二) + state.首])
    return _pinPiles(state, maximumSizeListPermutationSpace, pileProcessingOrder, CPUlimit=CPUlimit)

def pinPile零Ante首零(state: EliminationState, maximumSizeListPermutationSpace: int = 2 ** 14, *, CPUlimit: Limitation = None) -> EliminationState:
    if not mapShapeIs2上nDimensions(state.mapShape):
        return state
    if not state.listPermutationSpace:
        state = pinPilesAtEnds(state, 0)
    state = pinPilesAtEnds(state, 4, maximumSizeListPermutationSpace)
    if not mapShapeIs2上nDimensions(state.mapShape, youMustBeDimensionsTallToPinThis=6):
        return state
    pileProcessingOrder: deque[Pile] = deque([neg(零) + 首零(state.dimensionsTotal)])
    return _pinPiles(state, maximumSizeListPermutationSpace, pileProcessingOrder, CPUlimit=CPUlimit)

def _pinLeavesByDomain(state: EliminationState, leaves: Sequence[Leaf], leavesDomain: Iterable[Sequence[Pile]], *, youMustBeDimensionsTallToPinThis: int = 3, CPUlimit: Limitation = None) -> EliminationState:
    if not mapShapeIs2上nDimensions(state.mapShape, youMustBeDimensionsTallToPinThis=youMustBeDimensionsTallToPinThis):
        return state
    if not state.listPermutationSpace:
        state = pinPilesAtEnds(state, 0)
    listPermutationSpace: deque[PermutationSpace] = state.listPermutationSpace
    state.listPermutationSpace = deque()
    with ProcessPoolExecutor(defineProcessorLimit(CPUlimit)) as concurrencyManager:
        listClaimTickets: list[Future[EliminationState]] = [concurrencyManager.submit(_pinLeavesByDomainConcurrentTask, EliminationState(state.mapShape, permutationSpace=permutationSpace), leaves, leavesDomain) for permutationSpace in listPermutationSpace]
        for claimTicket in tqdm(as_completed(listClaimTickets), total=len(listClaimTickets), desc=f"Pinning leaves {', '.join(map(f'{{:{len(str(state.leafLast))}d}}'.format, leaves))} of {state.leafLast}", disable=False):
            state.listPermutationSpace.extend(claimTicket.result().listPermutationSpace)
            state.listFolding.extend(claimTicket.result().listFolding)
    return state

def _pinLeavesByDomainConcurrentTask(state: EliminationState, leaves: Sequence[Leaf], leavesDomain: Iterable[Sequence[Pile]]) -> EliminationState:
    state.listPermutationSpace = state.permutationSpace.deconstructByDomainsCombined(leaves, leavesDomain)
    return state.reduceAllPermutationSpace(listFunctionsReduction2上nDimensional).removeCreaseViolations().moveToListFolding()

def _pinLeafByDomain(state: EliminationState, leaf: Leaf, getLeafDomain: CallableFunction[[EliminationState, Leaf], tuple[Pile, ...]], *, youMustBeDimensionsTallToPinThis: int = 3, CPUlimit: Limitation = None) -> EliminationState:
    if not mapShapeIs2上nDimensions(state.mapShape, youMustBeDimensionsTallToPinThis=youMustBeDimensionsTallToPinThis):
        return state
    if not state.listPermutationSpace:
        state = pinPilesAtEnds(state, 0)
    workersMaximum: int = defineProcessorLimit(CPUlimit)
    listPermutationSpace: deque[PermutationSpace] = state.listPermutationSpace
    state.listPermutationSpace = deque()
    with ProcessPoolExecutor(workersMaximum) as concurrencyManager:
        listClaimTickets: list[Future[EliminationState]] = [concurrencyManager.submit(_pinLeafByDomainConcurrentTask, state=EliminationState(mapShape=state.mapShape, permutationSpace=permutationSpace), leaves=leaf, leavesDomain=getLeafDomain(EliminationState(mapShape=state.mapShape, permutationSpace=permutationSpace), leaf)) for permutationSpace in listPermutationSpace]
        for claimTicket in tqdm(as_completed(listClaimTickets), total=len(listClaimTickets), desc=f'Pinning leaf {leaf:16d} of {state.leafLast:3d}', disable=False):
            state.listPermutationSpace.extend(claimTicket.result().listPermutationSpace)
            state.listFolding.extend(claimTicket.result().listFolding)
    return state

def _pinLeafByDomainConcurrentTask(state: EliminationState, leaves: Leaf, leavesDomain: tuple[Pile, ...]) -> EliminationState:
    state.listPermutationSpace = state.permutationSpace.deconstructByDomainOfLeaf(leaves, leavesDomain)
    return state.reduceAllPermutationSpace(listFunctionsReduction2上nDimensional).removeCreaseViolations().moveToListFolding()

def pinLeavesDimension0(state: EliminationState, *, CPUlimit: Limitation = None) -> EliminationState:
    leaves: tuple[Leaf, Leaf] = (leafOrigin, 首零(state.dimensionsTotal))
    return _pinLeavesByDomain(state, leaves, leavesDomain=((pileOrigin, state.pileLast),), CPUlimit=CPUlimit)

def pinLeaf首零Plus零(state: EliminationState, *, CPUlimit: Limitation = None) -> EliminationState:
    leaf: Leaf = 零 + 首零(state.dimensionsTotal)
    return _pinLeafByDomain(state, leaf, getLeaf首零Plus零Domain, CPUlimit=CPUlimit)

def pinLeavesDimension零(state: EliminationState, *, CPUlimit: Limitation = None) -> EliminationState:
    state = pinPilesAtEnds(state, 0)
    return pinLeaf首零Plus零(state, CPUlimit=CPUlimit)

def pinLeavesDimension一(state: EliminationState, *, CPUlimit: Limitation = None) -> EliminationState:
    leaves: tuple[Leaf, Leaf, Leaf, Leaf] = (一 + 零, 一, 首一(state.dimensionsTotal), 首零一(state.dimensionsTotal))
    return _pinLeavesByDomain(state, leaves, getDomainDimension一(state), CPUlimit=CPUlimit)

def pinLeavesDimensions0零一(state: EliminationState, *, CPUlimit: Limitation = None) -> EliminationState:
    state = pinLeavesDimension一(state, CPUlimit=CPUlimit)
    return pinLeavesDimension零(state, CPUlimit=CPUlimit)

def pinLeavesDimension二(state: EliminationState, *, CPUlimit: Limitation = None) -> EliminationState:
    leaves: tuple[Leaf, Leaf, Leaf, Leaf] = (二 + 一, 二 + 一 + 零, 二 + 零, 二)
    return _pinLeavesByDomain(state, leaves, getDomainDimension二(state), youMustBeDimensionsTallToPinThis=5, CPUlimit=CPUlimit)

def pinLeavesDimension首二(state: EliminationState, *, CPUlimit: Limitation = None) -> EliminationState:
    leaves: tuple[Leaf, Leaf, Leaf, Leaf] = (首二(state.dimensionsTotal), 首零二(state.dimensionsTotal), 首零一二(state.dimensionsTotal), 首一二(state.dimensionsTotal))
    return _pinLeavesByDomain(state, leaves, getDomainDimension首二(state), youMustBeDimensionsTallToPinThis=5, CPUlimit=CPUlimit)

def pin3beans2(state: EliminationState, *, CPUlimit: Limitation = None) -> EliminationState:
    return _pinLeavesByDomain(state, (一 + 零, 一), tuple((pile, pile + 1) for pile in getLeafDomain(state, 一 + 零)), CPUlimit=CPUlimit)

def pin首beans(state: EliminationState, *, CPUlimit: Limitation = None) -> EliminationState:
    return _pinLeavesByDomain(state, (首一(state.dimensionsTotal), 首零一(state.dimensionsTotal)), tuple((pile, pile + 1) for pile in getLeafDomain(state, 首一(state.dimensionsTotal))), CPUlimit=CPUlimit)

def _getLeavesCrease(state: EliminationState, leaf: Leaf) -> tuple[Leaf, ...]:
    if 0 < leaf:
        return tuple(getLeavesCreaseAnte(state, abs(leaf)))
    return tuple(getLeavesCreasePost(state, abs(leaf)))

def pinPile一ByCrease(state: EliminationState) -> Iterator[Leaf]:
    direction: CallableFunction[[int, int], int] = sub
    listCreaseIndicesExcluded: list[int] = []
    leafRoot: Leaf = raiseIfNone(state.permutationSpace.getLeaf(direction(state.pile, 1)), f'I could not find an `int` type `Leaf` at {direction(state.pile, 1)}.')
    tupleLeavesCrease: tuple[Leaf, ...] = _getLeavesCrease(state, direction(0, leafRoot))
    leafAt一Ante首: Leaf | None = state.permutationSpace.getLeaf(neg(一) + state.首)
    if leafAt一Ante首 and 0 < dimensionNearestTail(leafAt一Ante首):
        listCreaseIndicesExcluded.extend([*range(dimensionNearestTail(leafAt一Ante首) - 零, state.dimensionsTotal - 一)])
    return exclude(tupleLeavesCrease, listCreaseIndicesExcluded)

def pinPile一Ante首ByCrease(state: EliminationState) -> Iterator[Leaf]:
    direction: CallableFunction[[int, int], int] = add
    listCreaseIndicesExcluded: list[int] = []
    leafRoot: Leaf = raiseIfNone(state.permutationSpace.getLeaf(direction(state.pile, 1)), f'I could not find an `int` type `Leaf` at {direction(state.pile, 1)}.')
    tupleLeavesCrease: tuple[Leaf, ...] = _getLeavesCrease(state, direction(0, leafRoot))
    leafAt一: Leaf | None = state.permutationSpace.getLeaf(一)
    if leafAt一 and leafAt一.bit_length() < state.dimensionsTotal:
        listCreaseIndicesExcluded.extend([*range(零, dimensionNearest首(leafAt一) + inclusive)])
    return exclude(tupleLeavesCrease, listCreaseIndicesExcluded)

def pinPile一零ByCrease(state: EliminationState) -> Iterator[Leaf]:
    direction: CallableFunction[[int, int], int] = sub
    listCreaseIndicesExcluded: list[int] = []
    leafRoot: Leaf = raiseIfNone(state.permutationSpace.getLeaf(direction(state.pile, 1)), f'I could not find an `int` type `Leaf` at {direction(state.pile, 1)}.')
    tupleLeavesCrease: tuple[Leaf, ...] = _getLeavesCrease(state, direction(0, leafRoot))
    leafAt一: Leaf = raiseIfNone(state.permutationSpace.getLeaf(一))
    leafAt一Ante首: Leaf = raiseIfNone(state.permutationSpace.getLeaf(neg(一) + state.首))
    if 1 < len(tupleLeavesCrease):
        listCreaseIndicesExcluded.append(0)
    if isEven吗(leafAt一Ante首) and leafAt一 == 零 + 首零(state.dimensionsTotal):
        listCreaseIndicesExcluded.extend([*range(dimensionNearestTail(leafAt一Ante首) + 零, state.dimensionsTotal)])
    return exclude(tupleLeavesCrease, listCreaseIndicesExcluded)

def pinPile零一Ante首ByCrease(state: EliminationState) -> Iterator[Leaf]:
    direction: CallableFunction[[int, int], int] = add
    listCreaseIndicesExcluded: list[int] = []
    leafRoot: Leaf = raiseIfNone(state.permutationSpace.getLeaf(direction(state.pile, 1)), f'I could not find an `int` type `Leaf` at {direction(state.pile, 1)}.')
    tupleLeavesCrease: tuple[Leaf, ...] = _getLeavesCrease(state, direction(0, leafRoot))
    leafAt一: Leaf = raiseIfNone(state.permutationSpace.getLeaf(一))
    leafAt一Ante首: Leaf = raiseIfNone(state.permutationSpace.getLeaf(neg(一) + state.首))
    if leafAt一Ante首 < 首零一(state.dimensionsTotal):
        listCreaseIndicesExcluded.append(-1)
    if leafAt一Ante首 == 零 + 首零(state.dimensionsTotal) and leafAt一 != 一 + 零:
        listCreaseIndicesExcluded.extend([*range(dimensionNearest首(leafAt一) - 零)])
    return exclude(tupleLeavesCrease, listCreaseIndicesExcluded)

def pinPile二ByCrease(state: EliminationState) -> Iterator[Leaf]:
    direction: CallableFunction[[int, int], int] = sub
    listCreaseIndicesExcluded: list[int] = []
    leafRoot: Leaf = raiseIfNone(state.permutationSpace.getLeaf(direction(state.pile, 1)), f'I could not find an `int` type `Leaf` at {direction(state.pile, 1)}.')
    tupleLeavesCrease: tuple[Leaf, ...] = _getLeavesCrease(state, direction(0, leafRoot))
    leafAt一: Leaf = raiseIfNone(state.permutationSpace.getLeaf(一))
    leafAt一Ante首: Leaf = raiseIfNone(state.permutationSpace.getLeaf(neg(一) + state.首))
    leafAt一零: Leaf = raiseIfNone(state.permutationSpace.getLeaf(一 + 零))
    leafAt零一Ante首: Leaf = raiseIfNone(state.permutationSpace.getLeaf(neg(零 + 一) + state.首))
    if isOdd吗(leafAt一零):
        listCreaseIndicesExcluded.extend([*range(dimensionNearest首(leafAt一零), 5), ptount(leafAt一零)])
        listCreaseIndicesExcluded.append((dimensionIndex(leafInSubHyperplane(leafAt一Ante首)) + 4) % 5)
    if isEven吗(leafAt一零):
        listCreaseIndicesExcluded.extend([*range(state.dimensionsTotal - 3)][state.dimensionsTotal - 3 - (state.dimensionsTotal - 2 - leafInSubHyperplane(leafAt零一Ante首 - (leafAt零一Ante首.bit_count() - isEven吗(leafAt零一Ante首))).bit_count()) % (state.dimensionsTotal - 2) - isEven吗(leafAt零一Ante首):None])
        if isEven吗(leafAt一Ante首):
            listCreaseIndicesExcluded.extend([*range(dimensionNearestTail(leafInSubHyperplane(leafAt一Ante首)) - 一, state.dimensionsTotal - 3)])
    if leafAt一 == 零 + 首零(state.dimensionsTotal):
        listCreaseIndicesExcluded.extend([(dimensionIndex(leafInSubHyperplane(leafAt一Ante首)) + 4) % 5, dimensionNearestTail(leafAt零一Ante首) - 1])
        if 零 + 首零(state.dimensionsTotal) < leafAt零一Ante首:
            listCreaseIndicesExcluded.extend([*range(int(leafAt零一Ante首 - int(bit_flip(0, dimensionNearest首(leafAt零一Ante首)))).bit_length() - 1, state.dimensionsTotal - 2)])
        if 0 < leafAt一零 - leafAt一 <= bit_flip(0, state.dimensionsTotal - 4) and 0 < leafAt一Ante首 - leafAt一零 <= bit_flip(0, state.dimensionsTotal - 3):
            listCreaseIndicesExcluded.extend([ptount(leafAt一零), state.dimensionsTotal - 3, state.dimensionsTotal - 4])
    return exclude(tupleLeavesCrease, listCreaseIndicesExcluded)

def pinPile二Ante首ByCrease(state: EliminationState) -> Iterator[Leaf]:
    direction: CallableFunction[[int, int], int] = add
    listCreaseIndicesExcluded: list[int] = []
    leafRoot: Leaf = raiseIfNone(state.permutationSpace.getLeaf(direction(state.pile, 1)), f'I could not find an `int` type `Leaf` at {direction(state.pile, 1)}.')
    tupleLeavesCrease: tuple[Leaf, ...] = _getLeavesCrease(state, direction(0, leafRoot))
    leafAt一: Leaf = raiseIfNone(state.permutationSpace.getLeaf(一))
    leafAt一Ante首: Leaf = raiseIfNone(state.permutationSpace.getLeaf(neg(一) + state.首))
    leafAt一零: Leaf = raiseIfNone(state.permutationSpace.getLeaf(一 + 零))
    leafAt零一Ante首: Leaf = raiseIfNone(state.permutationSpace.getLeaf(neg(零 + 一) + state.首))
    leafAt二: Leaf = raiseIfNone(state.permutationSpace.getLeaf(二))
    addendDimension首零: int = leafAt零一Ante首 - leafAt一Ante首
    addendDimension一零: int = leafAt二 - leafAt一零
    addendDimension一: int = leafAt一零 - leafAt一
    addendDimension零: int = leafAt一 - 零
    if addendDimension一零 in {一, 二, 三, 四} or (addendDimension一零 == 五 and addendDimension首零 != 一) or addendDimension一 in {二, 三} or (addendDimension一 == 一 and (not (addendDimension零 == addendDimension首零 and addendDimension一零 < 0))):
        if leafAt零一Ante首 == 首一(state.dimensionsTotal):
            if addendDimension零 == 三:
                listCreaseIndicesExcluded.append(dimensionIndex(二))
            if addendDimension零 == 五:
                if addendDimension一 == 二:
                    listCreaseIndicesExcluded.append(dimensionIndex(二))
                if addendDimension一 == 三:
                    listCreaseIndicesExcluded.append(dimensionIndex(三))
            if addendDimension一零 == 三:
                listCreaseIndicesExcluded.append(dimensionIndex(二))
        if 0 < (dimensionTail := dimensionNearestTail(leafAt零一Ante首)) < 5:
            listCreaseIndicesExcluded.extend(list(range(dimensionTail % 4)) or [dimensionIndex(一)])
        if addendDimension首零 == neg(五):
            listCreaseIndicesExcluded.append(dimensionIndex(一))
        if addendDimension首零 == 一:
            listCreaseIndicesExcluded.append(dimensionIndex(二))
        if addendDimension首零 == 四:
            if addendDimension零 == 三:
                listCreaseIndicesExcluded.extend([*range(dimensionIndex(一), dimensionIndex(二) + inclusive)])
            if addendDimension一 == 一 and addendDimension一零 == 三:
                listCreaseIndicesExcluded.append(dimensionIndex(二))
        if addendDimension零 == 一:
            listCreaseIndicesExcluded.append(dimensionIndex(一))
            if addendDimension一零 == 三:
                listCreaseIndicesExcluded.extend([*range(dimensionIndex(二), dimensionIndex(三) + inclusive)])
            if addendDimension一零 == 四:
                listCreaseIndicesExcluded.extend([*range(dimensionIndex(三), dimensionIndex(四) + inclusive)])
        if addendDimension零 == 二:
            listCreaseIndicesExcluded.extend([*range(dimensionIndex(一), dimensionIndex(二) + inclusive)])
        if addendDimension零 == 三:
            listCreaseIndicesExcluded.append(dimensionIndex(三))
        if addendDimension一 == 二:
            listCreaseIndicesExcluded.append(dimensionIndex(一))
        if addendDimension一 == 三:
            listCreaseIndicesExcluded.extend([*range(dimensionIndex(一), dimensionIndex(二) + inclusive)])
        if addendDimension一 == 四:
            listCreaseIndicesExcluded.append(dimensionIndex(一))
            if addendDimension一零 == 三:
                listCreaseIndicesExcluded.extend([*range(dimensionIndex(一), dimensionIndex(三) + inclusive)])
        if addendDimension一零 == 一:
            listCreaseIndicesExcluded.append(dimensionIndex(一))
        if addendDimension一零 == 二:
            listCreaseIndicesExcluded.append(dimensionIndex(二))
        if addendDimension一零 == 三:
            listCreaseIndicesExcluded.append(dimensionIndex(三))
        if addendDimension一零 == 五:
            listCreaseIndicesExcluded.append(dimensionIndex(一))
    return exclude(tupleLeavesCrease, listCreaseIndicesExcluded)

def pinPile零Ante首零AfterDepth4(state: EliminationState) -> list[int]:
    leafAt一: Leaf = raiseIfNone(state.permutationSpace.getLeaf(一))
    leafAt一Ante首: Leaf = raiseIfNone(state.permutationSpace.getLeaf(neg(一) + state.首))
    leafAt一零: Leaf = raiseIfNone(state.permutationSpace.getLeaf(一 + 零))
    leafAt零一Ante首: Leaf = raiseIfNone(state.permutationSpace.getLeaf(neg(零 + 一) + state.首))
    leafAt二: Leaf = raiseIfNone(state.permutationSpace.getLeaf(二))
    leafAt二Ante首: Leaf = raiseIfNone(state.permutationSpace.getLeaf(neg(二) + state.首))
    dictionaryLeafOptions: dict[Pile, LeafOptions] = getDictionaryLeafOptions(state)
    listRemoveLeaves: list[int] = []
    pileExcluder: Pile = 一
    for dimension, leaf in enumerate(getIteratorOfLeaves(dictionaryLeafOptions[pileExcluder])):
        if leaf == leafAt一:
            if dimension < state.dimensionsTotal - 2:
                listRemoveLeaves.extend([一, 首零(state.dimensionsTotal) + leafAt一])
            if 0 < dimension < state.dimensionsTotal - 2:
                listRemoveLeaves.extend([一 + leafAt一])
            if dimension == 1:
                listRemoveLeaves.extend([首零(state.dimensionsTotal) + leafAt一 + 零])
            if dimension == state.dimensionsTotal - 2:
                listRemoveLeaves.extend([首一(state.dimensionsTotal), 首一(state.dimensionsTotal) + leafAt一])
    del pileExcluder
    if leafAt一 == 零 + 首零(state.dimensionsTotal):
        listRemoveLeaves.extend([首一(state.dimensionsTotal), leafAt一Ante首 + 零])
    if dimensionNearest首(leafAt一) < state.dimensionsTotal - 3:
        listRemoveLeaves.extend([一, leafAt一Ante首 + 一])
    pileExcluder = neg(一) + state.首
    for dimension, leaf in enumerate(getIteratorOfLeaves(dictionaryLeafOptions[pileExcluder])):
        if leaf == leafAt一Ante首:
            if dimension == 0:
                listRemoveLeaves.extend([一])
            if dimension < state.dimensionsTotal - 2:
                listRemoveLeaves.extend([首一(state.dimensionsTotal) + leafAt一Ante首])
            if 0 < dimension < state.dimensionsTotal - 2:
                listRemoveLeaves.extend([getitem(state.productsOfDimensions, dimension), 首一(state.dimensionsTotal) + leafAt一Ante首 - getitem(state.sumsOfProductsOfDimensions, dimension)])
            if 0 < dimension < state.dimensionsTotal - 3:
                listRemoveLeaves.extend([零 + leafAt一Ante首])
            if 0 < dimension < state.dimensionsTotal - 1:
                listRemoveLeaves.extend([首一(state.dimensionsTotal)])
    del pileExcluder
    if leafAt一 == 零 + 首二(state.dimensionsTotal) and leafAt一Ante首 == 首零一(state.dimensionsTotal):
        listRemoveLeaves.extend([首二(state.dimensionsTotal), 首零一二(state.dimensionsTotal)])
    listRemoveLeaves.extend([leafAt一零])
    if leafAt一零 == 三 + 二 + 零:
        listRemoveLeaves.extend([二 + 一 + 零, 零 + 二 + 首零(state.dimensionsTotal)])
    if leafAt一零 == 零 + 二 + 首一(state.dimensionsTotal):
        listRemoveLeaves.extend([首二(state.dimensionsTotal), leafAt一零 + getitem(state.productsOfDimensions, raiseIfNone(dimensionSecondNearest首(leafAt一零))), leafAt一零 + getitem(state.sumsOfProductsOfDimensions, raiseIfNone(dimensionSecondNearest首(leafAt一零)) + 1), 首零一二(state.dimensionsTotal)])
    if leafAt一零 == 零 + 首一二(state.dimensionsTotal):
        listRemoveLeaves.extend([首一(state.dimensionsTotal) + (一 + 零), last(getLeavesCreaseAnte(state, leafInSubHyperplane(leafAt一零)))])
    if leafAt一零 == 零 + 首零一(state.dimensionsTotal):
        listRemoveLeaves.extend([首零一二(state.dimensionsTotal)])
    if isOdd吗(leafAt一零):
        dimensionHeadSecond: int = raiseIfNone(dimensionSecondNearest首(leafAt一零))
        indexBy首Second: int = dimensionHeadSecond * decreasing + decreasing
        listRemoveLeaves.extend([getitem(state.productsOfDimensions, dimensionHeadSecond)])
        if leafAt一零 < 首零(state.dimensionsTotal):
            sumsOfProductsOfDimensionsNearest首InSubHyperplane: tuple[int, ...] = getSumsOfProductsOfDimensionsNearest首(state.productsOfDimensions, state.dimensionsTotal, state.dimensionsTotal - 1)
            listRemoveLeaves.extend([一, leafAt一零 + getitem(state.sumsOfProductsOfDimensions, state.dimensionsTotal - 1), leafAt一零 + getitem(sumsOfProductsOfDimensionsNearest首InSubHyperplane, indexBy首Second)])
            if dimensionHeadSecond == 2:
                listRemoveLeaves.extend([getitem(state.sumsOfProductsOfDimensions, dimensionHeadSecond) + getitem(state.productsOfDimensions, dimensionNearest首(leafAt一零)), getitem(state.sumsOfProductsOfDimensions, dimensionHeadSecond) + 首零(state.dimensionsTotal)])
            if dimensionHeadSecond == 3:
                listRemoveLeaves.extend([一 + leafAt一零 + getitem(state.productsOfDimensions, state.dimensionsTotal - 1)])
        if 首零(state.dimensionsTotal) < leafAt一零:
            listRemoveLeaves.extend([零 + 首零一(state.dimensionsTotal), getitem(state.productsOfDimensions, dimensionNearest首(leafAt一零) - 1)])
    listRemoveLeaves.extend([leafAt零一Ante首])
    if 首零(state.dimensionsTotal) < leafAt零一Ante首:
        listRemoveLeaves.extend([零 + 首零一(state.dimensionsTotal)])
        if isEven吗(leafAt零一Ante首):
            listRemoveLeaves.extend([首一(state.dimensionsTotal)])
            dimension: int = 一
            if isBit1吗(leafAt零一Ante首, dimensionIndex(dimension)):
                listRemoveLeaves.extend([dimension, 首零(state.dimensionsTotal) + dimension + 零, state.首 - sum(state.productsOfDimensions[dimensionIndex(dimension):state.dimensionsTotal - 2]), leafAt零一Ante首 - dimension - getitem(state.sumsOfProductsOfDimensions, dimensionIndex(dimension) + 1)])
            dimension = 二
            if isBit1吗(leafAt零一Ante首, dimensionIndex(dimension)):
                listRemoveLeaves.extend([dimension, 首零(state.dimensionsTotal) + dimension + 零])
                if 1 < dimensionNearestTail(leafAt零一Ante首):
                    listRemoveLeaves.extend([state.首 - sum(state.productsOfDimensions[dimensionIndex(dimension):state.dimensionsTotal - 2])])
                else:
                    listRemoveLeaves.extend([getitem(tuple(getLeavesCreaseAnte(state, leafInSubHyperplane(leafAt零一Ante首))), dimensionIndex(dimension)) - 零])
            dimension = 三
            if isBit1吗(leafAt零一Ante首, dimensionIndex(dimension)):
                if 1 < dimensionNearestTail(leafAt零一Ante首):
                    listRemoveLeaves.extend([dimension])
                    listRemoveLeaves.extend([state.首 - sum(state.productsOfDimensions[dimensionIndex(dimension):state.dimensionsTotal - 2])])
                if dimensionNearestTail(leafAt零一Ante首) < dimensionIndex(dimension):
                    listRemoveLeaves.extend([首零(state.dimensionsTotal) + dimension + 零])
            sheepOrGoat = 0
            shepherdOfDimensions: int = int(bit_flip(0, state.dimensionsTotal - 5))
            if leafAt零一Ante首 // shepherdOfDimensions & bit_mask(5) == 21:
                listRemoveLeaves.extend([二])
                sheepOrGoat: int = ptount(leafAt零一Ante首 // shepherdOfDimensions)
                if 0 < sheepOrGoat < state.dimensionsTotal - 3:
                    comebackOffset: int = state.productsOfDimensions[dimensionNearest首(leafAt零一Ante首)] - 二
                    listRemoveLeaves.extend([leafAt零一Ante首 - comebackOffset])
                if 0 < sheepOrGoat < state.dimensionsTotal - 4:
                    comebackOffset = state.productsOfDimensions[raiseIfNone(dimensionSecondNearest首(leafAt零一Ante首))] - 二
                    listRemoveLeaves.extend([leafAt零一Ante首 - comebackOffset])
        if isOdd吗(leafAt零一Ante首):
            listRemoveLeaves.extend([一])
            if leafAt零一Ante首 & bit_mask(4) == 9:
                listRemoveLeaves.extend([11])
            sheepOrGoat = ptount(leafAt零一Ante首)
            if 0 < sheepOrGoat < state.dimensionsTotal - 3:
                comebackOffset = state.productsOfDimensions[dimensionNearest首(leafAt零一Ante首)] - 一
                listRemoveLeaves.extend([leafAt零一Ante首 - comebackOffset])
            if 0 < sheepOrGoat < state.dimensionsTotal - 4:
                comebackOffset = state.productsOfDimensions[raiseIfNone(dimensionSecondNearest首(leafAt零一Ante首))] - 一
                listRemoveLeaves.extend([leafAt零一Ante首 - comebackOffset])
    if leafAt一 == 一 + 零 and leafAt零一Ante首 != next(getLeavesCreaseAnte(state, 零 + 首零(state.dimensionsTotal))):
        listRemoveLeaves.append(首一(state.dimensionsTotal))
    dimensionHead: int = dimensionNearest首(leafAt二)
    creasePostAt二: tuple[int, ...] = tuple(getLeavesCreasePost(state, leafAt二))
    listIndicesCreasePostToKeep: list[int] = []
    if 二 < leafAt二 < neg(零) + 首一(state.dimensionsTotal):
        listRemoveLeaves.extend([leafAt二 + 首零(state.dimensionsTotal)])
        dimension = 一
        if isBit1吗(leafAt二, dimensionIndex(dimension)):
            listRemoveLeaves.extend([leafAt二 + 首零(state.dimensionsTotal) + dimension])
        if not isBit1吗(leafAt二, dimensionIndex(dimension)):
            listRemoveLeaves.extend([leafAt二 + 首零(state.dimensionsTotal) - dimension])
        if isOdd吗(leafAt二):
            dimension = 三
            if isBit1吗(leafAt二, dimensionIndex(dimension)):
                listRemoveLeaves.extend([leafAt二 + 首零(state.dimensionsTotal) + dimension])
                dimension = 四
                if not isBit1吗(leafAt二, dimensionIndex(dimension)):
                    listRemoveLeaves.extend([leafAt二 + 首零(state.dimensionsTotal) - dimension])
    if 首一(state.dimensionsTotal) < leafAt二 < 首零(state.dimensionsTotal) and raiseIfNone(dimensionSecondNearest首(leafAt二)) != 2:
        listRemoveLeaves.extend([leafAt二 + 首零(state.dimensionsTotal)])
        if isOdd吗(leafAt二):
            dimension = 二
            if not isBit1吗(leafAt二, dimensionIndex(dimension)):
                listRemoveLeaves.extend([leafAt二 + 首零(state.dimensionsTotal) - getitem(state.sumsOfProductsOfDimensions, dimensionIndex(dimension))])
            dimension = 三
            if not isBit1吗(leafAt二, dimensionIndex(dimension)):
                listRemoveLeaves.extend([leafAt二 + 首零(state.dimensionsTotal) - dimension, leafAt二 + 首零(state.dimensionsTotal) + getitem(state.sumsOfProductsOfDimensions, dimensionIndex(dimension))])
            dimension = 四
            if isBit1吗(leafAt二, dimensionIndex(dimension)):
                listRemoveLeaves.extend([leafAt二 - dimension])
    if isEven吗(leafAt二):
        listIndicesCreasePostToKeep.extend(range(state.dimensionsTotal - dimensionHead + 1, state.dimensionsTotal - zeroIndexed))
        listRemoveLeaves.extend([leafAt二 + 零, leafAt二 + 首零(state.dimensionsTotal), leafAt二 + getitem(state.sumsOfProductsOfDimensions, state.dimensionsTotal - 1), getitem(state.productsOfDimensions, dimensionHead) + (一 + 零)])
        dimension = 一
        if isBit1吗(leafAt二, dimensionIndex(dimension)):
            listRemoveLeaves.extend([dimension, 首零(state.dimensionsTotal) + dimension + 零])
        dimension = 二
        if not isBit1吗(leafAt二, dimensionIndex(dimension)):
            listIndicesCreasePostToKeep.append(creasePostAt二.index(state.productsOfDimensions[dimensionHead]))
        if leafAt二 < 首零(state.dimensionsTotal):
            listRemoveLeaves.extend([getitem(state.productsOfDimensions, dimensionIndex(二)), getitem(state.sumsOfProductsOfDimensions, dimensionIndex(二) + 1)])
        dimension = 四
        if not isBit1吗(leafAt二, dimensionIndex(dimension)) and 首零(state.dimensionsTotal) < leafAt二:
            listRemoveLeaves.extend([getitem(state.productsOfDimensions, dimensionIndex(dimension))])
        zerosAtThe首 = 2
        if state.dimensionsTotal - zeroIndexed - dimensionHead == zerosAtThe首:
            sumsOfProductsOfDimensionsNearest首InSubSubHyperplane: tuple[int, ...] = getSumsOfProductsOfDimensionsNearest首(state.productsOfDimensions, state.dimensionsTotal, state.dimensionsTotal - zerosAtThe首)
            addendForUnknownReasons: int = -1
            leavesWeDontWant: list[int] = [aLeaf + addendForUnknownReasons for aLeaf in filter(notLeafOriginOrLeaf零, sumsOfProductsOfDimensionsNearest首InSubSubHyperplane)]
            listRemoveLeaves.extend(leavesWeDontWant)
    if isOdd吗(leafAt二):
        if dimensionNearestTail(leafAt二 - 1) == 1:
            listRemoveLeaves.extend([一])
        if leafInSubHyperplane(leafAt二) == state.sumsOfProductsOfDimensions[3]:
            listRemoveLeaves.extend([二])
        dimension = 零
        if isBit1吗(leafAt二, dimensionIndex(dimension)):
            listRemoveLeaves.extend([dimension, leafAt二 - dimension, 首零(state.dimensionsTotal) + dimension + 零])
        dimension = 二
        if not isBit1吗(leafAt二, dimensionIndex(dimension)):
            listIndicesCreasePostToKeep.append(dimensionIndex(dimension))
        if isBit1吗(leafAt二, dimensionIndex(dimension)) and isBit1吗(leafAt二, dimensionIndex(一)):
            listRemoveLeaves.extend([leafAt二 - dimension, 首零(state.dimensionsTotal) + dimension + 零])
        dimension = 三
        if isBit1吗(leafAt二, dimensionIndex(dimension)):
            listRemoveLeaves.extend([leafAt二 - dimension, 首零(state.dimensionsTotal) + dimension + 零])
        if not isBit1吗(leafAt二, dimensionIndex(dimension)):
            listIndicesCreasePostToKeep.append(dimensionIndex(dimension))
            dimension = 四
            if not isBit1吗(leafAt二, dimensionIndex(dimension)):
                listIndicesCreasePostToKeep.append(dimensionIndex(dimension))
        dimension = 四
        if isBit1吗(leafAt二, dimensionIndex(dimension)):
            dimensionBonus: int = 零
            if isBit1吗(leafAt二, dimensionIndex(dimensionBonus)):
                listRemoveLeaves.extend([首零(state.dimensionsTotal) + dimension + dimensionBonus])
            dimensionBonus = 二
            if isBit1吗(leafAt二, dimensionIndex(dimensionBonus)):
                listRemoveLeaves.extend([首零(state.dimensionsTotal) + dimension + dimensionBonus])
            dimensionBonus = 三
            if isBit1吗(leafAt二, dimensionIndex(dimensionBonus)):
                listRemoveLeaves.extend([首零(state.dimensionsTotal) + dimension + dimensionBonus])
        dimension = 五
        if isBit1吗(leafAt二, dimensionIndex(dimension)):
            listRemoveLeaves.extend([首一(state.dimensionsTotal), 零 + 首零一(state.dimensionsTotal)])
        if leafAt二 < 首一(state.dimensionsTotal):
            listRemoveLeaves.extend([一])
        if 首一(state.dimensionsTotal) < leafAt二 < 首零(state.dimensionsTotal):
            listRemoveLeaves.extend([leafAt二 + getitem(state.sumsOfProductsOfDimensions, state.dimensionsTotal - 2), 首一(state.dimensionsTotal) + (一 + 零)])
        if 首零(state.dimensionsTotal) < leafAt二:
            dimension = 二
            if isBit1吗(leafAt二, dimensionIndex(dimension)):
                listRemoveLeaves.extend([leafAt二 - dimension, 首零(state.dimensionsTotal) + dimension + 零])
            dimension = 四
            if isBit1吗(leafAt二, dimensionIndex(dimension)):
                listRemoveLeaves.extend([dimension, leafAt二 - dimension, 首零(state.dimensionsTotal) + dimension + 零, 首零一二(state.dimensionsTotal)])
                if isBit1吗(leafAt二, dimensionIndex(三)):
                    listRemoveLeaves.extend([leafAt二 - 五])
    listRemoveLeaves.extend(exclude(creasePostAt二, listIndicesCreasePostToKeep))
    dimensionHead: int = dimensionNearest首(leafAt二Ante首)
    dimensionTail: int = dimensionNearestTail(leafAt二Ante首)
    if isBit1吗(getitem(dictionaryLeafOptions, neg(二) + state.首), leafAt二Ante首 - 1):
        dimension = 三
        if not isBit1吗(leafAt二Ante首, dimensionIndex(dimension)):
            enumerateFrom1: int = zeroIndexed
            for bitToTest, leafToRemove in enumerate(tuple(getLeavesCreaseAnte(state, leafAt二Ante首 - 1)), start=enumerateFrom1):
                if isBit1吗(leafAt二Ante首, bitToTest):
                    listRemoveLeaves.extend([leafToRemove])
                if dimensionHead < bitToTest:
                    listRemoveLeaves.extend([leafToRemove])
    theLastPossibleIndexOfCreaseAnteIfCountingFromTheHead: int = 1
    if isBit1吗(leafAt二Ante首, theLastPossibleIndexOfCreaseAnteIfCountingFromTheHead):
        creaseAnteAt二Ante首: tuple[int, ...] = tuple(getLeavesCreaseAnte(state, leafAt二Ante首))
        largestPossibleLengthOfListOfCreases: int = state.dimensionsTotal - 1
        if len(creaseAnteAt二Ante首) == largestPossibleLengthOfListOfCreases:
            voodooAddend: int = 2
            if not isBit1吗(leafAt二Ante首, voodooAddend + theLastPossibleIndexOfCreaseAnteIfCountingFromTheHead):
                voodooMath: int = creaseAnteAt二Ante首[largestPossibleLengthOfListOfCreases - zeroIndexed]
                listRemoveLeaves.extend([voodooMath])
    if leafAt二Ante首 != 零 + 首一(state.dimensionsTotal):
        listRemoveLeaves.extend([零 + 首零一(state.dimensionsTotal)])
    if howManyDimensionsHaveOddParity(leafAt二Ante首) == 1:
        listRemoveLeaves.extend([leafInSubHyperplane(leafAt二Ante首)])
    dimension = 二
    if isBit1吗(leafAt二Ante首, dimensionIndex(dimension)):
        listRemoveLeaves.extend([leafAt二Ante首 - dimension])
        if isEven吗(leafAt二Ante首) or (isOdd吗(leafAt二Ante首) and dimensionIndex(dimension) < dimensionsConsecutiveAtTail(state, leafAt二Ante首)):
            listRemoveLeaves.extend([dimension])
    dimension = 三
    if isBit1吗(leafAt二Ante首, dimensionIndex(dimension)):
        listRemoveLeaves.extend([leafAt二Ante首 - dimension])
        dimension = 四
        if isEven吗(leafAt二Ante首) and (not isBit1吗(leafAt二Ante首, dimensionIndex(dimension))):
            listRemoveLeaves.extend([leafAt二Ante首 - getitem(state.sumsOfProductsOfDimensions, dimensionIndex(dimension))])
    if dimensionTail == 3:
        listRemoveLeaves.extend([getitem(state.sumsOfProductsOfDimensionsNearest首, dimensionTail)])
    if 首零(state.dimensionsTotal) < leafAt二Ante首:
        dimension = 一
        if isBit1吗(leafAt二Ante首, dimensionIndex(dimension)):
            listRemoveLeaves.extend([dimension, 首零(state.dimensionsTotal) + dimension + 零])
        if isOdd吗(leafAt二Ante首) and (not isBit1吗(leafAt二Ante首, dimensionIndex(dimension))):
            listRemoveLeaves.extend([leafAt二Ante首 - 首零(state.dimensionsTotal) - dimension])
            dimension = 二
            if isBit1吗(leafAt二Ante首, dimensionIndex(dimension)):
                listRemoveLeaves.extend([首零(state.dimensionsTotal) + getitem(state.sumsOfProductsOfDimensions, dimensionIndex(dimension))])
        dimension = 二
        if isBit1吗(leafAt二Ante首, dimensionIndex(dimension)):
            listRemoveLeaves.extend([首零(state.dimensionsTotal) + dimension + 零])
            dimension = 三
            if isEven吗(leafAt二Ante首) and isBit1吗(leafAt二Ante首, dimensionIndex(dimension)):
                listRemoveLeaves.extend([dimension])
        dimension = 四
        if isBit1吗(leafAt二Ante首, dimensionIndex(dimension)):
            listRemoveLeaves.extend([leafAt二Ante首 - dimension])
        if not isBit1吗(leafAt二Ante首, dimensionIndex(dimension)):
            listRemoveLeaves.extend([leafAt二Ante首 + dimension])
    if isOdd吗(leafAt二Ante首):
        dimension = 零
        if isBit1吗(leafAt二Ante首, dimensionIndex(dimension)):
            listRemoveLeaves.extend([一, leafAt二Ante首 - dimension, leafAt二Ante首 - getitem(state.productsOfDimensions, raiseIfNone(dimensionSecondNearest首(leafAt二Ante首)))])
    if isEven吗(leafAt二Ante首):
        dimension = 零
        if not isBit1吗(leafAt二Ante首, dimensionIndex(dimension)):
            listRemoveLeaves.extend([leafAt二Ante首 + dimension, state.productsOfDimensions[dimensionTail], leafAt二Ante首 - state.productsOfDimensions[dimensionTail]])
        dimension = 二
        if isBit1吗(leafAt二Ante首, dimensionIndex(dimension)):
            listRemoveLeaves.extend([dimension])
            if 首零(state.dimensionsTotal) < leafAt二Ante首 < 首零一二(state.dimensionsTotal):
                listRemoveLeaves.extend([leafAt二Ante首 + dimensionTail])
                if dimensionTail == 2:
                    addendIDC: int = (state.首 - leafAt二Ante首) // 2
                    listRemoveLeaves.extend([addendIDC + leafAt二Ante首])
            if leafAt二Ante首 < 首零(state.dimensionsTotal):
                listRemoveLeaves.extend([leafAt二Ante首 + state.sumsOfProductsOfDimensions[dimensionTail], state.首 - leafAt二Ante首])
        if leafAt二Ante首 < 首零(state.dimensionsTotal):
            listRemoveLeaves.extend([首一(state.dimensionsTotal), leafAt二Ante首 + state.productsOfDimensions[dimensionNearest首(leafAt二Ante首) + 1]])
            dimension = 三
            if not isBit1吗(leafAt二Ante首, dimensionIndex(dimension)):
                listRemoveLeaves.extend([dimension, leafAt二Ante首 + dimension, state.sumsOfProductsOfDimensionsNearest首[dimensionIndex(dimension)]])
        if leafAt二Ante首 != 一 + 首零(state.dimensionsTotal):
            listRemoveLeaves.extend([首一(state.dimensionsTotal)])
    del dimensionHead, dimensionTail
    return sorted(set(getIteratorOfLeaves(dictionaryLeafOptions[state.pile])).difference(set(listRemoveLeaves)))

def _byCrease2上nDimensional(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
    permutationSpaceHasNewLeaf: bool = True
    while permutationSpaceHasNewLeaf:
        permutationSpaceHasNewLeaf = False
        leafCount: int = permutationSpace.leafCount
        for (pile_k, leafSpace_k), (pile_r, leafSpace_r) in pairwise(permutationSpace.items()):
            if isLeaf吗(leafSpace_k) and isLeafOptions吗(leafSpace_r):
                pilesToUpdate: tuple[tuple[Pile, LeafOptions]] = ((pile_r, leafSpace_r),)
                leavesCrease: Iterator[Leaf] = getLeavesCreasePost(state, leafSpace_k)
            elif isLeafOptions吗(leafSpace_k) and isLeaf吗(leafSpace_r):
                pilesToUpdate = ((pile_k, leafSpace_k),)
                leavesCrease = getLeavesCreaseAnte(state, leafSpace_r)
            else:
                continue
            if not (permutationSpace := reduceLeafSpace(permutationSpace, pilesToUpdate, makeLeafAntiOptions(state.leavesTotal, set(range(state.leavesTotal)).difference(leavesCrease)))):
                return None
        if permutationSpace.leafCount < leafCount:
            permutationSpaceHasNewLeaf = True
    return permutationSpace

def _conditionalPredecessors2上nDimensional(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
    if not mapShapeIs2上nDimensions(state.mapShape, youMustBeDimensionsTallToPinThis=6):
        return permutationSpace
    leafAtPilePredecessors: dict[Leaf, dict[Pile, list[Leaf]]] = getDictionaryConditionalLeafPredecessors(state)
    permutationSpaceHasNewLeaf: bool = True
    while permutationSpaceHasNewLeaf:
        permutationSpaceHasNewLeaf = False
        leafCount: int = permutationSpace.leafCount
        for pile, leaf in DOTitems(filterPile(notPileLast(state.pileLast), filterLeaf(notLeafOriginOrLeaf零, filterLeaf(leafAtPilePredecessors.__contains__, permutationSpace.extractPinnedLeaves())))):
            if pile in leafAtPilePredecessors[leaf] and (not (permutationSpace := reduceLeafSpace(permutationSpace, DOTitems(methodcaller('extractUndeterminedPiles')(filterPile(between吗(pile + inclusive, state.pileLast - inclusive), permutationSpace, factory=PermutationSpace))), makeLeafAntiOptions(state.leavesTotal, leafAtPilePredecessors[leaf][pile])))):
                return None
        if permutationSpace.leafCount < leafCount:
            permutationSpaceHasNewLeaf = True
    return permutationSpace

def _crossedCreases2上nDimensional(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
    pileOf_kCrease: Pile = errorL33T
    pileOf_rCrease: Pile = errorL33T
    pilesForbidden: Iterable[Pile] = []
    permutationSpaceHasNewLeaf: bool = True
    generators: deque[CartesianProduct[tuple[DimensionIndex, PinnedLeaves, tuple[tuple[Pile, Leaf], tuple[Pile, Leaf]]]]] = deque()
    for dimension in range(state.dimensionsTotal):
        odd吗: Callable[[tuple[Pile, Leaf]], bool] = compose(oddLeaf2上nDimensional吗(dimension), itemgetter(1))
        grouped: dict[bool, list[tuple[Pile, Leaf]]] = toolz_groupby(odd吗, DOTitems(permutationSpace.extractPinnedLeaves()))
        parityEven: PinnedLeaves = dict(get(False, grouped, ()))
        parityOdd: PinnedLeaves = dict(get(True, grouped, ()))
        generators.append(CartesianProduct((dimension,), (parityOdd,), combinations(parityEven.items(), 2)))
        generators.append(CartesianProduct((dimension,), (parityEven,), combinations(parityOdd.items(), 2)))
    while permutationSpaceHasNewLeaf:
        permutationSpaceHasNewLeaf = False
        leafCount: int = permutationSpace.leafCount
        for dimension, leavesPinnedParityOpposite, ((pileOf_k, leaf_k), (pileOf_r, leaf_r)) in concat(generators):
            leaf_kCrease: Leaf = int(bit_flip(leaf_k, dimension))
            leaf_rCrease: Leaf = int(bit_flip(leaf_r, dimension))
            if (leaf_kCreaseIsPinned := leafPinned吗(leavesPinnedParityOpposite, leaf_kCrease)):
                pileOf_kCrease = raiseIfNone(reverseLookup(permutationSpace, leaf_kCrease))
            if (leaf_rCreaseIsPinned := leafPinned吗(leavesPinnedParityOpposite, leaf_rCrease)):
                pileOf_rCrease = raiseIfNone(reverseLookup(permutationSpace, leaf_rCrease))
            if leaf_kCreaseIsPinned and (not leaf_rCreaseIsPinned):
                leafAntiOptions: LeafOptions = makeLeafAntiOptions(state.leavesTotal, (leaf_rCrease,))
                if pileOf_k < pileOf_r < pileOf_kCrease:
                    pilesForbidden = frozenset([*range(pileOf_k), *range(pileOf_kCrease + 1, state.pileLast + inclusive)])
                elif pileOf_kCrease < pileOf_r < pileOf_k:
                    pilesForbidden = frozenset([*range(pileOf_kCrease), *range(pileOf_k + 1, state.pileLast + inclusive)])
                elif pileOf_r < pileOf_kCrease < pileOf_k or pileOf_kCrease < pileOf_k < pileOf_r:
                    pilesForbidden = range(pileOf_kCrease + 1, pileOf_k)
                elif pileOf_r < pileOf_k < pileOf_kCrease or pileOf_k < pileOf_kCrease < pileOf_r:
                    pilesForbidden = range(pileOf_k + 1, pileOf_kCrease)
            elif not leaf_kCreaseIsPinned and leaf_rCreaseIsPinned:
                leafAntiOptions = makeLeafAntiOptions(state.leavesTotal, (leaf_kCrease,))
                if pileOf_rCrease < pileOf_k < pileOf_r:
                    pilesForbidden = frozenset([*range(pileOf_rCrease), *range(pileOf_r + 1, state.pileLast + inclusive)])
                elif pileOf_r < pileOf_k < pileOf_rCrease:
                    pilesForbidden = frozenset([*range(pileOf_r), *range(pileOf_rCrease + 1, state.pileLast + inclusive)])
                elif pileOf_k < pileOf_r < pileOf_rCrease or pileOf_r < pileOf_rCrease < pileOf_k:
                    pilesForbidden = range(pileOf_r + 1, pileOf_rCrease)
                elif pileOf_k < pileOf_rCrease < pileOf_r or pileOf_rCrease < pileOf_r < pileOf_k:
                    pilesForbidden = range(pileOf_rCrease + 1, pileOf_r)
            elif leaf_kCreaseIsPinned and leaf_rCreaseIsPinned:
                if creaseViolation吗(pileOf_k, pileOf_r, pileOf_kCrease, pileOf_rCrease):
                    return None
                continue
            else:
                continue
            if not (permutationSpace := reduceLeafSpace(permutationSpace, DOTitems(filterPile(thisHasThat吗(pilesForbidden), permutationSpace.extractUndeterminedPiles())), leafAntiOptions)):
                return None
        if leafCount < permutationSpace.leafCount:
            permutationSpaceHasNewLeaf = True
    return permutationSpace

def _headsBeforeTails2上nDimensional(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
    permutationSpaceHasNewLeaf: bool = True
    while permutationSpaceHasNewLeaf:
        permutationSpaceHasNewLeaf = False
        leafCount: int = permutationSpace.leafCount
        pile1stOpen: int = 2
        for pile, leaf in DOTitems(filterPile(notPileLast(state.pileLast), filterLeaf(notLeafOriginOrLeaf零, permutationSpace.extractPinnedLeaves()))):
            dimensionHead: int = dimensionNearest首(leaf)
            if 0 < dimensionHead and (not (permutationSpace := reduceLeafSpace(permutationSpace, DOTitems(methodcaller('extractUndeterminedPiles')(filterPile(between吗(pile1stOpen, pile - inclusive), permutationSpace, factory=PermutationSpace))), makeLeafAntiOptions(state.leavesTotal, range(state.productsOfDimensions[dimensionHead], state.leavesTotal, state.productsOfDimensions[dimensionHead]))))):
                return None
            dimensionTail: int = dimensionNearestTail(leaf)
            if 0 < dimensionTail and (not (permutationSpace := reduceLeafSpace(permutationSpace, DOTitems(methodcaller('extractUndeterminedPiles')(filterPile(between吗(pile + inclusive, state.pileLast - inclusive), permutationSpace, factory=PermutationSpace))), makeLeafAntiOptions(state.leavesTotal, range(leafOrigin, state.sumsOfProductsOfDimensions[dimensionTail]))))):
                return None
        if permutationSpace.leafCount < leafCount:
            permutationSpaceHasNewLeaf = True
    return permutationSpace

def _noConsecutiveDimensions2上nDimensional(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
    permutationSpaceHasNewLeaf: bool = True
    while permutationSpaceHasNewLeaf:
        permutationSpaceHasNewLeaf = False
        leafCount: int = permutationSpace.leafCount
        for (pile_k, leafSpace_k), (pile, leafSpace), (pile_r, leafSpace_r) in triplewise(sorted(DOTitems(permutationSpace))):
            if isLeaf吗(leafSpace_k) and isLeaf吗(leafSpace) and isLeafOptions吗(leafSpace_r):
                pilesToUpdate: tuple[tuple[Pile, LeafOptions]] = ((pile_r, leafSpace_r),)
                leafForbidden: Leaf = leafSpace + (leafSpace - leafSpace_k)
            elif isLeaf吗(leafSpace_k) and isLeafOptions吗(leafSpace) and isLeaf吗(leafSpace_r):
                pilesToUpdate = ((pile, leafSpace),)
                leafForbidden = (leafSpace_k + leafSpace_r) // 2
            elif isLeafOptions吗(leafSpace_k) and isLeaf吗(leafSpace) and isLeaf吗(leafSpace_r):
                pilesToUpdate = ((pile_k, leafSpace_k),)
                leafForbidden = leafSpace - (leafSpace_r - leafSpace)
            else:
                continue
            if 0 <= leafForbidden < state.leavesTotal and (not (permutationSpace := reduceLeafSpace(permutationSpace, pilesToUpdate, makeLeafAntiOptions(state.leavesTotal, [leafForbidden])))):
                return None
        if permutationSpace.leafCount < leafCount:
            permutationSpaceHasNewLeaf = True
    return permutationSpace
listFunctionsReduction2上nDimensional: Sequence[Callable[[EliminationState, PermutationSpace], PermutationSpace | None]] = (reducePermutationSpace_LeafIsPinned, _byCrease2上nDimensional, reducePermutationSpace_leafDomainOf1, reducePermutationSpace_nakedSubset, _headsBeforeTails2上nDimensional, _conditionalPredecessors2上nDimensional, _crossedCreases2上nDimensional, _noConsecutiveDimensions2上nDimensional)
_dimensionLength: int = 2
_dimensionIndex: DimensionIndex = 0
零: int = _dimensionLength ** _dimensionIndex
_base: int = _dimensionLength
_dimensionIndex += 1
_power: int = _dimensionIndex
一: int = _base ** _power
_radix: int = _dimensionLength
_dimensionIndex += 1
_place_ValueIndex: int = _dimensionIndex
二: int = _radix ** _place_ValueIndex
三: int = _dimensionLength ** 3
四: int = _dimensionLength ** 4
五: int = _dimensionLength ** 5
六: int = _dimensionLength ** 6
七: int = _dimensionLength ** 7
八: int = _dimensionLength ** 8
九: int = _dimensionLength ** 9

@cache
def dimensionIndex(dimensionAsNonnegativeInteger: int, /, *, dimensionLength: int = _dimensionLength) -> DimensionIndex:
    return int(log(dimensionAsNonnegativeInteger, dimensionLength))

@cache
def 首零(dimensionsTotal: int, /) -> int:
    return int('1' + '0' * (dimensionsTotal - 1), _dimensionLength)

@cache
def 首零一(dimensionsTotal: int, /) -> int:
    return int('11' + '0' * (dimensionsTotal - 2), _dimensionLength)

@cache
def 首零一二(dimensionsTotal: int, /) -> int:
    return int('111' + '0' * (dimensionsTotal - 3), _dimensionLength)

@cache
def 首零二(dimensionsTotal: int, /) -> int:
    return int('101' + '0' * (dimensionsTotal - 3), _dimensionLength)

@cache
def 首一(dimensionsTotal: int, /) -> int:
    return int('01' + '0' * (dimensionsTotal - 2), _dimensionLength)

@cache
def 首一二(dimensionsTotal: int, /) -> int:
    return int('011' + '0' * (dimensionsTotal - 3), _dimensionLength)

@cache
def 首二(dimensionsTotal: int, /) -> int:
    return int('001' + '0' * (dimensionsTotal - 3), _dimensionLength)

@cache
def 首三(dimensionsTotal: int, /) -> int:
    return int('0001' + '0' * (dimensionsTotal - 4), _dimensionLength)

@cache
def 首零一二三(dimensionsTotal: int, /) -> int:
    return int('1111' + '0' * (dimensionsTotal - 4), _dimensionLength)

@cache
def 首零一三(dimensionsTotal: int, /) -> int:
    return int('1101' + '0' * (dimensionsTotal - 4), _dimensionLength)

@cache
def 首零二三(dimensionsTotal: int, /) -> int:
    return int('1011' + '0' * (dimensionsTotal - 4), _dimensionLength)

@cache
def 首零三(dimensionsTotal: int, /) -> int:
    return int('1001' + '0' * (dimensionsTotal - 4), _dimensionLength)

@cache
def 首一二三(dimensionsTotal: int, /) -> int:
    return int('0111' + '0' * (dimensionsTotal - 4), _dimensionLength)

@cache
def 首一三(dimensionsTotal: int, /) -> int:
    return int('0101' + '0' * (dimensionsTotal - 4), _dimensionLength)

@cache
def 首二三(dimensionsTotal: int, /) -> int:
    return int('0011' + '0' * (dimensionsTotal - 4), _dimensionLength)

def mapShapeIs2上nDimensions(mapShape: tuple[int, ...], *, youMustBeDimensionsTallToPinThis: int = 3) -> bool:
    return youMustBeDimensionsTallToPinThis <= len(mapShape) and all(map(2 .__eq__, mapShape))

def dimensionsConsecutiveAtTail(state: EliminationState, integerNonnegative: int) -> int:
    return bit_scan1(invertLeafIn2上nDimensions(state.dimensionsTotal, integerNonnegative)) or 0

@cache
def dimensionNearest首(integerNonnegative: int, /) -> int:
    return max(0, integerNonnegative.bit_length() - 1)

@cache
def dimensionSecondNearest首(integerNonnegative: int, /) -> int | None:
    anotherInteger: int = int(bit_flip(integerNonnegative, dimensionNearest首(integerNonnegative)))
    if anotherInteger == 0:
        dimensionSecondNearest: int | None = None
    else:
        dimensionSecondNearest = dimensionNearest首(anotherInteger)
    return dimensionSecondNearest

@cache
def dimensionThirdNearest首(integerNonnegative: int, /) -> int | None:
    dimensionNearest: int = dimensionNearest首(integerNonnegative)
    dimensionSecondNearest: int | None = dimensionSecondNearest首(integerNonnegative)
    if dimensionSecondNearest in {0, None}:
        dimensionThirdNearest: int | None = None
    else:
        anotherInteger: int = int(bit_flip(integerNonnegative, dimensionNearest).bit_flip(raiseIfNone(dimensionSecondNearest)))
        if anotherInteger == 0:
            dimensionThirdNearest = None
        else:
            dimensionThirdNearest = dimensionNearest首(anotherInteger)
    return dimensionThirdNearest

@cache
def dimensionFourthNearest首(integerNonnegative: int, /) -> int | None:
    dimensionNearest: int = dimensionNearest首(integerNonnegative)
    dimensionSecondNearest: int | None = dimensionSecondNearest首(integerNonnegative)
    dimensionThirdNearest: int | None = dimensionThirdNearest首(integerNonnegative)
    if dimensionThirdNearest in {0, None}:
        dimensionFourthNearest: int | None = None
    else:
        anotherInteger: int = int(bit_flip(integerNonnegative, dimensionNearest).bit_flip(raiseIfNone(dimensionSecondNearest)).bit_flip(raiseIfNone(dimensionThirdNearest)))
        if anotherInteger == 0:
            dimensionFourthNearest = None
        else:
            dimensionFourthNearest = dimensionNearest首(anotherInteger)
    return dimensionFourthNearest

@cache
def leafInSubHyperplane(notLeafOrigin: int, /) -> int:
    return int(f_mod_2exp(notLeafOrigin, dimensionNearest首(notLeafOrigin)))

@cache
def dimensionNearestTail(integerNonnegative: int, /) -> int:
    return bit_scan1(integerNonnegative) or 0

@cache
def howManyDimensionsHaveOddParity(integerNonnegative: int, /) -> int:
    return max(0, integerNonnegative.bit_count() - 1)

@cache
def invertLeafIn2上nDimensions(dimensionsTotal: int, integerNonnegative: int) -> int:
    return int(integerNonnegative ^ bit_mask(dimensionsTotal))

@cache
def ptount(integerAbove3: int, /) -> int:
    return leafInSubHyperplane(integerAbove3 - (一 + 零)).bit_count()

def getLeavesCreaseAnte(state: EliminationState, leaf: Leaf) -> Iterator[Leaf]:
    return iter(_getCreases(state, leaf, increase=False))

def getLeavesCreasePost(state: EliminationState, leaf: Leaf) -> Iterator[Leaf]:
    return iter(_getCreases(state, leaf, increase=True))

def _getCreases(state: EliminationState, leaf: Leaf, *, increase: bool = True) -> tuple[Leaf, ...]:
    return _makeCreases(leaf, state.dimensionsTotal)[increase]

@cache
def _makeCreases(leaf: Leaf, dimensionsTotal: int) -> tuple[tuple[Leaf, ...], tuple[Leaf, ...]]:
    listLeavesCrease: list[Leaf] = [int(bit_flip(leaf, dimension)) for dimension in range(dimensionsTotal)]
    if leaf == leafOrigin:
        listLeavesCreasePost: list[Leaf] = [1]
        listLeavesCreaseAnte: list[Leaf] = []
    else:
        slicingIndices: int = isOdd吗(howManyDimensionsHaveOddParity(leaf))
        slicerAnte: slice = slice(slicingIndices, dimensionNearest首(leaf) * bit_flip(slicingIndices, 0) or None)
        slicerPost: slice = slice(bit_flip(slicingIndices, 0), dimensionNearest首(leaf) * slicingIndices or None)
        if isEven吗(leaf):
            if slicerAnte.start == 1:
                slicerAnte = slice(slicerAnte.start + dimensionNearestTail(leaf), slicerAnte.stop)
            if slicerPost.start == 1:
                slicerPost = slice(slicerPost.start + dimensionNearestTail(leaf), slicerPost.stop)
        listLeavesCreaseAnte: list[Leaf] = listLeavesCrease[slicerAnte]
        listLeavesCreasePost: list[Leaf] = listLeavesCrease[slicerPost]
        if leaf == 1:
            listLeavesCreaseAnte = [0]
    return (tuple(listLeavesCreaseAnte), tuple(listLeavesCreasePost))

@cache
def _getLeafDomain(leaf: Leaf, dimensionsTotal: int, mapShape: tuple[int, ...], leavesTotal: int) -> range:
    state: EliminationState = EliminationState(mapShape)
    if mapShapeIs2上nDimensions(state.mapShape):
        originPinned: bool = leaf == leafOrigin
        return range(state.sumsOfProductsOfDimensions[dimensionNearestTail(leaf) + inclusive] + howManyDimensionsHaveOddParity(leaf) - originPinned, state.sumsOfProductsOfDimensionsNearest首[dimensionNearest首(leaf)] + 2 - howManyDimensionsHaveOddParity(leaf) - originPinned, 2 + 2 * (leaf == 首零(dimensionsTotal) + 零))
    return range(leavesTotal)

def getDomainDimension一(state: EliminationState) -> tuple[tuple[int, int, int, int], ...]:
    domain一零: tuple[int, ...] = tuple(getLeafDomain(state, 一 + 零))
    domain首一: tuple[int, ...] = tuple(getLeafDomain(state, 首一(state.dimensionsTotal)))
    return _getDomainDimension一(domain一零, domain首一, state.dimensionsTotal)

@cache
def _getDomainDimension一(domain一零: tuple[int, ...], domain首一: tuple[int, ...], dimensionsTotal: int) -> tuple[tuple[int, int, int, int], ...]:
    domainCombined: list[tuple[int, int, int, int]] = []
    for pileOfLeaf一零 in domain一零:
        domainOfLeaf首一: tuple[int, ...] = domain首一
        pilesTotal: int = len(domainOfLeaf首一)
        listIndicesPilesExcluded: list[int] = []
        if pileOfLeaf一零 <= 首二(dimensionsTotal):
            pass
        elif 首二(dimensionsTotal) < pileOfLeaf一零 < 首一(dimensionsTotal):
            listIndicesPilesExcluded.extend([*range(1, pilesTotal // 2), *range(1 + pilesTotal // 2, 3 * pilesTotal // 4)])
        elif pileOfLeaf一零 == 首一(dimensionsTotal):
            listIndicesPilesExcluded.extend([*range(1, pilesTotal // 2)])
        elif 首一(dimensionsTotal) < pileOfLeaf一零 < 首零(dimensionsTotal) - 一:
            listIndicesPilesExcluded.extend([*range(3 * pilesTotal // 4)])
        elif pileOfLeaf一零 == 首零(dimensionsTotal) - 一:
            listIndicesPilesExcluded.extend([*range(1, 3 * pilesTotal // 4)])
        elif pileOfLeaf一零 == 首零(dimensionsTotal):
            listIndicesPilesExcluded.extend([*range(2, pilesTotal // 2)])
        domainOfLeaf首一 = tuple(exclude(domainOfLeaf首一, listIndicesPilesExcluded))
        domainCombined.extend([(pileOfLeaf一零, pileOfLeaf一零 + 1, pileOfLeaf首一, pileOfLeaf首一 + 1) for pileOfLeaf首一 in domainOfLeaf首一])
    return tuple(filter(allUnique吗, domainCombined))

def getDomainDimension二(state: EliminationState) -> tuple[tuple[int, int, int, int], ...]:
    domain二零and二: tuple[tuple[int, int], ...] = getDomain二零and二(state)
    domain二一零and二一: tuple[tuple[int, int], ...] = getDomain二一零and二一(state)
    return _getDomainDimension二(domain二零and二, domain二一零and二一, state.dimensionsTotal)

@cache
def _getDomainDimension二(domain二零and二: tuple[tuple[int, int], ...], domain二一零and二一: tuple[tuple[int, int], ...], dimensionsTotal: int) -> tuple[tuple[int, int, int, int], ...]:
    domain0corners: tuple[tuple[int, int], ...] = tuple(filter(consecutive吗, domain二零and二))
    domain一corners: tuple[tuple[int, int], ...] = tuple(filter(consecutive吗, domain二一零and二一))
    pilesTotal: int = len(domain一corners)
    domainCombined: list[tuple[int, int, int, int]] = []
    productsOfDimensions: tuple[int, ...] = tuple(int(bit_flip(0, dimension)) for dimension in range(dimensionsTotal + 1))
    for index, (pileOfLeaf二一零, pileOfLeaf二一) in enumerate(domain一corners):
        listIndicesPilesExcluded: list[int] = []
        dimensionTail: int = dimensionNearestTail(pileOfLeaf二一)
        excludeBelow: int = index
        listIndicesPilesExcluded.extend(range(excludeBelow))
        excludeAbove: int = pilesTotal
        if pileOfLeaf二一 <= 首一(dimensionsTotal):
            if dimensionTail == 1:
                excludeAbove = pilesTotal // 2 + index
                if howManyDimensionsHaveOddParity(pileOfLeaf二一) == 2:
                    excludeAbove -= 1
                if howManyDimensionsHaveOddParity(pileOfLeaf二一) == 1 and 2 < dimensionNearest首(pileOfLeaf二一):
                    excludeAbove += 2
                if howManyDimensionsHaveOddParity(pileOfLeaf二一) == 1 and dimensionNearest首(pileOfLeaf二一) - raiseIfNone(dimensionSecondNearest首(pileOfLeaf二一)) < 2:
                    addend: int = productsOfDimensions[dimensionsTotal - 2] + 4
                    excludeAbove = domain0corners.index((pileOfLeaf二一 + addend, pileOfLeaf二一零 + addend))
            else:
                excludeAbove = 3 * pilesTotal // 4 + 2
                if index == 0:
                    excludeAbove = 1
                elif index <= 2:
                    addend = 三 + sum(productsOfDimensions[1:dimensionsTotal - 2])
                    excludeAbove = domain0corners.index((pileOfLeaf二一 + addend, pileOfLeaf二一零 + addend))
        listIndicesPilesExcluded.extend(range(excludeAbove, pilesTotal))
        if pileOfLeaf二一 < 首一二(dimensionsTotal):
            if dimensionTail == 4:
                addend = int(bit_flip(0, dimensionTail))
                start: int = domain0corners.index((pileOfLeaf二一 + addend, pileOfLeaf二一零 + addend))
                listIndicesPilesExcluded.extend([*range(start, start + dimensionTail)])
            if dimensionTail == 3:
                addend = int(bit_flip(0, dimensionTail))
                start = domain0corners.index((pileOfLeaf二一 + addend, pileOfLeaf二一零 + addend))
                listIndicesPilesExcluded.extend([*range(start, start + dimensionTail - 1)])
                start = domain0corners.index((pileOfLeaf二一 + addend * 2, pileOfLeaf二一零 + addend * 2))
                listIndicesPilesExcluded.extend([*range(start - 1, start + dimensionTail - 1)])
            if dimensionTail < 3 and 2 < dimensionNearest首(pileOfLeaf二一):
                if 5 < dimensionsTotal:
                    addend = 四
                    start = domain0corners.index((pileOfLeaf二一 + addend, pileOfLeaf二一零 + addend))
                    stop: int = start + addend
                    step: int = 2
                    if dimensionTail == 1 and dimensionNearest首(pileOfLeaf二一) == 4:
                        start += 2
                        stop = start + 1
                    if dimensionTail == 2:
                        start += 3
                        if dimensionNearest首(pileOfLeaf二一) == 4:
                            start -= 2
                        stop = start + dimensionTail + inclusive
                    if howManyDimensionsHaveOddParity(pileOfLeaf二一) == 2:
                        stop = start + 1
                    listIndicesPilesExcluded.extend([*range(start, stop, step)])
                if (dimensionNearest首(pileOfLeaf二一) == 3 and howManyDimensionsHaveOddParity(pileOfLeaf二一) == 1) or dimensionNearest首(pileOfLeaf二一) - raiseIfNone(dimensionSecondNearest首(pileOfLeaf二一)) == 3:
                    addend = pileOfLeaf二一
                    start = domain0corners.index((pileOfLeaf二一 + addend, pileOfLeaf二一零 + addend))
                    stop = start + 2
                    if dimensionTail == 2:
                        start += 1
                        stop += 1
                    if dimensionNearest首(pileOfLeaf二一) == 4:
                        start += 3
                        stop += 4
                    step = 1
                    listIndicesPilesExcluded.extend([*range(start, stop, step)])
            if dimensionNearest首(pileOfLeaf二一) == 2:
                addend = 三
                start = domain0corners.index((pileOfLeaf二一 + addend, pileOfLeaf二一零 + addend))
                listIndicesPilesExcluded.extend([*range(start, start + addend, 2)])
        domainCombined.extend([(pileOfLeaf二一, pileOfLeaf二一零, pileOfLeaf二零, pileOfLeaf二) for pileOfLeaf二零, pileOfLeaf二 in exclude(domain0corners, listIndicesPilesExcluded)])
    domain一nonCorners: tuple[tuple[int, int], ...] = tuple(set(domain二一零and二一).difference(set(domain一corners)))
    domainCombined.extend([(pileOfLeaf一二, pileOfLeaf二一零, pileOfLeaf二一零 - 1, pileOfLeaf一二 + 1) for pileOfLeaf二一零, pileOfLeaf一二 in domain一nonCorners])
    return tuple(sorted(filter(allUnique吗, set(domainCombined))))

def getDomainDimension首二(state: EliminationState) -> tuple[tuple[int, int, int, int], ...]:
    domain首零二and首二: tuple[tuple[int, int], ...] = getDomain首零二and首二(state)
    domain首零一二and首一二: tuple[tuple[int, int], ...] = getDomain首零一二and首一二(state)
    return _getDomainDimension首二(state.dimensionsTotal, domain首零二and首二, domain首零一二and首一二)

@cache
def _getDomainDimension首二(dimensionsTotal: int, domain首零二and首二: tuple[tuple[int, int], ...], domain首零一二and首一二: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int, int, int], ...]:
    domain0corners: tuple[tuple[int, int], ...] = tuple(filter(consecutive吗, domain首零二and首二))
    domain一corners: tuple[tuple[int, int], ...] = tuple(filter(consecutive吗, domain首零一二and首一二))
    pilesTotal: Leaf = len(domain一corners)
    domainCombined: list[tuple[int, int, int, int]] = []
    for index, (pileOfLeaf首零二, pileOfLeaf首二) in enumerate(domain0corners):
        listIndicesPilesExcluded: list[int] = []
        dimensionTail: int = dimensionNearestTail(pileOfLeaf首零二)
        excludeBelow: int = index - 1
        listIndicesPilesExcluded.extend(range(excludeBelow))
        excludeAbove: int = pilesTotal
        if dimensionTail == 1:
            excludeAbove = pilesTotal - (int(pileOfLeaf首二 ^ bit_mask(dimensionsTotal)) // 4 - 1)
            if howManyDimensionsHaveOddParity(pileOfLeaf首二) == 3 and dimensionsTotal - dimensionNearest首(pileOfLeaf首二) >= 2:
                excludeAbove += 2
            if howManyDimensionsHaveOddParity(pileOfLeaf首二) == 1 and dimensionsTotal - dimensionNearest首(pileOfLeaf首二) >= 2 and (dimensionNearest首(pileOfLeaf首二) - raiseIfNone(dimensionSecondNearest首(pileOfLeaf首二)) > 3):
                excludeAbove += 2
            if howManyDimensionsHaveOddParity(pileOfLeaf首二) == 1 and dimensionNearest首(pileOfLeaf首二) - raiseIfNone(dimensionSecondNearest首(pileOfLeaf首二)) > 4:
                excludeAbove += 2
            if howManyDimensionsHaveOddParity(pileOfLeaf首二) == dimensionsTotal - dimensionNearest首(pileOfLeaf首二) and 4 <= dimensionNearest首(pileOfLeaf首二) and (howManyDimensionsHaveOddParity(pileOfLeaf首二) > 1):
                excludeAbove -= 1
        else:
            if 首零二(dimensionsTotal) <= pileOfLeaf首零二:
                excludeAbove = pilesTotal - 1
            if 首零(dimensionsTotal) < pileOfLeaf首零二 < 首零二(dimensionsTotal):
                excludeAbove = pilesTotal - (int(pileOfLeaf首零二 ^ bit_mask(dimensionsTotal)) // 8 - 1)
            if 首一二(dimensionsTotal) < pileOfLeaf首零二 <= 首零(dimensionsTotal):
                excludeAbove = pilesTotal - int(bit_mask(dimensionsTotal - 4))
            if pileOfLeaf首零二 == 首一二(dimensionsTotal):
                excludeAbove = pilesTotal - int(bit_mask(dimensionsTotal - 4)) - 1
            if pileOfLeaf首零二 < 首一二(dimensionsTotal):
                excludeAbove = pilesTotal - int(bit_mask(dimensionsTotal - 3)) - (dimensionTail == 2)
        listIndicesPilesExcluded.extend(range(excludeAbove, pilesTotal))
        if dimensionTail == 1 and abs(pileOfLeaf首零二 - 首零(dimensionsTotal)) == 2 and isEven吗(dimensionsTotal):
            listIndicesPilesExcluded.extend([excludeAbove - 2])
        if dimensionTail != 1 and 首一二(dimensionsTotal) <= pileOfLeaf首零二 <= 首零一(dimensionsTotal):
            if dimensionTail == 2 and howManyDimensionsHaveOddParity(pileOfLeaf首零二) + 1 != dimensionNearest首(pileOfLeaf首零二) - raiseIfNone(dimensionSecondNearest首(pileOfLeaf首零二)):
                listIndicesPilesExcluded.extend([pilesTotal - (int(pileOfLeaf首零二 ^ bit_mask(dimensionsTotal)) // 8 + 2)])
                if pileOfLeaf首零二 <= 首零(dimensionsTotal) and isEven吗(dimensionsTotal):
                    listIndicesPilesExcluded.extend([pilesTotal - (int(pileOfLeaf首零二 ^ bit_mask(dimensionsTotal)) // 4 - 1)])
            if dimensionTail == 3:
                listIndicesPilesExcluded.extend([excludeAbove - 2])
            if 3 < dimensionTail:
                listIndicesPilesExcluded.extend([pilesTotal - int(pileOfLeaf首零二 ^ bit_mask(dimensionsTotal)) // 4])
        domainCombined.extend([(pileOfLeaf首二, pileOfLeaf首零二, pileOfLeaf首零一二, pileOfLeaf首一二) for pileOfLeaf首零一二, pileOfLeaf首一二 in exclude(domain一corners, listIndicesPilesExcluded)])
    domain0nonCorners: tuple[tuple[int, int], ...] = tuple(set(domain首零二and首二).difference(set(domain0corners)))
    domainCombined.extend([(pileOfLeaf首二, pileOfLeaf首零二, pileOfLeaf首零二 - 1, pileOfLeaf首二 + 1) for pileOfLeaf首零二, pileOfLeaf首二 in domain0nonCorners])
    return tuple(sorted(filter(allUnique吗, set(domainCombined))))

def getDomain二零and二(state: EliminationState) -> tuple[tuple[int, int], ...]:
    domain二零: tuple[int, ...] = tuple(getLeafDomain(state, 二 + 零))
    domain二: tuple[int, ...] = tuple(getLeafDomain(state, 二))
    direction: CallableFunction[[int, int], int] = add
    return _getDomains二Or二一(domain二零, domain二, direction, state.dimensionsTotal, state.sumsOfProductsOfDimensions)

def getDomain二一零and二一(state: EliminationState) -> tuple[tuple[int, int], ...]:
    domain二一零: tuple[int, ...] = tuple(getLeafDomain(state, 二 + 一 + 零))
    domain二一: tuple[int, ...] = tuple(getLeafDomain(state, 二 + 一))
    direction: CallableFunction[[int, int], int] = sub
    return _getDomains二Or二一(domain二一零, domain二一, direction, state.dimensionsTotal, state.sumsOfProductsOfDimensions)

@cache
def _getDomains二Or二一(domain零: tuple[int, ...], domain0: tuple[int, ...], direction: CallableFunction[[int, int], int], dimensionsTotal: int, sumsOfProductsOfDimensions: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    if direction(0, 6009) == 6009:
        ImaDomain二零and二: bool = True
        ImaDomain二一零and二一: bool = False
    else:
        ImaDomain二零and二 = False
        ImaDomain二一零and二一 = True
    domainCombined: list[tuple[int, int]] = []
    pilesTotal: int = len(domain零)
    pilesFewerDomain0: int = pilesTotal - len(domain0)
    for indexDomain零, pileOfLeaf零 in enumerate(filter(between吗(pileOrigin, 首零(dimensionsTotal) - 零), domain零)):
        indicesDomain0ToExclude: list[int] = []
        dimensionTail: int = dimensionNearestTail(pileOfLeaf零 - isOdd吗(pileOfLeaf零))
        excludeBelowAddend: int = 0
        steppingBasisForUnknownReasons: int = indexDomain零
        if ImaDomain二零and二:
            excludeBelowAddend = 0
            steppingBasisForUnknownReasons = int(bit_mask(dimensionTail - 1).bit_flip(0))
        elif ImaDomain二一零and二一:
            excludeBelowAddend = int(isEven吗(indexDomain零) or dimensionTail)
            steppingBasisForUnknownReasons = indexDomain零
        if ImaDomain二零and二:
            if pileOfLeaf零 == 二:
                indicesDomain0ToExclude.extend([*range(indexDomain零 + 1)])
            if pileOfLeaf零 == 首一(dimensionsTotal) + 首二(dimensionsTotal) + 首三(dimensionsTotal):
                indexDomain0: int = int(7 * pilesTotal / 8)
                indexDomain0 -= pilesFewerDomain0
                indicesDomain0ToExclude.extend([indexDomain0])
        excludeBelow: int = indexDomain零 + excludeBelowAddend
        excludeBelow -= pilesFewerDomain0
        indicesDomain0ToExclude.extend(range(excludeBelow))
        if pileOfLeaf零 <= 首一(dimensionsTotal):
            excludeAbove: int = indexDomain零 + 3 * pilesTotal // 4
            excludeAbove -= pilesFewerDomain0
            indicesDomain0ToExclude.extend(range(excludeAbove, pilesTotal))
        if 首一(dimensionsTotal) < pileOfLeaf零 < 首零(dimensionsTotal):
            excludeAbove = int(pileOfLeaf零 ^ bit_mask(dimensionsTotal)) // 2
            indicesDomain0ToExclude.extend(range(excludeAbove, pilesTotal))
        for dimension in range(dimensionTail):
            indicesDomain0ToExclude.extend(range(steppingBasisForUnknownReasons + int(bit_mask(dimension)), pilesTotal, int(bit_flip(0, dimension + 1))))
        if dimensionTail == 1:
            if 首二(dimensionsTotal) < pileOfLeaf零 < 首零(dimensionsTotal) - 零 and 2 < dimensionNearest首(pileOfLeaf零):
                if dimensionSecondNearest首(pileOfLeaf零) == 零:
                    indexDomain0: int = pilesTotal // 2
                    indexDomain0 -= pilesFewerDomain0
                    if 4 < domain0[indexDomain0].bit_length():
                        indicesDomain0ToExclude.extend([indexDomain0])
                    if 首一(dimensionsTotal) < pileOfLeaf零:
                        indexDomain0 = -(pilesTotal // 4 - isOdd吗(pileOfLeaf零))
                        indexDomain0 -= -pilesFewerDomain0
                        indicesDomain0ToExclude.extend([indexDomain0])
                if dimensionSecondNearest首(pileOfLeaf零) == 一:
                    indexDomain0 = pilesTotal // 2 + 2
                    indexDomain0 -= pilesFewerDomain0
                    if domain0[indexDomain0] < 首零(dimensionsTotal):
                        indicesDomain0ToExclude.extend([indexDomain0])
                    indexDomain0 = -(pilesTotal // 4 - 2)
                    indexDomain0 -= -pilesFewerDomain0
                    if 首一(dimensionsTotal) < pileOfLeaf零:
                        indicesDomain0ToExclude.extend([indexDomain0])
                if dimensionSecondNearest首(pileOfLeaf零) == 一 + 零:
                    indexDomain0 = -(pilesTotal // 4)
                    indexDomain0 -= -pilesFewerDomain0
                    indicesDomain0ToExclude.extend([indexDomain0])
                indexDomain0 = 3 * pilesTotal // 4
                indexDomain0 -= pilesFewerDomain0
                if pileOfLeaf零 < 首一二(dimensionsTotal):
                    dimensionIndexPart首: int = dimensionsTotal
                    dimensionIndexPart一: int = dimensionIndex(一)
                    dimensionIndexPart二: int = dimensionIndex(二)
                    indexSumsOfProductsOfDimensions: int = dimensionIndexPart首 - (dimensionIndexPart一 + dimensionIndexPart二)
                    addend: int = sumsOfProductsOfDimensions[indexSumsOfProductsOfDimensions]
                    if ImaDomain二一零and二一:
                        addend -= 1
                    pileOfLeaf0: int = addend + 首零(dimensionsTotal)
                    indexDomain0 = domain0.index(pileOfLeaf0)
                    indicesDomain0ToExclude.extend([indexDomain0])
                if dimensionThirdNearest首(pileOfLeaf零) == 零:
                    if dimensionSecondNearest首(pileOfLeaf零) == 一 + 零:
                        indicesDomain0ToExclude.extend([indexDomain0 - 2])
                    if dimensionNearest首(pileOfLeaf零) == 一 + 零:
                        indicesDomain0ToExclude.extend([indexDomain0 - 2])
        elif 首一(dimensionsTotal) + 首三(dimensionsTotal) + isOdd吗(pileOfLeaf零) == pileOfLeaf零:
            indexDomain0 = 3 * pilesTotal // 4 - 1
            indexDomain0 -= pilesFewerDomain0
            indicesDomain0ToExclude.extend([indexDomain0])
        domainCombined.extend([(pileOfLeaf零, pileOfLeaf0) for pileOfLeaf0 in exclude(domain0, indicesDomain0ToExclude)])
    domainCombined.extend([(pile, direction(pile, 零)) for pile in domain零 if direction(pile, 零) in domain0])
    return tuple(sorted(set(domainCombined)))

def getDomain首零二and首二(state: EliminationState) -> tuple[tuple[int, int], ...]:
    domain首零二: tuple[int, ...] = tuple(getLeafDomain(state, 首零二(state.dimensionsTotal)))
    domain首二: tuple[int, ...] = tuple(getLeafDomain(state, 首二(state.dimensionsTotal)))
    return _getDomain首零二and首二(domain首零二, domain首二, state.dimensionsTotal)

@cache
def _getDomain首零二and首二(domain首零二: tuple[int, ...], domain首二: tuple[int, ...], dimensionsTotal: int) -> tuple[tuple[int, int], ...]:
    domainCombined: list[tuple[int, int]] = []
    domain零: tuple[int, ...] = domain首零二
    domain0: tuple[int, ...] = domain首二
    direction: CallableFunction[[int, int], int] = sub
    domainCombined.extend([(pile, direction(pile, 零)) for pile in domain零 if direction(pile, 零) in domain0])
    pilesTotal: int = len(domain零)
    pilesFewerDomain0: int = pilesTotal - len(domain0)
    for index, pileOfLeaf零 in enumerate(domain零):
        if pileOfLeaf零 < 首零(dimensionsTotal) + 零:
            continue
        listIndicesPilesExcluded: list[int] = []
        dimensionTail: int = dimensionNearestTail(direction(pileOfLeaf零, isOdd吗(pileOfLeaf零)))
        if 首零一(dimensionsTotal) < pileOfLeaf零:
            excludeBelow: int = index + 3 - 3 * pilesTotal // 4
        else:
            excludeBelow = 2 + (首零一(dimensionsTotal) - direction(pileOfLeaf零, isOdd吗(pileOfLeaf零))) // 2
        excludeBelow -= pilesFewerDomain0
        listIndicesPilesExcluded.extend(range(excludeBelow))
        excludeAbove: int = index + 2 - int(bit_mask(dimensionTail))
        excludeAbove -= pilesFewerDomain0
        listIndicesPilesExcluded.extend(range(excludeAbove, pilesTotal))
        countFromTheEnd: int = pilesTotal - 1
        countFromTheEnd -= pilesFewerDomain0
        steppingBasisForUnknownReasons: int = countFromTheEnd - int(bit_mask(dimensionTail - 1).bit_flip(0))
        for dimension in range(dimensionTail):
            listIndicesPilesExcluded.extend(range(steppingBasisForUnknownReasons - int(bit_mask(dimension)), decreasing, decreasing * int(bit_flip(0, dimension + 1))))
        if dimensionTail == 1:
            if dimensionThirdNearest首(pileOfLeaf零) == 一 and 二 + 零 <= dimensionNearest首(pileOfLeaf零):
                indexDomain0: int = pilesTotal // 2 + 1
                indexDomain0 -= pilesFewerDomain0
                listIndicesPilesExcluded.extend([indexDomain0])
                indexDomain0: int = pilesTotal // 4 + 1
                indexDomain0 -= pilesFewerDomain0
                listIndicesPilesExcluded.extend([indexDomain0])
                if pileOfLeaf零 < 首零一(dimensionsTotal):
                    listIndicesPilesExcluded.extend([indexDomain0 - 2])
            if howManyDimensionsHaveOddParity(pileOfLeaf零) == 一:
                indexDomain0 = pilesTotal // 4 + 3
                indexDomain0 -= pilesFewerDomain0
                if dimensionSecondNearest首(pileOfLeaf零) == 一:
                    listIndicesPilesExcluded.extend([indexDomain0])
                if dimensionSecondNearest首(pileOfLeaf零) == 二:
                    listIndicesPilesExcluded.extend([indexDomain0])
                if (dimensionNearest首(pileOfLeaf零) == dimensionsTotal - 1 and dimensionSecondNearest首(pileOfLeaf零) == dimensionsTotal - 3) or dimensionSecondNearest首(pileOfLeaf零) == 二:
                    listIndicesPilesExcluded.extend([indexDomain0 - 2])
                    indexDomain0 = pilesTotal // 2 - 1
                    indexDomain0 -= pilesFewerDomain0
                    listIndicesPilesExcluded.extend([indexDomain0])
        elif 首零一(dimensionsTotal) - direction(首三(dimensionsTotal), isOdd吗(pileOfLeaf零)) == pileOfLeaf零:
            indexDomain0 = pilesTotal // 4 + 2
            indexDomain0 -= pilesFewerDomain0
            listIndicesPilesExcluded.extend([indexDomain0])
        domainCombined.extend([(pileOfLeaf零, pileOfLeaf0) for pileOfLeaf0 in exclude(domain0, listIndicesPilesExcluded)])
    return tuple(sorted(set(domainCombined)))

def getDomain首零一二and首一二(state: EliminationState) -> tuple[tuple[int, int], ...]:
    domain首零一二: tuple[int, ...] = tuple(getLeafDomain(state, 首零一二(state.dimensionsTotal)))
    domain首一二: tuple[int, ...] = tuple(getLeafDomain(state, 首一二(state.dimensionsTotal)))
    direction: CallableFunction[[int, int], int] = add
    return _getDomain首零一二and首一二(domain首零一二, domain首一二, direction, state.dimensionsTotal)

@cache
def _getDomain首零一二and首一二(domain零: tuple[int, ...], domain0: tuple[int, ...], direction: CallableFunction[[int, int], int], dimensionsTotal: int) -> tuple[tuple[int, int], ...]:
    domainCombined: list[tuple[int, int]] = []
    pilesTotal: int = len(domain零)
    pilesFewerDomain0: int = pilesTotal - len(domain0)
    for indexDomain零, pileOfLeaf零 in enumerate(domain零):
        if pileOfLeaf零 < 首零(dimensionsTotal):
            continue
        indicesDomain0ToExclude: list[int] = []
        dimensionTail: int = dimensionNearestTail(direction(pileOfLeaf零, isOdd吗(pileOfLeaf零)))
        if 首零一(dimensionsTotal) < pileOfLeaf零:
            excludeBelow: int = indexDomain零 + 1 - 3 * pilesTotal // 4
        else:
            excludeBelow = (首零一(dimensionsTotal) - direction(pileOfLeaf零, isOdd吗(pileOfLeaf零))) // 2
        excludeBelow -= pilesFewerDomain0
        indicesDomain0ToExclude.extend(range(excludeBelow))
        excludeAbove: int = indexDomain零 + 1 - int(bit_mask(dimensionTail))
        excludeAbove -= pilesFewerDomain0
        indicesDomain0ToExclude.extend(range(excludeAbove, pilesTotal))
        steppingBasisForUnknownReasons: int = indexDomain零
        for dimension in range(dimensionTail):
            indicesDomain0ToExclude.extend(range(steppingBasisForUnknownReasons - int(bit_mask(dimension)), decreasing, decreasing * int(bit_flip(0, dimension + 1))))
        if dimensionTail == 1:
            if dimensionThirdNearest首(pileOfLeaf零) == 一 and 二 + 零 <= dimensionNearest首(pileOfLeaf零):
                indexDomain0: int = pilesTotal // 2
                indexDomain0 -= pilesFewerDomain0
                indicesDomain0ToExclude.extend([indexDomain0])
                indexDomain0: int = pilesTotal // 4
                indexDomain0 -= pilesFewerDomain0
                indicesDomain0ToExclude.extend([indexDomain0])
                if pileOfLeaf零 < 首零一(dimensionsTotal):
                    indicesDomain0ToExclude.extend([indexDomain0 - 2])
            if dimensionThirdNearest首(pileOfLeaf零) == 一 + 零:
                indexDomain0 = pilesTotal // 4
                indexDomain0 -= pilesFewerDomain0
                if dimensionFourthNearest首(pileOfLeaf零) == 一:
                    indicesDomain0ToExclude.extend([indexDomain0])
            if howManyDimensionsHaveOddParity(pileOfLeaf零) == 一:
                indexDomain0 = pilesTotal // 4 + 2
                indexDomain0 -= pilesFewerDomain0
                if dimensionSecondNearest首(pileOfLeaf零) == 一:
                    indexDomain0 = domain0.index(首零(dimensionsTotal) - 一)
                    indicesDomain0ToExclude.extend([indexDomain0])
                if dimensionSecondNearest首(pileOfLeaf零) == 二:
                    indicesDomain0ToExclude.extend([indexDomain0])
                if 首零二(dimensionsTotal) < pileOfLeaf零 and 二 + 零 <= dimensionNearest首(pileOfLeaf零):
                    indicesDomain0ToExclude.extend([indexDomain0 - 2])
                    indexDomain0 = pilesTotal // 2 - 2
                    indexDomain0 -= pilesFewerDomain0
                    indicesDomain0ToExclude.extend([indexDomain0])
        elif 首零一(dimensionsTotal) - direction(首三(dimensionsTotal), isOdd吗(pileOfLeaf零)) == pileOfLeaf零:
            indexDomain0 = pilesTotal // 4 + 1
            indexDomain0 -= pilesFewerDomain0
            indicesDomain0ToExclude.extend([indexDomain0])
        domainCombined.extend([(pileOfLeaf零, pileOfLeaf0) for pileOfLeaf0 in exclude(domain0, indicesDomain0ToExclude)])
    domainCombined.extend([(pile, direction(pile, 零)) for pile in domain零 if direction(pile, 零) in domain0])
    return tuple(sorted(set(domainCombined)))

def getLeaf首零Plus零Domain(state: EliminationState, leaf: Leaf | None = None) -> tuple[Pile, ...]:
    if leaf is None:
        leaf = 零 + 首零(state.dimensionsTotal)
    domain首零Plus零: tuple[Pile, ...] = tuple(getLeafDomain(state, leaf))
    leaf一零: Leaf = 一 + 零
    leaf首零一: Leaf = 首零一(state.dimensionsTotal)
    if state.permutationSpace.leafPinned吗(leaf一零) and state.permutationSpace.leafPinned吗(leaf首零一):
        pileOfLeaf一零: Pile = raiseIfNone(reverseLookup(state.permutationSpace, leaf一零))
        pileOfLeaf首零一: Pile = raiseIfNone(reverseLookup(state.permutationSpace, leaf首零一))
        domain首零Plus零 = _getLeaf首零Plus零Domain(domain首零Plus零, pileOfLeaf一零, pileOfLeaf首零一, state.dimensionsTotal, state.leavesTotal)
    return domain首零Plus零

@cache
def _getLeaf首零Plus零Domain(domain首零Plus零: tuple[Pile, ...], pileOfLeaf一零: Pile, pileOfLeaf首零一: Pile, dimensionsTotal: int, leavesTotal: int) -> tuple[Pile, ...]:
    pilesTotal: int = 首一(dimensionsTotal)
    bump: int = 1 - int(pileOfLeaf一零.bit_count() == 1)
    howMany: int = dimensionsTotal - (pileOfLeaf一零.bit_length() + bump)
    onesInBinary: int = int(bit_mask(howMany))
    ImaPattern: int = pilesTotal - onesInBinary
    listIndicesPilesExcluded: list[int] = []
    if pileOfLeaf一零 == 二:
        listIndicesPilesExcluded.extend([零, 一, 二])
    if 二 < pileOfLeaf一零 <= 首二(dimensionsTotal):
        stop: int = pilesTotal // 2 - 1
        listIndicesPilesExcluded.extend(range(1, stop))
        aDimensionPropertyNotFullyUnderstood: int = 5
        for _dimension in loops(dimensionsTotal - aDimensionPropertyNotFullyUnderstood):
            start: int = 1 + stop
            stop += (stop + 1) // 2
            listIndicesPilesExcluded.extend([*range(start, stop)])
        listIndicesPilesExcluded.extend([*range(1 + stop, ImaPattern)])
    if 首二(dimensionsTotal) < pileOfLeaf一零:
        listIndicesPilesExcluded.extend([*range(1, ImaPattern)])
    bump = 1 - int((leavesTotal - pileOfLeaf首零一).bit_count() == 1)
    howMany = dimensionsTotal - ((leavesTotal - pileOfLeaf首零一).bit_length() + bump)
    onesInBinary = int(bit_mask(howMany))
    ImaPattern = pilesTotal - onesInBinary
    aDimensionPropertyNotFullyUnderstood = 5
    if pileOfLeaf首零一 == leavesTotal - 二:
        listIndicesPilesExcluded.extend([-零 - 1, -一 - 1])
        if aDimensionPropertyNotFullyUnderstood <= dimensionsTotal:
            listIndicesPilesExcluded.extend([-二 - 1])
    if 首零一二(dimensionsTotal) < pileOfLeaf首零一 < leavesTotal - 二 and 首二(dimensionsTotal) < pileOfLeaf一零 <= 首零(dimensionsTotal):
        listIndicesPilesExcluded.extend([-1])
    if 首零一二(dimensionsTotal) <= pileOfLeaf首零一 < leavesTotal - 二:
        stop: int = pilesTotal // 2 - 1
        listIndicesPilesExcluded.extend(range((1 + inclusive) * decreasing, (stop + inclusive) * decreasing, decreasing))
        for _dimension in loops(dimensionsTotal - aDimensionPropertyNotFullyUnderstood):
            start: int = 1 + stop
            stop += (stop + 1) // 2
            listIndicesPilesExcluded.extend([*range((start + inclusive) * decreasing, (stop + inclusive) * decreasing, decreasing)])
        listIndicesPilesExcluded.extend([*range((1 + stop + inclusive) * decreasing, (ImaPattern + inclusive) * decreasing, decreasing)])
        if 二 <= pileOfLeaf一零 <= 首零(dimensionsTotal):
            listIndicesPilesExcluded.extend([零, 一, 二, pilesTotal // 2])
    if pileOfLeaf首零一 == 首零一二(dimensionsTotal) and 首一(dimensionsTotal) < pileOfLeaf一零 <= 首零(dimensionsTotal):
        listIndicesPilesExcluded.extend([-1])
    if 首零一(dimensionsTotal) < pileOfLeaf首零一 < 首零一二(dimensionsTotal):
        if pileOfLeaf一零 in {首一(dimensionsTotal), 首零(dimensionsTotal)}:
            listIndicesPilesExcluded.extend([-1])
        elif 二 < pileOfLeaf一零 < 首二(dimensionsTotal):
            listIndicesPilesExcluded.extend([0])
    if pileOfLeaf首零一 < 首零一二(dimensionsTotal):
        listIndicesPilesExcluded.extend([*range((1 + inclusive) * decreasing, (ImaPattern + inclusive) * decreasing, decreasing)])
    pileOfLeaf一零ARCHETYPICAL: int = 首一(dimensionsTotal)
    bump = 1 - int(pileOfLeaf一零ARCHETYPICAL.bit_count() == 1)
    howMany = dimensionsTotal - (pileOfLeaf一零ARCHETYPICAL.bit_length() + bump)
    onesInBinary = int(bit_mask(howMany))
    ImaPattern = pilesTotal - onesInBinary
    if pileOfLeaf首零一 == leavesTotal - 二:
        if pileOfLeaf一零 == 二:
            listIndicesPilesExcluded.extend([零, 一, 二, pilesTotal // 2 - 1, pilesTotal // 2])
        if 二 < pileOfLeaf一零 <= 首零(dimensionsTotal):
            IDK: int = ImaPattern - 1
            listIndicesPilesExcluded.extend([*range(1, 3 * pilesTotal // 4), *range(1 + 3 * pilesTotal // 4, IDK)])
        if 首一(dimensionsTotal) < pileOfLeaf一零 <= 首零(dimensionsTotal):
            listIndicesPilesExcluded.extend([-1])
    if pileOfLeaf首零一 == 首零一(dimensionsTotal):
        if pileOfLeaf一零 == 首零(dimensionsTotal):
            listIndicesPilesExcluded.extend([-1])
        elif 二 < pileOfLeaf一零 < 首二(dimensionsTotal) or 首二(dimensionsTotal) < pileOfLeaf一零 < 首一(dimensionsTotal):
            listIndicesPilesExcluded.extend([0])
    return tuple(exclude(domain首零Plus零, listIndicesPilesExcluded))

def getDictionaryLeafDomains(state: EliminationState) -> dict[int, range]:
    return {leaf: getLeafDomain(state, leaf) for leaf in range(state.leavesTotal)}

def getDictionaryConditionalLeafPredecessors(state: EliminationState) -> dict[Leaf, dict[Pile, list[Leaf]]]:
    dictionaryConditionalLeafPredecessors: dict[Leaf, dict[Pile, list[Leaf]]] = {}
    if mapShapeIs2上nDimensions(state.mapShape, youMustBeDimensionsTallToPinThis=6):
        dictionaryConditionalLeafPredecessors = _getDictionaryConditionalLeafPredecessors(state.mapShape)
    return dictionaryConditionalLeafPredecessors

@cache
def _getDictionaryConditionalLeafPredecessors(mapShape: tuple[int, ...]) -> dict[Leaf, dict[Pile, list[Leaf]]]:
    state = EliminationState(mapShape)
    dictionaryDomains: dict[Leaf, range] = getDictionaryLeafDomains(state)
    dictionaryPrecedence: dict[Leaf, dict[Pile, list[Leaf]]] = {}
    for dimension in range(3, state.dimensionsTotal + inclusive):
        for countDown in range(dimension - 2 + decreasing, decreasing, decreasing):
            for leaf in range(state.productsOfDimensions[dimension] - sum(state.productsOfDimensions[countDown:dimension - 2]), state.leavesTotal, state.productsOfDimensions[dimension - 1]):
                dictionaryPrecedence[leaf] = {aPile: [state.productsOfDimensions[dimensionNearest首(leaf)] + state.productsOfDimensions[dimensionNearestTail(leaf)]] for aPile in list(dictionaryDomains[leaf])[0:getSumsOfProductsOfDimensionsNearest首(state.productsOfDimensions, dimensionFrom首=dimension - 1)[dimension - 2 - countDown] // 2]}
    leaf = 零 + 首一(state.dimensionsTotal)
    dictionaryPrecedence[leaf] = {aPile: [2 * state.productsOfDimensions[dimensionNearest首(leaf)] + state.productsOfDimensions[dimensionNearestTail(leaf)], 3 * state.productsOfDimensions[dimensionNearest首(leaf)] + state.productsOfDimensions[dimensionNearestTail(leaf)]] for aPile in list(dictionaryDomains[leaf])[1:2]}
    del leaf
    leaf: Leaf = 零 + 首零一(state.dimensionsTotal)
    listOfPiles = list(dictionaryDomains[leaf])
    dictionaryPrecedence[leaf] = {aPile: [] for aPile in list(dictionaryDomains[leaf])}
    sumsOfProductsOfDimensionsNearest首: tuple[int, ...] = getSumsOfProductsOfDimensionsNearest首(state.productsOfDimensions)
    sumsOfProductsOfDimensionsNearest首InSubHyperplane: tuple[int, ...] = getSumsOfProductsOfDimensionsNearest首(state.productsOfDimensions, dimensionFrom首=state.dimensionsTotal - 1)
    pileStepAbsolute = 2
    for aPile in listOfPiles[listOfPiles.index(一 + 零):listOfPiles.index(neg(零) + 首零(state.dimensionsTotal)) + inclusive]:
        dictionaryPrecedence[leaf][aPile].append(零 + 首零(state.dimensionsTotal))
    for indexUniversal in range(state.dimensionsTotal - 2):
        leafPredecessorTheFirst: int = state.sumsOfProductsOfDimensions[indexUniversal + 2]
        leavesPredecessorInThisSeries: int = state.productsOfDimensions[howManyDimensionsHaveOddParity(leafPredecessorTheFirst)]
        for addend in range(leavesPredecessorInThisSeries):
            leafPredecessor = leafPredecessorTheFirst + addend * decreasing
            pileFirst: int = sumsOfProductsOfDimensionsNearest首[indexUniversal] + state.sumsOfProductsOfDimensions[2] + state.productsOfDimensions[state.dimensionsTotal - (indexUniversal + 2)] - pileStepAbsolute * 2 * (howManyDimensionsHaveOddParity(leafPredecessor) - 1 + isEven吗(leafPredecessor)) * (1 + (2 == howManyDimensionsHaveOddParity(leafPredecessor) + isEven吗(leafPredecessor) == dimensionNearest首(leafPredecessor)))
            for aPile in listOfPiles[listOfPiles.index(pileFirst):None]:
                dictionaryPrecedence[leaf][aPile].append(leafPredecessor)
            leafPredecessor首零: int = leafPredecessor + 首零(state.dimensionsTotal)
            if leafInSubHyperplane(leafPredecessor) == 0 and isOdd吗(dimensionNearestTail(leafPredecessor)):
                dictionaryPrecedence[leaf][pileFirst].append(leafPredecessor首零)
            if leafPredecessor首零 == leaf:
                continue
            pileFirst = listOfPiles[-1] - pileStepAbsolute * (howManyDimensionsHaveOddParity(leafPredecessor首零) - 1 + isEven吗(leafPredecessor首零) - isOdd吗(leafPredecessor首零) - int(dimensionNearestTail(leafPredecessor首零) == state.dimensionsTotal - 2) - int(leaf < leafPredecessor首零))
            for aPile in listOfPiles[listOfPiles.index(pileFirst):None]:
                dictionaryPrecedence[leaf][aPile].append(leafPredecessor首零)
            if indexUniversal < state.dimensionsTotal - 4 and isOdd吗(dimensionNearestTail(leafPredecessor - isOdd吗(leafPredecessor))):
                pileFirst = sumsOfProductsOfDimensionsNearest首InSubHyperplane[indexUniversal] + state.sumsOfProductsOfDimensions[2 + 1 + indexUniversal] - pileStepAbsolute * 2 * (howManyDimensionsHaveOddParity(leafPredecessor首零) - 1 + isEven吗(leafPredecessor首零) * indexUniversal - isEven吗(leafPredecessor首零) * int(not bool(indexUniversal))) + state.productsOfDimensions[state.dimensionsTotal - 1 + addend * int(not bool(indexUniversal)) - (indexUniversal + 2)]
                for aPile in listOfPiles[listOfPiles.index(pileFirst) + indexUniversal:listOfPiles.index(neg(零) + 首零(state.dimensionsTotal)) - indexUniversal + inclusive]:
                    dictionaryPrecedence[leaf][aPile].append(leafPredecessor首零)
    del leaf, listOfPiles, sumsOfProductsOfDimensionsNearest首, pileStepAbsolute, sumsOfProductsOfDimensionsNearest首InSubHyperplane
    leaf: Leaf = 零 + 首零(state.dimensionsTotal)
    listOfPiles: list[Pile] = list(dictionaryDomains[leaf])[1:None]
    dictionaryPrecedence[leaf] = {aPile: [] for aPile in listOfPiles}
    sumsOfProductsOfDimensionsNearest首: tuple[int, ...] = getSumsOfProductsOfDimensionsNearest首(state.productsOfDimensions)
    pileStepAbsolute = 4
    for indexUniversal in range(state.dimensionsTotal - 2):
        leafPredecessorTheFirst: int = state.sumsOfProductsOfDimensions[indexUniversal + 2]
        leavesPredecessorInThisSeries = state.productsOfDimensions[howManyDimensionsHaveOddParity(leafPredecessorTheFirst)]
        for addend in range(leavesPredecessorInThisSeries):
            leafPredecessor: int = leafPredecessorTheFirst + addend * decreasing
            leafPredecessor首零: int = leafPredecessor + 首零(state.dimensionsTotal)
            pileFirst = sumsOfProductsOfDimensionsNearest首[indexUniversal] + 6 - pileStepAbsolute * (howManyDimensionsHaveOddParity(leafPredecessor) - 1 + isEven吗(leafPredecessor))
            for aPile in listOfPiles[listOfPiles.index(pileFirst):None]:
                dictionaryPrecedence[leaf][aPile].append(leafPredecessor)
                dictionaryPrecedence[leaf][aPile].append(leafPredecessor首零)
    del leaf, listOfPiles, sumsOfProductsOfDimensionsNearest首, pileStepAbsolute
    if state.dimensionsTotal == 6:
        leaf = 22
        sliceOfPiles = slice(0, None)
        listOfPiles = list(dictionaryDomains[leaf])[sliceOfPiles]
        leafPredecessorPileFirstPileLast = [(15, 43, 43)]
        for leafPredecessor, pileFirst, pileLast in leafPredecessorPileFirstPileLast:
            for pile in listOfPiles[listOfPiles.index(pileFirst):listOfPiles.index(pileLast) + inclusive]:
                dictionaryPrecedence[leaf].setdefault(pile, []).append(leafPredecessor)
    return dictionaryPrecedence

def getDictionaryConditionalLeafSuccessors(state: EliminationState) -> dict[Leaf, dict[Pile, list[Leaf]]]:
    return _getDictionaryConditionalLeafSuccessors(state.mapShape)

@cache
def _getDictionaryConditionalLeafSuccessors(mapShape: tuple[int, ...]) -> dict[Leaf, dict[Pile, list[Leaf]]]:
    state = EliminationState(mapShape)
    dictionaryDomains: dict[Leaf, range] = getDictionaryLeafDomains(state)
    dictionarySuccessor: dict[Leaf, dict[Pile, list[Leaf]]] = {}
    dictionaryPrecedence: dict[Leaf, dict[Pile, list[Leaf]]] = getDictionaryConditionalLeafPredecessors(state)
    for leafLater, dictionaryPiles in dictionaryPrecedence.items():
        tupleDomainLater: tuple[Pile, ...] = tuple(dictionaryDomains[leafLater])
        dictionaryPilesByPredecessor: defaultdict[Leaf, set[Pile]] = defaultdict(set)
        for pileLater, listLeafPredecessors in dictionaryPiles.items():
            for leafEarlier in listLeafPredecessors:
                dictionaryPilesByPredecessor[leafEarlier].add(pileLater)
        for leafEarlier, setPilesRequiring in dictionaryPilesByPredecessor.items():
            tupleDomainEarlier: tuple[Pile, ...] = tuple(dictionaryDomains[leafEarlier])
            listOptionalPiles: list[Pile] = sorted(pile for pile in tupleDomainLater if pile not in setPilesRequiring)
            for pileEarlier in tupleDomainEarlier:
                optionalLessEqualCount: int = bisect_right(listOptionalPiles, pileEarlier)
                if optionalLessEqualCount == 0:
                    listSuccessors: list[Leaf] = dictionarySuccessor.setdefault(leafEarlier, {}).setdefault(pileEarlier, [])
                    if leafLater not in listSuccessors:
                        listSuccessors.append(leafLater)
    return dictionarySuccessor

@syntacticCurry
def filterCeiling(pile: Pile, dimensionsTotal: int, leaf: Leaf) -> bool:
    return pile < int(bit_mask(dimensionsTotal) ^ bit_mask(dimensionsTotal - dimensionNearest首(leaf))) - howManyDimensionsHaveOddParity(leaf) + 2 - (leaf == leafOrigin)

@syntacticCurry
def filterFloor(pile: Pile, leaf: Leaf) -> bool:
    return int(bit_flip(0, dimensionNearestTail(leaf) + 1)) + howManyDimensionsHaveOddParity(leaf) - 1 - (leaf == leafOrigin) <= pile

@syntacticCurry
def filterParity(pile: Pile, leaf: Leaf) -> bool:
    return pile & 1 == int(bit_flip(0, dimensionNearestTail(leaf) + 1)) + howManyDimensionsHaveOddParity(leaf) - 1 - (leaf == leafOrigin) & 1

@syntacticCurry
def filterDoubleParity(pile: Pile, dimensionsTotal: int, leaf: Leaf) -> bool:
    if leaf != 首零(dimensionsTotal) + 零:
        return True
    return pile >> 1 & 1 == int(bit_flip(0, dimensionNearestTail(leaf) + 1)) + howManyDimensionsHaveOddParity(leaf) - 1 - (leaf == leafOrigin) >> 1 & 1

@cache
def _getLeafOptions(pile: Pile, dimensionsTotal: int, mapShape: tuple[int, ...], leavesTotal: int) -> LeafOptions:
    leafOptions: Iterable[Leaf] = range(leavesTotal)
    if mapShapeIs2上nDimensions(mapShape):
        parityMatch: Callable[[Leaf], bool] = filterParity(pile)
        pileAboveFloor: Callable[[Leaf], bool] = filterFloor(pile)
        pileBelowCeiling: Callable[[Leaf], bool] = filterCeiling(pile, dimensionsTotal)
        matchLargerStep: Callable[[Leaf], bool] = filterDoubleParity(pile, dimensionsTotal)
        leafOptions = filter(parityMatch, leafOptions)
        leafOptions = filter(pileAboveFloor, leafOptions)
        leafOptions = filter(pileBelowCeiling, leafOptions)
        leafOptions = filter(matchLargerStep, leafOptions)
    return makeLeafOptions(leavesTotal, leafOptions)

def notLeafOriginOrLeaf零(leaf: LeafSpace) -> bool:
    return 零 < leaf

@syntacticCurry
def oddLeaf2上nDimensional吗(dimension: DimensionIndex, leaf: Leaf) -> bool:
    return isBit1吗(leaf, dimension)

def creaseViolation吗(pile: Pile, pileComparand: Pile, pileCrease: Pile, pileComparandCrease: Pile) -> bool:
    if pile < pileComparand:
        if pileComparandCrease < pile:
            if pileCrease < pileComparandCrease:
                return True
            return pileComparand < pileCrease
        if pileComparand < pileCrease:
            return pileCrease < pileComparandCrease
        else:
            return pile < pileComparandCrease < pileCrease < pileComparand
    return False

def foldingValid吗(folding: Folding, mapShape: tuple[int, ...]) -> bool:
    leavesPinned: PinnedLeaves = dict(enumerate(folding))
    leafToPile: dict[Leaf, Pile] = {leafValue: pileKey for pileKey, leafValue in DOTitems(leavesPinned)}
    for dimension in range(_dimensionsTotal(mapShape)):
        listPilePileCreaseByParity: list[deque[tuple[Pile, Pile]]] = [deque(), deque()]
        for pile, leaf in leavesPinned.items():
            crease: int | None = getCreasePost(mapShape, leaf, dimension)
            if crease:
                listPilePileCreaseByParity[oddLeaf吗(mapShape, leaf, dimension)].append((pile, leafToPile[crease]))
        for groupedParity in listPilePileCreaseByParity:
            if any((creaseViolation吗(pile, pileComparand, pileCrease, pileComparandCrease) for (pile, pileCrease), (pileComparand, pileComparandCrease) in combinations(sorted(groupedParity), 2))):
                return False
    return True

def leavesPinnedValid吗(leavesPinned: PinnedLeaves, mapShape: tuple[int, ...]) -> bool:
    leafToPile: dict[Leaf, Pile] = {leafValue: pileKey for pileKey, leafValue in DOTitems(leavesPinned)}
    for dimension in range(_dimensionsTotal(mapShape)):
        listPilePileCreaseByParity: list[deque[tuple[Pile, Pile]]] = [deque(), deque()]
        for pile, leaf in leavesPinned.items():
            crease: int | None = getCreasePost(mapShape, leaf, dimension)
            if crease:
                listPilePileCreaseByParity[oddLeaf吗(mapShape, leaf, dimension)].append((pile, leafToPile[crease]))
        for groupedParity in listPilePileCreaseByParity:
            if any((creaseViolation吗(pile, pileComparand, pileCrease, pileComparandCrease) for (pile, pileCrease), (pileComparand, pileComparandCrease) in combinations(sorted(groupedParity), 2))):
                return False
    return True

@cache
def _dimensionsTotal(mapShape: tuple[int, ...]) -> int:
    return len(mapShape)

@cache
def _leavesTotal(mapShape: tuple[int, ...]) -> int:
    return getLeavesTotal(mapShape)

@cache
def getCreasePost(mapShape: tuple[int, ...], leaf: Leaf, dimension: int) -> Leaf | None:
    leafCrease: Leaf | None = None
    if leaf // productOfDimensions(mapShape, dimension) % mapShape[dimension] + 1 < mapShape[dimension]:
        leafCrease = leaf + productOfDimensions(mapShape, dimension)
    return leafCrease

@cache
def oddLeaf吗(mapShape: tuple[int, ...], leaf: Leaf, dimension: int) -> int:
    return leaf // productOfDimensions(mapShape, dimension) % mapShape[dimension] & 1

@cache
def productOfDimensions(mapShape: tuple[int, ...], dimension: int) -> int:
    return prod(mapShape[0:dimension], start=1)

def pinByCrease(state: EliminationState) -> EliminationState:
    listFolding: deque[Folding] = deque()
    while state.listPermutationSpace:
        permutationSpace: PermutationSpace = state.listPermutationSpace.pop()
        sherpa: EliminationState = EliminationState(state.mapShape, permutationSpace=permutationSpace)
        sherpa.listPermutationSpace.extend(sherpa.permutationSpace.deconstructAtPile())
        sherpa = sherpa.reduceAllPermutationSpace(listFunctionsReduction2上nDimensional).removeCreaseViolations().moveToListFolding()
        listFolding.extend(sherpa.listFolding)
        state.listPermutationSpace.extend(sherpa.listPermutationSpace)
    state.listFolding.extend(listFolding)
    return state

def doTheNeedful(state: EliminationState, workersMaximum: int) -> EliminationState:
    if not mapShapeIs2上nDimensions(state.mapShape):
        return state
    if not state.listPermutationSpace:
        state = pinPilesAtEnds(state, 1)
    with ProcessPoolExecutor(workersMaximum) as concurrencyManager:
        listPermutationSpace: deque[PermutationSpace] = state.listPermutationSpace.copy()
        state.listPermutationSpace = deque()
        listClaimTickets: list[Future[EliminationState]] = [concurrencyManager.submit(pinByCrease, EliminationState(state.mapShape, listPermutationSpace=deque([permutationSpace]))) for permutationSpace in listPermutationSpace]
        for claimTicket in tqdm(as_completed(listClaimTickets), total=len(listClaimTickets), disable=False):
            state.listFolding.extend(claimTicket.result().listFolding)
    state.Theorem4Multiplier = factorial(state.dimensionsTotal)
    state.groupsOfFolds = len(state.listFolding)
    return state
if __name__ == '__main__':
    CPUlimit: int | float | None = None
    state: EliminationState = EliminationState((2,) * 5)
    state = pinPilesAtEnds(state, 3)
    state = pinLeavesDimension首二(state)
    state = pinLeavesDimensions0零一(state)
    workersMaximum: int = defineProcessorLimit(CPUlimit)
    print(doTheNeedful(state, workersMaximum).foldsTotal)
