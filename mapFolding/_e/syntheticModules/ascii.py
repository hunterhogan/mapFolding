from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict, deque
from concurrent.futures import as_completed, ProcessPoolExecutor
from functools import cache, partial, reduce
from gmpy2 import (
	bit_clear, bit_flip, bit_mask, bit_scan1, bit_set, bit_test as isBit1Ma, f_mod_2exp, is_even as isEvenMa, is_odd as isOddMa, mpz, xmpz)
from humpy_cytoolz import (
	assoc as associateKeyValue, compose, concat, curry as syntacticCurry, dissoc as dissociatePile, first, get, groupby as toolz_groupby,
	itemfilter, keyfilter as filterPile, merge, unique, valfilter as filterLeaf, valfilter as filterLeafOptions, valfilter as filterValue)
from hunterMakesPy import decreasing, errorL33T, inclusive, raiseIfNone, zeroIndexed
from hunterMakesPy.parseParameters import defineConcurrencyLimit, intInnit
from itertools import accumulate, chain, combinations, filterfalse, product as CartesianProduct
from math import factorial, log, prod
from more_itertools import all_unique as allUniqueMa, iter_index, last, loops, one, pairwise, partition, triplewise
from operator import add, attrgetter, getitem, itemgetter, methodcaller, mul, neg, sub
from sys import maxsize as sysMaxsize
from tqdm import tqdm
from typing import cast, overload, TYPE_CHECKING, TypeAlias
from Z0Z_tools import betweenMa, consecutiveMa, DOTitems, DOTkeys, DOTvalues, exclude, reverseLookup, thisHasThatMa, thisNotHaveThatMa
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

def defineProcessorLimit(CPUlimit: Limitation, concurrencyPackage: str | None=None) -> int:
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

def leafOptionsLeafNone(leafOptions: LeafOptions) -> LeafOptions | Leaf | None:
    whoAmI: LeafOptions | Leaf | None = leafOptions
    if isLeafOptionsMa(leafOptions):
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

def getSumsOfProductsOfDimensionsNearestShou(productsOfDimensions: tuple[int, ...], dimensionsTotal: int | None=None, dimensionFromShou: int | None=None) -> tuple[int, ...]:
    dimensionsTotal = dimensionsTotal or len(productsOfDimensions) - 1
    if dimensionFromShou is None:
        dimensionFromShou = dimensionsTotal
    productsOfDimensionsTruncator: int = dimensionFromShou - (dimensionsTotal + zeroIndexed)
    productsOfDimensionsFromShou: tuple[int, ...] = productsOfDimensions[0:productsOfDimensionsTruncator][::-1]
    sumsOfProductsOfDimensionsNearestShou: tuple[int, ...] = tuple(accumulate(productsOfDimensionsFromShou, add, initial=0))
    return sumsOfProductsOfDimensionsNearestShou

def indicesMapShapeDimensionLengthsAreEqual(mapShape: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    return filter(lambda indices: 1 < len(indices), map(tuple, map(partial(iter_index, mapShape), unique(filter(1 .__lt__, mapShape)))))

class PermutationSpace(dict[Pile, LeafSpace]):

    def addMissingPileLeafSpace(self, missing: PermutationSpace | UndeterminedPiles | PinnedLeaves) -> PermutationSpace:
        self = PermutationSpace(sorted(DOTitems(merge(missing, self, factory=PermutationSpace))))
        return self.copy()

    def atPilePinLeaf(self, pile: Pile, leaf: Leaf) -> PermutationSpace:
        return PermutationSpace(associateKeyValue(self, pile, leaf, PermutationSpace))

    def atPilePinLeafSafetyFilter(self, pile: Pile, leaf: Leaf) -> bool:
        return self.leafPinnedAtPileMa(leaf, pile) or (self.pileUndeterminedMa(pile) and self.leafNotPinnedMa(leaf))

    def bifurcate(self) -> tuple[PinnedLeaves, UndeterminedPiles]:
        leavesPinned: PinnedLeaves = self.extractPinnedLeaves()
        return (leavesPinned, cast('UndeterminedPiles', dissociatePile(self, *DOTkeys(leavesPinned))))

    def copy(self) -> PermutationSpace:
        return PermutationSpace(self)

    def deconstructAtPile(self, pile: Pile | None=None, leavesToPin: Iterable[Leaf]=()) -> Iterable[PermutationSpace]:
        if pile is None:
            pile = first(filterLeaf(isLeafOptionsMa, self))
        if (leafOptions := self.getLeafOptions(pile)) is None:
            deconstructed: Iterable[PermutationSpace] = deque([self])
        else:
            leavesToPin = leavesToPin or getIteratorOfLeaves(leafOptions)
            deconstructed = map(partial(self.atPilePinLeaf, pile), filter(self.leafNotPinnedMa, leavesToPin))
        return deconstructed

    def deconstructByDomainOfLeaf(self, leaf: Leaf, leafDomain: Iterable[Pile]) -> deque[PermutationSpace]:
        deconstructedPermutationSpace: deque[PermutationSpace] = deque()
        if self.leafNotPinnedMa(leaf):
            leafInPileRange: Callable[[int], bool] = compose(leafInLeafOptionsMa(leaf), partial(self.getLeafOptions, default=bit_mask(len(self))))
            pinLeafAt: Callable[[int], PermutationSpace] = partial(self.atPilePinLeaf, leaf=leaf)
            deconstructedPermutationSpace.extend(map(pinLeafAt, filter(leafInPileRange, filter(self.pileUndeterminedMa, leafDomain))))
        else:
            deconstructedPermutationSpace.append(self)
        return deconstructedPermutationSpace

    def deconstructByDomainsCombined(self, leaves: Sequence[Leaf], leavesDomain: Iterable[Sequence[Pile]]) -> deque[PermutationSpace]:
        deconstructedPermutationSpace: deque[PermutationSpace] = deque()

        def pileOpenByIndex(index: int) -> CallableFunction[[Sequence[Pile]], bool]:

            def workhorse(domain: Sequence[Pile]) -> bool:
                return self.pileUndeterminedMa(domain[index])
            return workhorse

        def leafInPileRangeByIndex(index: int) -> CallableFunction[[Sequence[Pile]], bool]:

            def workhorse(domain: Sequence[Pile]) -> bool:
                leafOptions: LeafOptions = raiseIfNone(self.getLeafOptions(domain[index], default=bit_mask(len(self))))
                return leafInLeafOptionsMa(leaves[index], leafOptions)
            return workhorse

        def isPinnedAtPileByIndex(leaf: Leaf, index: int) -> CallableFunction[[Sequence[Pile]], bool]:

            def workhorse(domain: Sequence[Pile]) -> bool:
                return self.leafPinnedAtPileMa(leaf, domain[index])
            return workhorse
        if any(map(self.leafNotPinnedMa, leaves)):
            for index in range(len(leaves)):
                if self.leafNotPinnedMa(leaves[index]):
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
        return dict(sorted(DOTitems(filterLeaf(isLeafMa, self))))

    def extractUndeterminedPiles(self) -> UndeterminedPiles:
        return dict(sorted(DOTitems(filterLeaf(isLeafOptionsMa, self))))

    @overload
    def getLeaf(self, pile: Pile, default: None=None) -> Leaf | None:
        ...

    @overload
    def getLeaf(self, pile: Pile, default: Leaf) -> Leaf:
        ...

    @overload
    def getLeaf[Ge](self, pile: Pile, default: Ge) -> Leaf | Ge:
        ...

    def getLeaf[Ge](self, pile: Pile, default: Leaf | Ge | None=None) -> Leaf | Ge | None:
        ImaLeaf: LeafSpace | None = self.get(pile)
        if isLeafMa(ImaLeaf):
            return ImaLeaf
        return default

    @overload
    def getLeafOptions(self, pile: Pile, default: None=None) -> LeafOptions | None:
        ...

    @overload
    def getLeafOptions(self, pile: Pile, default: LeafOptions) -> LeafOptions:
        ...

    @overload
    def getLeafOptions[Ge](self, pile: Pile, default: Ge) -> LeafOptions | Ge:
        ...

    def getLeafOptions[Ge](self, pile: Pile, default: LeafOptions | Ge | None=None) -> LeafOptions | Ge | None:
        ImaLeafOptions: LeafSpace | None = self.get(pile)
        if isLeafOptionsMa(ImaLeafOptions):
            return ImaLeafOptions
        return default

    def leafNotPinnedMa(self, leaf: Leaf) -> bool:
        return leaf not in self.values()

    @property
    def leafCount(self) -> int:
        return sum(map(isLeafMa, self.values()))

    def leafPinnedMa(self, leaf: Leaf) -> bool:
        return leaf in self.values()

    def leafPinnedAtPileMa(self, leaf: Leaf, pile: Pile) -> bool:
        return leaf == self.get(pile)

    def makeFolding(self, leavesToInsert: Sequence[Leaf]=()) -> Folding:
        pilesToInsert: Iterator[Pile] = DOTkeys(self.extractUndeterminedPiles())
        return tuple(DOTvalues(dict(sorted(DOTitems(cast('PinnedLeaves', merge(self, dict(zip(pilesToInsert, leavesToInsert, strict=True)), factory=PermutationSpace)))))))

    def pilePinnedMa(self, pile: Pile) -> bool:
        return isLeafMa(self[pile])

    def pileUndeterminedMa(self, pile: Pile) -> bool:
        return not isLeafMa(self[pile])

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
    sumsOfProductsOfDimensionsNearestShou: tuple[int, ...] = dataclasses.field(init=False)
    Shou: int = dataclasses.field(init=False)

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
        self.Shou = self.leavesTotal
        self.productsOfDimensions = getProductsOfDimensions(self.mapShape)
        self.sumsOfProductsOfDimensions = getSumsOfProductsOfDimensions(self.mapShape)
        self.sumsOfProductsOfDimensionsNearestShou = getSumsOfProductsOfDimensionsNearestShou(self.productsOfDimensions, self.dimensionsTotal, self.dimensionsTotal)

    def moveToListFolding(self) -> Self:
        foldingGroupMa: dict[bool, list[PermutationSpace]] = toolz_groupby(compose(self.leavesTotal.__eq__, attrgetter('leafCount')), self.listPermutationSpace)
        self.listPermutationSpace = deque(foldingGroupMa.get(False, ()))
        self.listFolding.extend(map(methodcaller('makeFolding'), foldingGroupMa.get(True, ())))
        return self

    def permutationSpaceCreaseViolationMa(self, permutationSpace: PermutationSpace) -> bool:
        leafToPile: dict[Leaf, Pile] = {leafValue: pileKey for pileKey, leafValue in DOTitems(permutationSpace.extractPinnedLeaves())}
        for dimension in range(self.dimensionsTotal):
            listPileCreaseByParity: list[list[tuple[Pile, Pile]]] = [[], []]
            for pile, leaf in permutationSpace.extractPinnedLeaves().items():
                crease: int | None = getCreasePost(self.mapShape, leaf, dimension)
                if crease:
                    pileCrease: int | None = leafToPile.get(crease)
                    if pileCrease:
                        listPileCreaseByParity[oddLeafMa(self.mapShape, leaf, dimension)].append((pile, pileCrease))
            for groupedParity in listPileCreaseByParity:
                if any((creaseViolationMa(pile, pileComparand, pileCrease, pileComparandCrease) for (pile, pileCrease), (pileComparand, pileComparandCrease) in combinations(sorted(groupedParity), 2))):
                    return True
        return False

    def pinAt_pileMa(self, leaf: Leaf) -> bool:
        return all((self.permutationSpace.leafNotPinnedMa(leaf), self.permutationSpace.pileUndeterminedMa(self.pile), self.pile in getLeafDomain(self, leaf)))

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
        self.listPermutationSpace.extend(filterfalse(self.permutationSpaceCreaseViolationMa, listPermutationSpace))
        return self

@syntacticCurry
def leafInLeafOptionsMa(leaf: Leaf, leafOptions: LeafOptions) -> bool:
    return leafOptions.bit_test(leaf)

@syntacticCurry
def leafPinnedMa(leavesPinned: PinnedLeaves, leaf: Leaf) -> bool:
    return leaf in leavesPinned.values()

@syntacticCurry
def notPileLast(pileLast: Pile, pile: Pile) -> bool:
    return pileLast != pile

def isLeafMa(leafSpace: LeafSpace | None) -> TypeIs[Leaf]:
    return isinstance(leafSpace, Leaf)

def isLeafOptionsMa(leafSpace: LeafSpace | None) -> TypeIs[LeafOptions]:
    return isinstance(leafSpace, LeafOptions)

def segregateLeafPinnedAtPile(listPermutationSpace: Iterable[PermutationSpace], leaf: Leaf, pile: Pile) -> tuple[list[PermutationSpace], list[PermutationSpace]]:
    isPinned: Callable[[PermutationSpace], bool] = partial(PermutationSpace.leafPinnedAtPileMa, leaf=leaf, pile=pile)
    grouped: dict[bool, list[PermutationSpace]] = toolz_groupby(isPinned, listPermutationSpace)
    return (grouped.get(False, []), grouped.get(True, []))

def excludeLeaf_rBeforeLeaf_k(state: EliminationState, leaf_k: Leaf, leaf_r: Leaf, domain_k: Iterable[Pile] | None=None, domain_r: Iterable[Pile] | None=None) -> EliminationState:
    if domain_k is None:
        domain_k = getLeafDomain(state, leaf_k)
    for pile_k in reversed(tuple(domain_k)):
        state = excludeLeaf_rBeforeLeaf_kAtPile_k(state, leaf_k, leaf_r, pile_k, domainOf_leaf_r=domain_r)
    return state

def excludeLeaf_rBeforeLeaf_kAtPile_k(state: EliminationState, leaf_k: Leaf, leaf_r: Leaf, pile_k: Pile, domainOf_leaf_r: Iterable[Pile] | None=None) -> EliminationState:
    listPermutationSpace: deque[PermutationSpace] = state.listPermutationSpace
    state.listPermutationSpace = deque()
    listPermutationSpaceUnchanged: deque[PermutationSpace] = deque()
    listExcludeLeaf_r: Iterable[PermutationSpace] = []
    for permutationSpace in listPermutationSpace:
        if permutationSpace.leafPinnedAtPileMa(leaf_k, pile_k):
            listExcludeLeaf_r.append(permutationSpace)
        elif leafInLeafOptionsMa(leaf_k, permutationSpace.getLeafOptions(pile_k, LeafOptions(0))):
            permutationSpaceCopy = permutationSpace.copy()
            permutationSpaceCopy[pile_k] = bit_clear(permutationSpaceCopy[pile_k], leaf_k)
            state.listPermutationSpace.append(permutationSpaceCopy)
            listExcludeLeaf_r.append(permutationSpace.atPilePinLeaf(pile_k, leaf_k))
        else:
            listPermutationSpaceUnchanged.append(permutationSpace)
    if domainOf_leaf_r is None:
        domainOf_leaf_r = getLeafDomain(state, leaf_r)
    for pile_r in filter(betweenMa(0, pile_k - inclusive), domainOf_leaf_r):
        listExcludeLeaf_r = excludeLeafAtPile(listExcludeLeaf_r, leaf_r, pile_r)
    state.listPermutationSpace.extend(listExcludeLeaf_r)
    state.reduceAllPermutationSpace(listFunctionsReduction).removeCreaseViolations()
    state.listPermutationSpace.extend(listPermutationSpaceUnchanged)
    return state

def excludeLeafAtPile(listPermutationSpace: Iterable[PermutationSpace], leaf: Leaf, pile: Pile) -> Iterator[PermutationSpace]:
    listPermutationSpace, _pinnedAtPile = segregateLeafPinnedAtPile(listPermutationSpace, leaf, pile)
    pilePinned: dict[bool, list[PermutationSpace]] = toolz_groupby(methodcaller('pilePinnedMa', pile), listPermutationSpace)
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
        for pile, leafOptions in DOTitems(filterLeafOptions(thisNotHaveThatMa(unique(pilesUndetermined.values())), pilesUndetermined)):
            groupByLeafOptions.setdefault(leafOptions, set()).add(pile)
        for leafOptions, setPiles in DOTitems(itemfilter(lambda groupBy: howManyLeavesInLeafOptions(groupBy[leafOptionsKey]) == len(groupBy[piles]), groupByLeafOptions)):
            if not (permutationSpace := reduceLeafSpace(permutationSpace, DOTitems(filterPile(thisNotHaveThatMa(setPiles), pilesUndetermined)), makeLeafAntiOptions(state.leavesTotal, getIteratorOfLeaves(leafOptions)))):
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
            sherpa: PermutationSpace | None = reducePermutationSpace_LeafIsPinned(state, permutationSpace.atPilePinLeaf(one(DOTkeys(filterLeaf(leafInLeafOptionsMa(leaf), pilesUndetermined))), leaf))
            if sherpa is None or not sherpa:
                return None
            else:
                permutationSpace = sherpa
            permutationSpaceHasNewLeaf = True
    return permutationSpace
listFunctionsReduction: Sequence[Callable[[EliminationState, PermutationSpace], PermutationSpace | None]] = (reducePermutationSpace_LeafIsPinned, reducePermutationSpace_leafDomainOf1, reducePermutationSpace_nakedSubset)

def _pinPiles(state: EliminationState, maximumSizeListPermutationSpace: int, pileProcessingOrder: deque[Pile], CPUlimit: Limitation=None) -> EliminationState:
    workersMaximum: int = defineProcessorLimit(CPUlimit)
    while pileProcessingOrder and len(state.listPermutationSpace) < maximumSizeListPermutationSpace:
        pile: Pile = pileProcessingOrder.popleft()
        thesePilesAreOpen: tuple[Iterator[PermutationSpace], Iterator[PermutationSpace]] = partition(partial(PermutationSpace.pileUndeterminedMa, pile=pile), state.listPermutationSpace)
        state.listPermutationSpace = deque(thesePilesAreOpen[False])
        with ProcessPoolExecutor(workersMaximum) as concurrencyManager:
            listClaimTickets: list[Future[EliminationState]] = [concurrencyManager.submit(_pinPilesConcurrentTask, EliminationState(mapShape=state.mapShape, permutationSpace=permutationSpace, pile=pile)) for permutationSpace in thesePilesAreOpen[True]]
            for claimTicket in tqdm(as_completed(listClaimTickets), total=len(listClaimTickets), desc=f'Pinning pile {pile:3d} of {state.pileLast:3d}', disable=False):
                state.listPermutationSpace.extend(claimTicket.result().listPermutationSpace)
                state.listFolding.extend(claimTicket.result().listFolding)
    return state

def _pinPilesConcurrentTask(state: EliminationState) -> EliminationState:
    state.listPermutationSpace.extend(state.permutationSpace.deconstructAtPile(state.pile, filter(state.pinAt_pileMa, _getLeavesAtPile(state))))
    return state.reduceAllPermutationSpace(listFunctionsReduction2ShangnDimensional).removeCreaseViolations().moveToListFolding()

def _getLeavesAtPile(state: EliminationState) -> Iterable[Leaf]:
    leavesToPin: Iterable[Leaf] = frozenset()
    if state.pile == pileOrigin:
        leavesToPin = frozenset([leafOrigin])
    elif state.pile == Ling:
        leavesToPin = frozenset([Ling])
    elif state.pile == neg(Ling) + state.Shou:
        leavesToPin = frozenset([ShouLing(state.dimensionsTotal)])
    elif state.pile == Yi:
        leavesToPin = pinPileYiByCrease(state)
    elif state.pile == neg(Yi) + state.Shou:
        leavesToPin = pinPileYiAnteShouByCrease(state)
    elif state.pile == Yi + Ling:
        leavesToPin = pinPileYiLingByCrease(state)
    elif state.pile == neg(Ling + Yi) + state.Shou:
        leavesToPin = pinPileLingYiAnteShouByCrease(state)
    elif state.pile == Er:
        leavesToPin = pinPileErByCrease(state)
    elif state.pile == neg(Er) + state.Shou:
        leavesToPin = pinPileErAnteShouByCrease(state)
    elif state.pile == neg(Ling) + ShouLing(state.dimensionsTotal):
        leavesToPin = pinPileLingAnteShouLingAfterDepth4(state)
    return leavesToPin

def pinPilesAtEnds(state: EliminationState, pileDepth: int=4, maximumSizeListPermutationSpace: int=2 ** 14, CPUlimit: Limitation=None) -> EliminationState:
    if not mapShapeIs2ShangnDimensions(state.mapShape):
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
        pileProcessingOrder.extend([Ling, neg(Ling) + state.Shou])
    if 2 <= depth:
        pileProcessingOrder.extend([Yi, neg(Yi) + state.Shou])
    if 3 <= depth:
        pileProcessingOrder.extend([Yi + Ling, neg(Ling + Yi) + state.Shou])
    if 4 <= depth:
        youMustBeDimensionsTallToPinThis = 4
        if youMustBeDimensionsTallToPinThis < state.dimensionsTotal:
            pileProcessingOrder.extend([Er])
        youMustBeDimensionsTallToPinThis = 5
        if youMustBeDimensionsTallToPinThis < state.dimensionsTotal:
            pileProcessingOrder.extend([neg(Er) + state.Shou])
    return _pinPiles(state, maximumSizeListPermutationSpace, pileProcessingOrder, CPUlimit=CPUlimit)

def pinPileLingAnteShouLing(state: EliminationState, maximumSizeListPermutationSpace: int=2 ** 14, CPUlimit: Limitation=None) -> EliminationState:
    if not mapShapeIs2ShangnDimensions(state.mapShape):
        return state
    if not state.listPermutationSpace:
        state = pinPilesAtEnds(state, 0)
    state = pinPilesAtEnds(state, 4, maximumSizeListPermutationSpace)
    if not mapShapeIs2ShangnDimensions(state.mapShape, youMustBeDimensionsTallToPinThis=6):
        return state
    pileProcessingOrder: deque[Pile] = deque([neg(Ling) + ShouLing(state.dimensionsTotal)])
    return _pinPiles(state, maximumSizeListPermutationSpace, pileProcessingOrder, CPUlimit=CPUlimit)

def _pinLeavesByDomain(state: EliminationState, leaves: Sequence[Leaf], leavesDomain: Iterable[Sequence[Pile]], youMustBeDimensionsTallToPinThis: int=3, CPUlimit: Limitation=None) -> EliminationState:
    if not mapShapeIs2ShangnDimensions(state.mapShape, youMustBeDimensionsTallToPinThis=youMustBeDimensionsTallToPinThis):
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
    return state.reduceAllPermutationSpace(listFunctionsReduction2ShangnDimensional).removeCreaseViolations().moveToListFolding()

def _pinLeafByDomain(state: EliminationState, leaf: Leaf, getLeafDomain: CallableFunction[[EliminationState, Leaf], tuple[Pile, ...]], youMustBeDimensionsTallToPinThis: int=3, CPUlimit: Limitation=None) -> EliminationState:
    if not mapShapeIs2ShangnDimensions(state.mapShape, youMustBeDimensionsTallToPinThis=youMustBeDimensionsTallToPinThis):
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
    return state.reduceAllPermutationSpace(listFunctionsReduction2ShangnDimensional).removeCreaseViolations().moveToListFolding()

def pinLeavesDimension0(state: EliminationState, CPUlimit: Limitation=None) -> EliminationState:
    leaves: tuple[Leaf, Leaf] = (leafOrigin, ShouLing(state.dimensionsTotal))
    return _pinLeavesByDomain(state, leaves, leavesDomain=((pileOrigin, state.pileLast),), CPUlimit=CPUlimit)

def pinLeafShouLingPlusLing(state: EliminationState, CPUlimit: Limitation=None) -> EliminationState:
    leaf: Leaf = Ling + ShouLing(state.dimensionsTotal)
    return _pinLeafByDomain(state, leaf, getLeafShouLingPlusLingDomain, CPUlimit=CPUlimit)

def pinLeavesDimensionLing(state: EliminationState, CPUlimit: Limitation=None) -> EliminationState:
    state = pinPilesAtEnds(state, 0)
    return pinLeafShouLingPlusLing(state, CPUlimit=CPUlimit)

def pinLeavesDimensionYi(state: EliminationState, CPUlimit: Limitation=None) -> EliminationState:
    leaves: tuple[Leaf, Leaf, Leaf, Leaf] = (Yi + Ling, Yi, ShouYi(state.dimensionsTotal), ShouLingYi(state.dimensionsTotal))
    return _pinLeavesByDomain(state, leaves, getDomainDimensionYi(state), CPUlimit=CPUlimit)

def pinLeavesDimensions0LingYi(state: EliminationState, CPUlimit: Limitation=None) -> EliminationState:
    state = pinLeavesDimensionYi(state, CPUlimit=CPUlimit)
    return pinLeavesDimensionLing(state, CPUlimit=CPUlimit)

def pinLeavesDimensionEr(state: EliminationState, CPUlimit: Limitation=None) -> EliminationState:
    leaves: tuple[Leaf, Leaf, Leaf, Leaf] = (Er + Yi, Er + Yi + Ling, Er + Ling, Er)
    return _pinLeavesByDomain(state, leaves, getDomainDimensionEr(state), youMustBeDimensionsTallToPinThis=5, CPUlimit=CPUlimit)

def pinLeavesDimensionShouEr(state: EliminationState, CPUlimit: Limitation=None) -> EliminationState:
    leaves: tuple[Leaf, Leaf, Leaf, Leaf] = (ShouEr(state.dimensionsTotal), ShouLingEr(state.dimensionsTotal), ShouLingYiEr(state.dimensionsTotal), ShouYiEr(state.dimensionsTotal))
    return _pinLeavesByDomain(state, leaves, getDomainDimensionShouEr(state), youMustBeDimensionsTallToPinThis=5, CPUlimit=CPUlimit)

def pin3beans2(state: EliminationState, CPUlimit: Limitation=None) -> EliminationState:
    return _pinLeavesByDomain(state, (Yi + Ling, Yi), tuple(((pile, pile + 1) for pile in getLeafDomain(state, Yi + Ling))), CPUlimit=CPUlimit)

def pinShoubeans(state: EliminationState, CPUlimit: Limitation=None) -> EliminationState:
    return _pinLeavesByDomain(state, (ShouYi(state.dimensionsTotal), ShouLingYi(state.dimensionsTotal)), tuple(((pile, pile + 1) for pile in getLeafDomain(state, ShouYi(state.dimensionsTotal)))), CPUlimit=CPUlimit)

def _getLeavesCrease(state: EliminationState, leaf: Leaf) -> tuple[Leaf, ...]:
    if 0 < leaf:
        return tuple(getLeavesCreaseAnte(state, abs(leaf)))
    return tuple(getLeavesCreasePost(state, abs(leaf)))

def pinPileYiByCrease(state: EliminationState) -> Iterator[Leaf]:
    direction: CallableFunction[[int, int], int] = sub
    listCreaseIndicesExcluded: list[int] = []
    leafRoot: Leaf = raiseIfNone(state.permutationSpace.getLeaf(direction(state.pile, 1)), f'I could not find an `int` type `Leaf` at {direction(state.pile, 1)}.')
    tupleLeavesCrease: tuple[Leaf, ...] = _getLeavesCrease(state, direction(0, leafRoot))
    leafAtYiAnteShou: Leaf | None = state.permutationSpace.getLeaf(neg(Yi) + state.Shou)
    if leafAtYiAnteShou and 0 < dimensionNearestTail(leafAtYiAnteShou):
        listCreaseIndicesExcluded.extend([*range(dimensionNearestTail(leafAtYiAnteShou) - Ling, state.dimensionsTotal - Yi)])
    return exclude(tupleLeavesCrease, listCreaseIndicesExcluded)

def pinPileYiAnteShouByCrease(state: EliminationState) -> Iterator[Leaf]:
    direction: CallableFunction[[int, int], int] = add
    listCreaseIndicesExcluded: list[int] = []
    leafRoot: Leaf = raiseIfNone(state.permutationSpace.getLeaf(direction(state.pile, 1)), f'I could not find an `int` type `Leaf` at {direction(state.pile, 1)}.')
    tupleLeavesCrease: tuple[Leaf, ...] = _getLeavesCrease(state, direction(0, leafRoot))
    leafAtYi: Leaf | None = state.permutationSpace.getLeaf(Yi)
    if leafAtYi and leafAtYi.bit_length() < state.dimensionsTotal:
        listCreaseIndicesExcluded.extend([*range(Ling, dimensionNearestShou(leafAtYi) + inclusive)])
    return exclude(tupleLeavesCrease, listCreaseIndicesExcluded)

def pinPileYiLingByCrease(state: EliminationState) -> Iterator[Leaf]:
    direction: CallableFunction[[int, int], int] = sub
    listCreaseIndicesExcluded: list[int] = []
    leafRoot: Leaf = raiseIfNone(state.permutationSpace.getLeaf(direction(state.pile, 1)), f'I could not find an `int` type `Leaf` at {direction(state.pile, 1)}.')
    tupleLeavesCrease: tuple[Leaf, ...] = _getLeavesCrease(state, direction(0, leafRoot))
    leafAtYi: Leaf = raiseIfNone(state.permutationSpace.getLeaf(Yi))
    leafAtYiAnteShou: Leaf = raiseIfNone(state.permutationSpace.getLeaf(neg(Yi) + state.Shou))
    if 1 < len(tupleLeavesCrease):
        listCreaseIndicesExcluded.append(0)
    if isEvenMa(leafAtYiAnteShou) and leafAtYi == Ling + ShouLing(state.dimensionsTotal):
        listCreaseIndicesExcluded.extend([*range(dimensionNearestTail(leafAtYiAnteShou) + Ling, state.dimensionsTotal)])
    return exclude(tupleLeavesCrease, listCreaseIndicesExcluded)

def pinPileLingYiAnteShouByCrease(state: EliminationState) -> Iterator[Leaf]:
    direction: CallableFunction[[int, int], int] = add
    listCreaseIndicesExcluded: list[int] = []
    leafRoot: Leaf = raiseIfNone(state.permutationSpace.getLeaf(direction(state.pile, 1)), f'I could not find an `int` type `Leaf` at {direction(state.pile, 1)}.')
    tupleLeavesCrease: tuple[Leaf, ...] = _getLeavesCrease(state, direction(0, leafRoot))
    leafAtYi: Leaf = raiseIfNone(state.permutationSpace.getLeaf(Yi))
    leafAtYiAnteShou: Leaf = raiseIfNone(state.permutationSpace.getLeaf(neg(Yi) + state.Shou))
    if leafAtYiAnteShou < ShouLingYi(state.dimensionsTotal):
        listCreaseIndicesExcluded.append(-1)
    if leafAtYiAnteShou == Ling + ShouLing(state.dimensionsTotal) and leafAtYi != Yi + Ling:
        listCreaseIndicesExcluded.extend([*range(dimensionNearestShou(leafAtYi) - Ling)])
    return exclude(tupleLeavesCrease, listCreaseIndicesExcluded)

def pinPileErByCrease(state: EliminationState) -> Iterator[Leaf]:
    direction: CallableFunction[[int, int], int] = sub
    listCreaseIndicesExcluded: list[int] = []
    leafRoot: Leaf = raiseIfNone(state.permutationSpace.getLeaf(direction(state.pile, 1)), f'I could not find an `int` type `Leaf` at {direction(state.pile, 1)}.')
    tupleLeavesCrease: tuple[Leaf, ...] = _getLeavesCrease(state, direction(0, leafRoot))
    leafAtYi: Leaf = raiseIfNone(state.permutationSpace.getLeaf(Yi))
    leafAtYiAnteShou: Leaf = raiseIfNone(state.permutationSpace.getLeaf(neg(Yi) + state.Shou))
    leafAtYiLing: Leaf = raiseIfNone(state.permutationSpace.getLeaf(Yi + Ling))
    leafAtLingYiAnteShou: Leaf = raiseIfNone(state.permutationSpace.getLeaf(neg(Ling + Yi) + state.Shou))
    if isOddMa(leafAtYiLing):
        listCreaseIndicesExcluded.extend([*range(dimensionNearestShou(leafAtYiLing), 5), ptount(leafAtYiLing)])
        listCreaseIndicesExcluded.append((dimensionIndex(leafInSubHyperplane(leafAtYiAnteShou)) + 4) % 5)
    if isEvenMa(leafAtYiLing):
        listCreaseIndicesExcluded.extend([*range(state.dimensionsTotal - 3)][state.dimensionsTotal - 3 - (state.dimensionsTotal - 2 - leafInSubHyperplane(leafAtLingYiAnteShou - (leafAtLingYiAnteShou.bit_count() - isEvenMa(leafAtLingYiAnteShou))).bit_count()) % (state.dimensionsTotal - 2) - isEvenMa(leafAtLingYiAnteShou):None])
        if isEvenMa(leafAtYiAnteShou):
            listCreaseIndicesExcluded.extend([*range(dimensionNearestTail(leafInSubHyperplane(leafAtYiAnteShou)) - Yi, state.dimensionsTotal - 3)])
    if leafAtYi == Ling + ShouLing(state.dimensionsTotal):
        listCreaseIndicesExcluded.extend([(dimensionIndex(leafInSubHyperplane(leafAtYiAnteShou)) + 4) % 5, dimensionNearestTail(leafAtLingYiAnteShou) - 1])
        if Ling + ShouLing(state.dimensionsTotal) < leafAtLingYiAnteShou:
            listCreaseIndicesExcluded.extend([*range(int(leafAtLingYiAnteShou - int(bit_flip(0, dimensionNearestShou(leafAtLingYiAnteShou)))).bit_length() - 1, state.dimensionsTotal - 2)])
        if 0 < leafAtYiLing - leafAtYi <= bit_flip(0, state.dimensionsTotal - 4) and 0 < leafAtYiAnteShou - leafAtYiLing <= bit_flip(0, state.dimensionsTotal - 3):
            listCreaseIndicesExcluded.extend([ptount(leafAtYiLing), state.dimensionsTotal - 3, state.dimensionsTotal - 4])
    return exclude(tupleLeavesCrease, listCreaseIndicesExcluded)

def pinPileErAnteShouByCrease(state: EliminationState) -> Iterator[Leaf]:
    direction: CallableFunction[[int, int], int] = add
    listCreaseIndicesExcluded: list[int] = []
    leafRoot: Leaf = raiseIfNone(state.permutationSpace.getLeaf(direction(state.pile, 1)), f'I could not find an `int` type `Leaf` at {direction(state.pile, 1)}.')
    tupleLeavesCrease: tuple[Leaf, ...] = _getLeavesCrease(state, direction(0, leafRoot))
    leafAtYi: Leaf = raiseIfNone(state.permutationSpace.getLeaf(Yi))
    leafAtYiAnteShou: Leaf = raiseIfNone(state.permutationSpace.getLeaf(neg(Yi) + state.Shou))
    leafAtYiLing: Leaf = raiseIfNone(state.permutationSpace.getLeaf(Yi + Ling))
    leafAtLingYiAnteShou: Leaf = raiseIfNone(state.permutationSpace.getLeaf(neg(Ling + Yi) + state.Shou))
    leafAtEr: Leaf = raiseIfNone(state.permutationSpace.getLeaf(Er))
    addendDimensionShouLing: int = leafAtLingYiAnteShou - leafAtYiAnteShou
    addendDimensionYiLing: int = leafAtEr - leafAtYiLing
    addendDimensionYi: int = leafAtYiLing - leafAtYi
    addendDimensionLing: int = leafAtYi - Ling
    if addendDimensionYiLing in {Yi, Er, San, Si} or (addendDimensionYiLing == Wu and addendDimensionShouLing != Yi) or addendDimensionYi in {Er, San} or (addendDimensionYi == Yi and (not (addendDimensionLing == addendDimensionShouLing and addendDimensionYiLing < 0))):
        if leafAtLingYiAnteShou == ShouYi(state.dimensionsTotal):
            if addendDimensionLing == San:
                listCreaseIndicesExcluded.append(dimensionIndex(Er))
            if addendDimensionLing == Wu:
                if addendDimensionYi == Er:
                    listCreaseIndicesExcluded.append(dimensionIndex(Er))
                if addendDimensionYi == San:
                    listCreaseIndicesExcluded.append(dimensionIndex(San))
            if addendDimensionYiLing == San:
                listCreaseIndicesExcluded.append(dimensionIndex(Er))
        if 0 < (dimensionTail := dimensionNearestTail(leafAtLingYiAnteShou)) < 5:
            listCreaseIndicesExcluded.extend(list(range(dimensionTail % 4)) or [dimensionIndex(Yi)])
        if addendDimensionShouLing == neg(Wu):
            listCreaseIndicesExcluded.append(dimensionIndex(Yi))
        if addendDimensionShouLing == Yi:
            listCreaseIndicesExcluded.append(dimensionIndex(Er))
        if addendDimensionShouLing == Si:
            if addendDimensionLing == San:
                listCreaseIndicesExcluded.extend([*range(dimensionIndex(Yi), dimensionIndex(Er) + inclusive)])
            if addendDimensionYi == Yi:
                if addendDimensionYiLing == San:
                    listCreaseIndicesExcluded.append(dimensionIndex(Er))
        if addendDimensionLing == Yi:
            listCreaseIndicesExcluded.append(dimensionIndex(Yi))
            if addendDimensionYiLing == San:
                listCreaseIndicesExcluded.extend([*range(dimensionIndex(Er), dimensionIndex(San) + inclusive)])
            if addendDimensionYiLing == Si:
                listCreaseIndicesExcluded.extend([*range(dimensionIndex(San), dimensionIndex(Si) + inclusive)])
        if addendDimensionLing == Er:
            listCreaseIndicesExcluded.extend([*range(dimensionIndex(Yi), dimensionIndex(Er) + inclusive)])
        if addendDimensionLing == San:
            listCreaseIndicesExcluded.append(dimensionIndex(San))
        if addendDimensionYi == Er:
            listCreaseIndicesExcluded.append(dimensionIndex(Yi))
        if addendDimensionYi == San:
            listCreaseIndicesExcluded.extend([*range(dimensionIndex(Yi), dimensionIndex(Er) + inclusive)])
        if addendDimensionYi == Si:
            listCreaseIndicesExcluded.append(dimensionIndex(Yi))
            if addendDimensionYiLing == San:
                listCreaseIndicesExcluded.extend([*range(dimensionIndex(Yi), dimensionIndex(San) + inclusive)])
        if addendDimensionYiLing == Yi:
            listCreaseIndicesExcluded.append(dimensionIndex(Yi))
        if addendDimensionYiLing == Er:
            listCreaseIndicesExcluded.append(dimensionIndex(Er))
        if addendDimensionYiLing == San:
            listCreaseIndicesExcluded.append(dimensionIndex(San))
        if addendDimensionYiLing == Wu:
            listCreaseIndicesExcluded.append(dimensionIndex(Yi))
    return exclude(tupleLeavesCrease, listCreaseIndicesExcluded)

def pinPileLingAnteShouLingAfterDepth4(state: EliminationState) -> list[int]:
    leafAtYi: Leaf = raiseIfNone(state.permutationSpace.getLeaf(Yi))
    leafAtYiAnteShou: Leaf = raiseIfNone(state.permutationSpace.getLeaf(neg(Yi) + state.Shou))
    leafAtYiLing: Leaf = raiseIfNone(state.permutationSpace.getLeaf(Yi + Ling))
    leafAtLingYiAnteShou: Leaf = raiseIfNone(state.permutationSpace.getLeaf(neg(Ling + Yi) + state.Shou))
    leafAtEr: Leaf = raiseIfNone(state.permutationSpace.getLeaf(Er))
    leafAtErAnteShou: Leaf = raiseIfNone(state.permutationSpace.getLeaf(neg(Er) + state.Shou))
    dictionaryLeafOptions: dict[Pile, LeafOptions] = getDictionaryLeafOptions(state)
    listRemoveLeaves: list[int] = []
    pileExcluder: Pile = Yi
    for dimension, leaf in enumerate(getIteratorOfLeaves(dictionaryLeafOptions[pileExcluder])):
        if leaf == leafAtYi:
            if dimension < state.dimensionsTotal - 2:
                listRemoveLeaves.extend([Yi, ShouLing(state.dimensionsTotal) + leafAtYi])
            if 0 < dimension < state.dimensionsTotal - 2:
                listRemoveLeaves.extend([Yi + leafAtYi])
            if dimension == 1:
                listRemoveLeaves.extend([ShouLing(state.dimensionsTotal) + leafAtYi + Ling])
            if dimension == state.dimensionsTotal - 2:
                listRemoveLeaves.extend([ShouYi(state.dimensionsTotal), ShouYi(state.dimensionsTotal) + leafAtYi])
    del pileExcluder
    if leafAtYi == Ling + ShouLing(state.dimensionsTotal):
        listRemoveLeaves.extend([ShouYi(state.dimensionsTotal), leafAtYiAnteShou + Ling])
    if dimensionNearestShou(leafAtYi) < state.dimensionsTotal - 3:
        listRemoveLeaves.extend([Yi, leafAtYiAnteShou + Yi])
    pileExcluder = neg(Yi) + state.Shou
    for dimension, leaf in enumerate(getIteratorOfLeaves(dictionaryLeafOptions[pileExcluder])):
        if leaf == leafAtYiAnteShou:
            if dimension == 0:
                listRemoveLeaves.extend([Yi])
            if dimension < state.dimensionsTotal - 2:
                listRemoveLeaves.extend([ShouYi(state.dimensionsTotal) + leafAtYiAnteShou])
            if 0 < dimension < state.dimensionsTotal - 2:
                listRemoveLeaves.extend([getitem(state.productsOfDimensions, dimension), ShouYi(state.dimensionsTotal) + leafAtYiAnteShou - getitem(state.sumsOfProductsOfDimensions, dimension)])
            if 0 < dimension < state.dimensionsTotal - 3:
                listRemoveLeaves.extend([Ling + leafAtYiAnteShou])
            if 0 < dimension < state.dimensionsTotal - 1:
                listRemoveLeaves.extend([ShouYi(state.dimensionsTotal)])
    del pileExcluder
    if leafAtYi == Ling + ShouEr(state.dimensionsTotal) and leafAtYiAnteShou == ShouLingYi(state.dimensionsTotal):
        listRemoveLeaves.extend([ShouEr(state.dimensionsTotal), ShouLingYiEr(state.dimensionsTotal)])
    listRemoveLeaves.extend([leafAtYiLing])
    if leafAtYiLing == San + Er + Ling:
        listRemoveLeaves.extend([Er + Yi + Ling, Ling + Er + ShouLing(state.dimensionsTotal)])
    if leafAtYiLing == Ling + Er + ShouYi(state.dimensionsTotal):
        listRemoveLeaves.extend([ShouEr(state.dimensionsTotal), leafAtYiLing + getitem(state.productsOfDimensions, raiseIfNone(dimensionSecondNearestShou(leafAtYiLing))), leafAtYiLing + getitem(state.sumsOfProductsOfDimensions, raiseIfNone(dimensionSecondNearestShou(leafAtYiLing)) + 1), ShouLingYiEr(state.dimensionsTotal)])
    if leafAtYiLing == Ling + ShouYiEr(state.dimensionsTotal):
        listRemoveLeaves.extend([ShouYi(state.dimensionsTotal) + (Yi + Ling), last(getLeavesCreaseAnte(state, leafInSubHyperplane(leafAtYiLing)))])
    if leafAtYiLing == Ling + ShouLingYi(state.dimensionsTotal):
        listRemoveLeaves.extend([ShouLingYiEr(state.dimensionsTotal)])
    if isOddMa(leafAtYiLing):
        dimensionHeadSecond: int = raiseIfNone(dimensionSecondNearestShou(leafAtYiLing))
        indexByShouSecond: int = dimensionHeadSecond * decreasing + decreasing
        listRemoveLeaves.extend([getitem(state.productsOfDimensions, dimensionHeadSecond)])
        if leafAtYiLing < ShouLing(state.dimensionsTotal):
            sumsOfProductsOfDimensionsNearestShouInSubHyperplane: tuple[int, ...] = getSumsOfProductsOfDimensionsNearestShou(state.productsOfDimensions, state.dimensionsTotal, state.dimensionsTotal - 1)
            listRemoveLeaves.extend([Yi, leafAtYiLing + getitem(state.sumsOfProductsOfDimensions, state.dimensionsTotal - 1), leafAtYiLing + getitem(sumsOfProductsOfDimensionsNearestShouInSubHyperplane, indexByShouSecond)])
            if dimensionHeadSecond == 2:
                listRemoveLeaves.extend([getitem(state.sumsOfProductsOfDimensions, dimensionHeadSecond) + getitem(state.productsOfDimensions, dimensionNearestShou(leafAtYiLing)), getitem(state.sumsOfProductsOfDimensions, dimensionHeadSecond) + ShouLing(state.dimensionsTotal)])
            if dimensionHeadSecond == 3:
                listRemoveLeaves.extend([Yi + leafAtYiLing + getitem(state.productsOfDimensions, state.dimensionsTotal - 1)])
        if ShouLing(state.dimensionsTotal) < leafAtYiLing:
            listRemoveLeaves.extend([Ling + ShouLingYi(state.dimensionsTotal), getitem(state.productsOfDimensions, dimensionNearestShou(leafAtYiLing) - 1)])
    listRemoveLeaves.extend([leafAtLingYiAnteShou])
    if ShouLing(state.dimensionsTotal) < leafAtLingYiAnteShou:
        listRemoveLeaves.extend([Ling + ShouLingYi(state.dimensionsTotal)])
        if isEvenMa(leafAtLingYiAnteShou):
            listRemoveLeaves.extend([ShouYi(state.dimensionsTotal)])
            dimension: int = Yi
            if isBit1Ma(leafAtLingYiAnteShou, dimensionIndex(dimension)):
                listRemoveLeaves.extend([dimension, ShouLing(state.dimensionsTotal) + dimension + Ling, state.Shou - sum(state.productsOfDimensions[dimensionIndex(dimension):state.dimensionsTotal - 2]), leafAtLingYiAnteShou - dimension - getitem(state.sumsOfProductsOfDimensions, dimensionIndex(dimension) + 1)])
            dimension = Er
            if isBit1Ma(leafAtLingYiAnteShou, dimensionIndex(dimension)):
                listRemoveLeaves.extend([dimension, ShouLing(state.dimensionsTotal) + dimension + Ling])
                if 1 < dimensionNearestTail(leafAtLingYiAnteShou):
                    listRemoveLeaves.extend([state.Shou - sum(state.productsOfDimensions[dimensionIndex(dimension):state.dimensionsTotal - 2])])
                else:
                    listRemoveLeaves.extend([getitem(tuple(getLeavesCreaseAnte(state, leafInSubHyperplane(leafAtLingYiAnteShou))), dimensionIndex(dimension)) - Ling])
            dimension = San
            if isBit1Ma(leafAtLingYiAnteShou, dimensionIndex(dimension)):
                if 1 < dimensionNearestTail(leafAtLingYiAnteShou):
                    listRemoveLeaves.extend([dimension])
                    listRemoveLeaves.extend([state.Shou - sum(state.productsOfDimensions[dimensionIndex(dimension):state.dimensionsTotal - 2])])
                if dimensionNearestTail(leafAtLingYiAnteShou) < dimensionIndex(dimension):
                    listRemoveLeaves.extend([ShouLing(state.dimensionsTotal) + dimension + Ling])
            sheepOrGoat = 0
            shepherdOfDimensions: int = int(bit_flip(0, state.dimensionsTotal - 5))
            if leafAtLingYiAnteShou // shepherdOfDimensions & bit_mask(5) == 21:
                listRemoveLeaves.extend([Er])
                sheepOrGoat: int = ptount(leafAtLingYiAnteShou // shepherdOfDimensions)
                if 0 < sheepOrGoat < state.dimensionsTotal - 3:
                    comebackOffset: int = state.productsOfDimensions[dimensionNearestShou(leafAtLingYiAnteShou)] - Er
                    listRemoveLeaves.extend([leafAtLingYiAnteShou - comebackOffset])
                if 0 < sheepOrGoat < state.dimensionsTotal - 4:
                    comebackOffset = state.productsOfDimensions[raiseIfNone(dimensionSecondNearestShou(leafAtLingYiAnteShou))] - Er
                    listRemoveLeaves.extend([leafAtLingYiAnteShou - comebackOffset])
        if isOddMa(leafAtLingYiAnteShou):
            listRemoveLeaves.extend([Yi])
            if leafAtLingYiAnteShou & bit_mask(4) == 9:
                listRemoveLeaves.extend([11])
            sheepOrGoat = ptount(leafAtLingYiAnteShou)
            if 0 < sheepOrGoat < state.dimensionsTotal - 3:
                comebackOffset = state.productsOfDimensions[dimensionNearestShou(leafAtLingYiAnteShou)] - Yi
                listRemoveLeaves.extend([leafAtLingYiAnteShou - comebackOffset])
            if 0 < sheepOrGoat < state.dimensionsTotal - 4:
                comebackOffset = state.productsOfDimensions[raiseIfNone(dimensionSecondNearestShou(leafAtLingYiAnteShou))] - Yi
                listRemoveLeaves.extend([leafAtLingYiAnteShou - comebackOffset])
    if leafAtYi == Yi + Ling and leafAtLingYiAnteShou != next(getLeavesCreaseAnte(state, Ling + ShouLing(state.dimensionsTotal))):
        listRemoveLeaves.append(ShouYi(state.dimensionsTotal))
    dimensionHead: int = dimensionNearestShou(leafAtEr)
    creasePostAtEr: tuple[int, ...] = tuple(getLeavesCreasePost(state, leafAtEr))
    listIndicesCreasePostToKeep: list[int] = []
    if Er < leafAtEr < neg(Ling) + ShouYi(state.dimensionsTotal):
        listRemoveLeaves.extend([leafAtEr + ShouLing(state.dimensionsTotal)])
        dimension = Yi
        if isBit1Ma(leafAtEr, dimensionIndex(dimension)):
            listRemoveLeaves.extend([leafAtEr + ShouLing(state.dimensionsTotal) + dimension])
        if not isBit1Ma(leafAtEr, dimensionIndex(dimension)):
            listRemoveLeaves.extend([leafAtEr + ShouLing(state.dimensionsTotal) - dimension])
        if isOddMa(leafAtEr):
            dimension = San
            if isBit1Ma(leafAtEr, dimensionIndex(dimension)):
                listRemoveLeaves.extend([leafAtEr + ShouLing(state.dimensionsTotal) + dimension])
                dimension = Si
                if not isBit1Ma(leafAtEr, dimensionIndex(dimension)):
                    listRemoveLeaves.extend([leafAtEr + ShouLing(state.dimensionsTotal) - dimension])
    if ShouYi(state.dimensionsTotal) < leafAtEr < ShouLing(state.dimensionsTotal) and raiseIfNone(dimensionSecondNearestShou(leafAtEr)) != 2:
        listRemoveLeaves.extend([leafAtEr + ShouLing(state.dimensionsTotal)])
        if isOddMa(leafAtEr):
            dimension = Er
            if not isBit1Ma(leafAtEr, dimensionIndex(dimension)):
                listRemoveLeaves.extend([leafAtEr + ShouLing(state.dimensionsTotal) - getitem(state.sumsOfProductsOfDimensions, dimensionIndex(dimension))])
            dimension = San
            if not isBit1Ma(leafAtEr, dimensionIndex(dimension)):
                listRemoveLeaves.extend([leafAtEr + ShouLing(state.dimensionsTotal) - dimension, leafAtEr + ShouLing(state.dimensionsTotal) + getitem(state.sumsOfProductsOfDimensions, dimensionIndex(dimension))])
            dimension = Si
            if isBit1Ma(leafAtEr, dimensionIndex(dimension)):
                listRemoveLeaves.extend([leafAtEr - dimension])
    if isEvenMa(leafAtEr):
        listIndicesCreasePostToKeep.extend(range(state.dimensionsTotal - dimensionHead + 1, state.dimensionsTotal - zeroIndexed))
        listRemoveLeaves.extend([leafAtEr + Ling, leafAtEr + ShouLing(state.dimensionsTotal), leafAtEr + getitem(state.sumsOfProductsOfDimensions, state.dimensionsTotal - 1), getitem(state.productsOfDimensions, dimensionHead) + (Yi + Ling)])
        dimension = Yi
        if isBit1Ma(leafAtEr, dimensionIndex(dimension)):
            listRemoveLeaves.extend([dimension, ShouLing(state.dimensionsTotal) + dimension + Ling])
        dimension = Er
        if not isBit1Ma(leafAtEr, dimensionIndex(dimension)):
            listIndicesCreasePostToKeep.append(creasePostAtEr.index(state.productsOfDimensions[dimensionHead]))
        if leafAtEr < ShouLing(state.dimensionsTotal):
            listRemoveLeaves.extend([getitem(state.productsOfDimensions, dimensionIndex(Er)), getitem(state.sumsOfProductsOfDimensions, dimensionIndex(Er) + 1)])
        dimension = Si
        if not isBit1Ma(leafAtEr, dimensionIndex(dimension)) and ShouLing(state.dimensionsTotal) < leafAtEr:
            listRemoveLeaves.extend([getitem(state.productsOfDimensions, dimensionIndex(dimension))])
        zerosAtTheShou = 2
        if state.dimensionsTotal - zeroIndexed - dimensionHead == zerosAtTheShou:
            sumsOfProductsOfDimensionsNearestShouInSubSubHyperplane: tuple[int, ...] = getSumsOfProductsOfDimensionsNearestShou(state.productsOfDimensions, state.dimensionsTotal, state.dimensionsTotal - zerosAtTheShou)
            addendForUnknownReasons: int = -1
            leavesWeDontWant: list[int] = [aLeaf + addendForUnknownReasons for aLeaf in filter(notLeafOriginOrLeafLing, sumsOfProductsOfDimensionsNearestShouInSubSubHyperplane)]
            listRemoveLeaves.extend(leavesWeDontWant)
    if isOddMa(leafAtEr):
        if dimensionNearestTail(leafAtEr - 1) == 1:
            listRemoveLeaves.extend([Yi])
        if leafInSubHyperplane(leafAtEr) == state.sumsOfProductsOfDimensions[3]:
            listRemoveLeaves.extend([Er])
        dimension = Ling
        if isBit1Ma(leafAtEr, dimensionIndex(dimension)):
            listRemoveLeaves.extend([dimension, leafAtEr - dimension, ShouLing(state.dimensionsTotal) + dimension + Ling])
        dimension = Er
        if not isBit1Ma(leafAtEr, dimensionIndex(dimension)):
            listIndicesCreasePostToKeep.append(dimensionIndex(dimension))
        if isBit1Ma(leafAtEr, dimensionIndex(dimension)) and isBit1Ma(leafAtEr, dimensionIndex(Yi)):
            listRemoveLeaves.extend([leafAtEr - dimension, ShouLing(state.dimensionsTotal) + dimension + Ling])
        dimension = San
        if isBit1Ma(leafAtEr, dimensionIndex(dimension)):
            listRemoveLeaves.extend([leafAtEr - dimension, ShouLing(state.dimensionsTotal) + dimension + Ling])
        if not isBit1Ma(leafAtEr, dimensionIndex(dimension)):
            listIndicesCreasePostToKeep.append(dimensionIndex(dimension))
            dimension = Si
            if not isBit1Ma(leafAtEr, dimensionIndex(dimension)):
                listIndicesCreasePostToKeep.append(dimensionIndex(dimension))
        dimension = Si
        if isBit1Ma(leafAtEr, dimensionIndex(dimension)):
            dimensionBonus: int = Ling
            if isBit1Ma(leafAtEr, dimensionIndex(dimensionBonus)):
                listRemoveLeaves.extend([ShouLing(state.dimensionsTotal) + dimension + dimensionBonus])
            dimensionBonus = Er
            if isBit1Ma(leafAtEr, dimensionIndex(dimensionBonus)):
                listRemoveLeaves.extend([ShouLing(state.dimensionsTotal) + dimension + dimensionBonus])
            dimensionBonus = San
            if isBit1Ma(leafAtEr, dimensionIndex(dimensionBonus)):
                listRemoveLeaves.extend([ShouLing(state.dimensionsTotal) + dimension + dimensionBonus])
        dimension = Wu
        if isBit1Ma(leafAtEr, dimensionIndex(dimension)):
            listRemoveLeaves.extend([ShouYi(state.dimensionsTotal), Ling + ShouLingYi(state.dimensionsTotal)])
        if leafAtEr < ShouYi(state.dimensionsTotal):
            listRemoveLeaves.extend([Yi])
        if ShouYi(state.dimensionsTotal) < leafAtEr < ShouLing(state.dimensionsTotal):
            listRemoveLeaves.extend([leafAtEr + getitem(state.sumsOfProductsOfDimensions, state.dimensionsTotal - 2), ShouYi(state.dimensionsTotal) + (Yi + Ling)])
        if ShouLing(state.dimensionsTotal) < leafAtEr:
            dimension = Er
            if isBit1Ma(leafAtEr, dimensionIndex(dimension)):
                listRemoveLeaves.extend([leafAtEr - dimension, ShouLing(state.dimensionsTotal) + dimension + Ling])
            dimension = Si
            if isBit1Ma(leafAtEr, dimensionIndex(dimension)):
                listRemoveLeaves.extend([dimension, leafAtEr - dimension, ShouLing(state.dimensionsTotal) + dimension + Ling, ShouLingYiEr(state.dimensionsTotal)])
                if isBit1Ma(leafAtEr, dimensionIndex(San)):
                    listRemoveLeaves.extend([leafAtEr - Wu])
    listRemoveLeaves.extend(exclude(creasePostAtEr, listIndicesCreasePostToKeep))
    dimensionHead: int = dimensionNearestShou(leafAtErAnteShou)
    dimensionTail: int = dimensionNearestTail(leafAtErAnteShou)
    if isBit1Ma(getitem(dictionaryLeafOptions, neg(Er) + state.Shou), leafAtErAnteShou - 1):
        dimension = San
        if not isBit1Ma(leafAtErAnteShou, dimensionIndex(dimension)):
            enumerateFrom1: int = zeroIndexed
            for bitToTest, leafToRemove in enumerate(tuple(getLeavesCreaseAnte(state, leafAtErAnteShou - 1)), start=enumerateFrom1):
                if isBit1Ma(leafAtErAnteShou, bitToTest):
                    listRemoveLeaves.extend([leafToRemove])
                if dimensionHead < bitToTest:
                    listRemoveLeaves.extend([leafToRemove])
    theLastPossibleIndexOfCreaseAnteIfCountingFromTheHead: int = 1
    if isBit1Ma(leafAtErAnteShou, theLastPossibleIndexOfCreaseAnteIfCountingFromTheHead):
        creaseAnteAtErAnteShou: tuple[int, ...] = tuple(getLeavesCreaseAnte(state, leafAtErAnteShou))
        largestPossibleLengthOfListOfCreases: int = state.dimensionsTotal - 1
        if len(creaseAnteAtErAnteShou) == largestPossibleLengthOfListOfCreases:
            voodooAddend: int = 2
            if not isBit1Ma(leafAtErAnteShou, voodooAddend + theLastPossibleIndexOfCreaseAnteIfCountingFromTheHead):
                voodooMath: int = creaseAnteAtErAnteShou[largestPossibleLengthOfListOfCreases - zeroIndexed]
                listRemoveLeaves.extend([voodooMath])
    if leafAtErAnteShou != Ling + ShouYi(state.dimensionsTotal):
        listRemoveLeaves.extend([Ling + ShouLingYi(state.dimensionsTotal)])
    if howManyDimensionsHaveOddParity(leafAtErAnteShou) == 1:
        listRemoveLeaves.extend([leafInSubHyperplane(leafAtErAnteShou)])
    dimension = Er
    if isBit1Ma(leafAtErAnteShou, dimensionIndex(dimension)):
        listRemoveLeaves.extend([leafAtErAnteShou - dimension])
        if isEvenMa(leafAtErAnteShou) or (isOddMa(leafAtErAnteShou) and dimensionIndex(dimension) < dimensionsConsecutiveAtTail(state, leafAtErAnteShou)):
            listRemoveLeaves.extend([dimension])
    dimension = San
    if isBit1Ma(leafAtErAnteShou, dimensionIndex(dimension)):
        listRemoveLeaves.extend([leafAtErAnteShou - dimension])
        dimension = Si
        if isEvenMa(leafAtErAnteShou) and (not isBit1Ma(leafAtErAnteShou, dimensionIndex(dimension))):
            listRemoveLeaves.extend([leafAtErAnteShou - getitem(state.sumsOfProductsOfDimensions, dimensionIndex(dimension))])
    if dimensionTail == 3:
        listRemoveLeaves.extend([getitem(state.sumsOfProductsOfDimensionsNearestShou, dimensionTail)])
    if ShouLing(state.dimensionsTotal) < leafAtErAnteShou:
        dimension = Yi
        if isBit1Ma(leafAtErAnteShou, dimensionIndex(dimension)):
            listRemoveLeaves.extend([dimension, ShouLing(state.dimensionsTotal) + dimension + Ling])
        if isOddMa(leafAtErAnteShou) and (not isBit1Ma(leafAtErAnteShou, dimensionIndex(dimension))):
            listRemoveLeaves.extend([leafAtErAnteShou - ShouLing(state.dimensionsTotal) - dimension])
            dimension = Er
            if isBit1Ma(leafAtErAnteShou, dimensionIndex(dimension)):
                listRemoveLeaves.extend([ShouLing(state.dimensionsTotal) + getitem(state.sumsOfProductsOfDimensions, dimensionIndex(dimension))])
        dimension = Er
        if isBit1Ma(leafAtErAnteShou, dimensionIndex(dimension)):
            listRemoveLeaves.extend([ShouLing(state.dimensionsTotal) + dimension + Ling])
            dimension = San
            if isEvenMa(leafAtErAnteShou) and isBit1Ma(leafAtErAnteShou, dimensionIndex(dimension)):
                listRemoveLeaves.extend([dimension])
        dimension = Si
        if isBit1Ma(leafAtErAnteShou, dimensionIndex(dimension)):
            listRemoveLeaves.extend([leafAtErAnteShou - dimension])
        if not isBit1Ma(leafAtErAnteShou, dimensionIndex(dimension)):
            listRemoveLeaves.extend([leafAtErAnteShou + dimension])
    if isOddMa(leafAtErAnteShou):
        dimension = Ling
        if isBit1Ma(leafAtErAnteShou, dimensionIndex(dimension)):
            listRemoveLeaves.extend([Yi, leafAtErAnteShou - dimension, leafAtErAnteShou - getitem(state.productsOfDimensions, raiseIfNone(dimensionSecondNearestShou(leafAtErAnteShou)))])
    if isEvenMa(leafAtErAnteShou):
        dimension = Ling
        if not isBit1Ma(leafAtErAnteShou, dimensionIndex(dimension)):
            listRemoveLeaves.extend([leafAtErAnteShou + dimension, state.productsOfDimensions[dimensionTail], leafAtErAnteShou - state.productsOfDimensions[dimensionTail]])
        dimension = Er
        if isBit1Ma(leafAtErAnteShou, dimensionIndex(dimension)):
            listRemoveLeaves.extend([dimension])
            if ShouLing(state.dimensionsTotal) < leafAtErAnteShou < ShouLingYiEr(state.dimensionsTotal):
                listRemoveLeaves.extend([leafAtErAnteShou + dimensionTail])
                if dimensionTail == 2:
                    addendIDC: int = (state.Shou - leafAtErAnteShou) // 2
                    listRemoveLeaves.extend([addendIDC + leafAtErAnteShou])
            if leafAtErAnteShou < ShouLing(state.dimensionsTotal):
                listRemoveLeaves.extend([leafAtErAnteShou + state.sumsOfProductsOfDimensions[dimensionTail], state.Shou - leafAtErAnteShou])
        if leafAtErAnteShou < ShouLing(state.dimensionsTotal):
            listRemoveLeaves.extend([ShouYi(state.dimensionsTotal), leafAtErAnteShou + state.productsOfDimensions[dimensionNearestShou(leafAtErAnteShou) + 1]])
            dimension = San
            if not isBit1Ma(leafAtErAnteShou, dimensionIndex(dimension)):
                listRemoveLeaves.extend([dimension, leafAtErAnteShou + dimension, state.sumsOfProductsOfDimensionsNearestShou[dimensionIndex(dimension)]])
        if leafAtErAnteShou != Yi + ShouLing(state.dimensionsTotal):
            listRemoveLeaves.extend([ShouYi(state.dimensionsTotal)])
    del dimensionHead, dimensionTail
    return sorted(set(getIteratorOfLeaves(dictionaryLeafOptions[state.pile])).difference(set(listRemoveLeaves)))

def _byCrease2ShangnDimensional(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
    permutationSpaceHasNewLeaf: bool = True
    while permutationSpaceHasNewLeaf:
        permutationSpaceHasNewLeaf = False
        leafCount: int = permutationSpace.leafCount
        for (pile_k, leafSpace_k), (pile_r, leafSpace_r) in pairwise(permutationSpace.items()):
            if isLeafMa(leafSpace_k) and isLeafOptionsMa(leafSpace_r):
                pilesToUpdate: tuple[tuple[Pile, LeafOptions]] = ((pile_r, leafSpace_r),)
                leavesCrease: Iterator[Leaf] = getLeavesCreasePost(state, leafSpace_k)
            elif isLeafOptionsMa(leafSpace_k) and isLeafMa(leafSpace_r):
                pilesToUpdate = ((pile_k, leafSpace_k),)
                leavesCrease = getLeavesCreaseAnte(state, leafSpace_r)
            else:
                continue
            if not (permutationSpace := reduceLeafSpace(permutationSpace, pilesToUpdate, makeLeafAntiOptions(state.leavesTotal, set(range(state.leavesTotal)).difference(leavesCrease)))):
                return None
        if permutationSpace.leafCount < leafCount:
            permutationSpaceHasNewLeaf = True
    return permutationSpace

def _conditionalPredecessors2ShangnDimensional(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
    if not mapShapeIs2ShangnDimensions(state.mapShape, youMustBeDimensionsTallToPinThis=6):
        return permutationSpace
    leafAtPilePredecessors: dict[Leaf, dict[Pile, list[Leaf]]] = getDictionaryConditionalLeafPredecessors(state)
    permutationSpaceHasNewLeaf: bool = True
    while permutationSpaceHasNewLeaf:
        permutationSpaceHasNewLeaf = False
        leafCount: int = permutationSpace.leafCount
        for pile, leaf in DOTitems(filterPile(notPileLast(state.pileLast), filterLeaf(notLeafOriginOrLeafLing, filterLeaf(leafAtPilePredecessors.__contains__, permutationSpace.extractPinnedLeaves())))):
            if pile in leafAtPilePredecessors[leaf] and (not (permutationSpace := reduceLeafSpace(permutationSpace, DOTitems(methodcaller('extractUndeterminedPiles')(filterPile(betweenMa(pile + inclusive, state.pileLast - inclusive), permutationSpace, factory=PermutationSpace))), makeLeafAntiOptions(state.leavesTotal, leafAtPilePredecessors[leaf][pile])))):
                return None
        if permutationSpace.leafCount < leafCount:
            permutationSpaceHasNewLeaf = True
    return permutationSpace

def _crossedCreases2ShangnDimensional(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
    pileOf_kCrease: Pile = errorL33T
    pileOf_rCrease: Pile = errorL33T
    pilesForbidden: Iterable[Pile] = []
    permutationSpaceHasNewLeaf: bool = True
    generators: deque[CartesianProduct[tuple[DimensionIndex, PinnedLeaves, tuple[tuple[Pile, Leaf], tuple[Pile, Leaf]]]]] = deque()
    for dimension in range(state.dimensionsTotal):
        oddMa: Callable[[tuple[Pile, Leaf]], bool] = compose(oddLeaf2ShangnDimensionalMa(dimension), itemgetter(1))
        grouped: dict[bool, list[tuple[Pile, Leaf]]] = toolz_groupby(oddMa, DOTitems(permutationSpace.extractPinnedLeaves()))
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
            if (leaf_kCreaseIsPinned := leafPinnedMa(leavesPinnedParityOpposite, leaf_kCrease)):
                pileOf_kCrease = raiseIfNone(reverseLookup(permutationSpace, leaf_kCrease))
            if (leaf_rCreaseIsPinned := leafPinnedMa(leavesPinnedParityOpposite, leaf_rCrease)):
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
                if creaseViolationMa(pileOf_k, pileOf_r, pileOf_kCrease, pileOf_rCrease):
                    return None
                continue
            else:
                continue
            if not (permutationSpace := reduceLeafSpace(permutationSpace, DOTitems(filterPile(thisHasThatMa(pilesForbidden), permutationSpace.extractUndeterminedPiles())), leafAntiOptions)):
                return None
        if leafCount < permutationSpace.leafCount:
            permutationSpaceHasNewLeaf = True
    return permutationSpace

def _headsBeforeTails2ShangnDimensional(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
    permutationSpaceHasNewLeaf: bool = True
    while permutationSpaceHasNewLeaf:
        permutationSpaceHasNewLeaf = False
        leafCount: int = permutationSpace.leafCount
        pile1stOpen: int = 2
        for pile, leaf in DOTitems(filterPile(notPileLast(state.pileLast), filterLeaf(notLeafOriginOrLeafLing, permutationSpace.extractPinnedLeaves()))):
            dimensionHead: int = dimensionNearestShou(leaf)
            if 0 < dimensionHead and (not (permutationSpace := reduceLeafSpace(permutationSpace, DOTitems(methodcaller('extractUndeterminedPiles')(filterPile(betweenMa(pile1stOpen, pile - inclusive), permutationSpace, factory=PermutationSpace))), makeLeafAntiOptions(state.leavesTotal, range(state.productsOfDimensions[dimensionHead], state.leavesTotal, state.productsOfDimensions[dimensionHead]))))):
                return None
            dimensionTail: int = dimensionNearestTail(leaf)
            if 0 < dimensionTail and (not (permutationSpace := reduceLeafSpace(permutationSpace, DOTitems(methodcaller('extractUndeterminedPiles')(filterPile(betweenMa(pile + inclusive, state.pileLast - inclusive), permutationSpace, factory=PermutationSpace))), makeLeafAntiOptions(state.leavesTotal, range(leafOrigin, state.sumsOfProductsOfDimensions[dimensionTail]))))):
                return None
        if permutationSpace.leafCount < leafCount:
            permutationSpaceHasNewLeaf = True
    return permutationSpace

def _noConsecutiveDimensions2ShangnDimensional(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
    permutationSpaceHasNewLeaf: bool = True
    while permutationSpaceHasNewLeaf:
        permutationSpaceHasNewLeaf = False
        leafCount: int = permutationSpace.leafCount
        for (pile_k, leafSpace_k), (pile, leafSpace), (pile_r, leafSpace_r) in triplewise(sorted(DOTitems(permutationSpace))):
            if isLeafMa(leafSpace_k) and isLeafMa(leafSpace) and isLeafOptionsMa(leafSpace_r):
                pilesToUpdate: tuple[tuple[Pile, LeafOptions]] = ((pile_r, leafSpace_r),)
                leafForbidden: Leaf = leafSpace + (leafSpace - leafSpace_k)
            elif isLeafMa(leafSpace_k) and isLeafOptionsMa(leafSpace) and isLeafMa(leafSpace_r):
                pilesToUpdate = ((pile, leafSpace),)
                leafForbidden = (leafSpace_k + leafSpace_r) // 2
            elif isLeafOptionsMa(leafSpace_k) and isLeafMa(leafSpace) and isLeafMa(leafSpace_r):
                pilesToUpdate = ((pile_k, leafSpace_k),)
                leafForbidden = leafSpace - (leafSpace_r - leafSpace)
            else:
                continue
            if 0 <= leafForbidden < state.leavesTotal and (not (permutationSpace := reduceLeafSpace(permutationSpace, pilesToUpdate, makeLeafAntiOptions(state.leavesTotal, [leafForbidden])))):
                return None
        if permutationSpace.leafCount < leafCount:
            permutationSpaceHasNewLeaf = True
    return permutationSpace
listFunctionsReduction2ShangnDimensional: Sequence[Callable[[EliminationState, PermutationSpace], PermutationSpace | None]] = (reducePermutationSpace_LeafIsPinned, _byCrease2ShangnDimensional, reducePermutationSpace_leafDomainOf1, reducePermutationSpace_nakedSubset, _headsBeforeTails2ShangnDimensional, _conditionalPredecessors2ShangnDimensional, _crossedCreases2ShangnDimensional, _noConsecutiveDimensions2ShangnDimensional)
_dimensionLength: int = 2
_dimensionIndex: DimensionIndex = 0
Ling: int = _dimensionLength ** _dimensionIndex
_base: int = _dimensionLength
_dimensionIndex += 1
_power: int = _dimensionIndex
Yi: int = _base ** _power
_radix: int = _dimensionLength
_dimensionIndex += 1
_place_ValueIndex: int = _dimensionIndex
Er: int = _radix ** _place_ValueIndex
San: int = _dimensionLength ** 3
Si: int = _dimensionLength ** 4
Wu: int = _dimensionLength ** 5
Liu: int = _dimensionLength ** 6
Qi: int = _dimensionLength ** 7
Ba: int = _dimensionLength ** 8
Jiu: int = _dimensionLength ** 9

@cache
def dimensionIndex(dimensionAsNonnegativeInteger: int, dimensionLength: int=_dimensionLength) -> DimensionIndex:
    return int(log(dimensionAsNonnegativeInteger, dimensionLength))

@cache
def ShouLing(dimensionsTotal: int) -> int:
    return int('1' + '0' * (dimensionsTotal - 1), _dimensionLength)

@cache
def ShouLingYi(dimensionsTotal: int) -> int:
    return int('11' + '0' * (dimensionsTotal - 2), _dimensionLength)

@cache
def ShouLingYiEr(dimensionsTotal: int) -> int:
    return int('111' + '0' * (dimensionsTotal - 3), _dimensionLength)

@cache
def ShouLingEr(dimensionsTotal: int) -> int:
    return int('101' + '0' * (dimensionsTotal - 3), _dimensionLength)

@cache
def ShouYi(dimensionsTotal: int) -> int:
    return int('01' + '0' * (dimensionsTotal - 2), _dimensionLength)

@cache
def ShouYiEr(dimensionsTotal: int) -> int:
    return int('011' + '0' * (dimensionsTotal - 3), _dimensionLength)

@cache
def ShouEr(dimensionsTotal: int) -> int:
    return int('001' + '0' * (dimensionsTotal - 3), _dimensionLength)

@cache
def ShouSan(dimensionsTotal: int) -> int:
    return int('0001' + '0' * (dimensionsTotal - 4), _dimensionLength)

@cache
def ShouLingYiErSan(dimensionsTotal: int) -> int:
    return int('1111' + '0' * (dimensionsTotal - 4), _dimensionLength)

@cache
def ShouLingYiSan(dimensionsTotal: int) -> int:
    return int('1101' + '0' * (dimensionsTotal - 4), _dimensionLength)

@cache
def ShouLingErSan(dimensionsTotal: int) -> int:
    return int('1011' + '0' * (dimensionsTotal - 4), _dimensionLength)

@cache
def ShouLingSan(dimensionsTotal: int) -> int:
    return int('1001' + '0' * (dimensionsTotal - 4), _dimensionLength)

@cache
def ShouYiErSan(dimensionsTotal: int) -> int:
    return int('0111' + '0' * (dimensionsTotal - 4), _dimensionLength)

@cache
def ShouYiSan(dimensionsTotal: int) -> int:
    return int('0101' + '0' * (dimensionsTotal - 4), _dimensionLength)

@cache
def ShouErSan(dimensionsTotal: int) -> int:
    return int('0011' + '0' * (dimensionsTotal - 4), _dimensionLength)

def mapShapeIs2ShangnDimensions(mapShape: tuple[int, ...], youMustBeDimensionsTallToPinThis: int=3) -> bool:
    return youMustBeDimensionsTallToPinThis <= len(mapShape) and all(map(2 .__eq__, mapShape))

def dimensionsConsecutiveAtTail(state: EliminationState, integerNonnegative: int) -> int:
    return bit_scan1(invertLeafIn2ShangnDimensions(state.dimensionsTotal, integerNonnegative)) or 0

@cache
def dimensionNearestShou(integerNonnegative: int) -> int:
    return max(0, integerNonnegative.bit_length() - 1)

@cache
def dimensionSecondNearestShou(integerNonnegative: int) -> int | None:
    anotherInteger: int = int(bit_flip(integerNonnegative, dimensionNearestShou(integerNonnegative)))
    if anotherInteger == 0:
        dimensionSecondNearest: int | None = None
    else:
        dimensionSecondNearest = dimensionNearestShou(anotherInteger)
    return dimensionSecondNearest

@cache
def dimensionThirdNearestShou(integerNonnegative: int) -> int | None:
    dimensionNearest: int = dimensionNearestShou(integerNonnegative)
    dimensionSecondNearest: int | None = dimensionSecondNearestShou(integerNonnegative)
    if dimensionSecondNearest in {0, None}:
        dimensionThirdNearest: int | None = None
    else:
        anotherInteger: int = int(bit_flip(integerNonnegative, dimensionNearest).bit_flip(raiseIfNone(dimensionSecondNearest)))
        if anotherInteger == 0:
            dimensionThirdNearest = None
        else:
            dimensionThirdNearest = dimensionNearestShou(anotherInteger)
    return dimensionThirdNearest

@cache
def dimensionFourthNearestShou(integerNonnegative: int) -> int | None:
    dimensionNearest: int = dimensionNearestShou(integerNonnegative)
    dimensionSecondNearest: int | None = dimensionSecondNearestShou(integerNonnegative)
    dimensionThirdNearest: int | None = dimensionThirdNearestShou(integerNonnegative)
    if dimensionThirdNearest in {0, None}:
        dimensionFourthNearest: int | None = None
    else:
        anotherInteger: int = int(bit_flip(integerNonnegative, dimensionNearest).bit_flip(raiseIfNone(dimensionSecondNearest)).bit_flip(raiseIfNone(dimensionThirdNearest)))
        if anotherInteger == 0:
            dimensionFourthNearest = None
        else:
            dimensionFourthNearest = dimensionNearestShou(anotherInteger)
    return dimensionFourthNearest

@cache
def leafInSubHyperplane(notLeafOrigin: int) -> int:
    return int(f_mod_2exp(notLeafOrigin, dimensionNearestShou(notLeafOrigin)))

@cache
def dimensionNearestTail(integerNonnegative: int) -> int:
    return bit_scan1(integerNonnegative) or 0

@cache
def howManyDimensionsHaveOddParity(integerNonnegative: int) -> int:
    return max(0, integerNonnegative.bit_count() - 1)

@cache
def invertLeafIn2ShangnDimensions(dimensionsTotal: int, integerNonnegative: int) -> int:
    return int(integerNonnegative ^ bit_mask(dimensionsTotal))

@cache
def ptount(integerAbove3: int) -> int:
    return leafInSubHyperplane(integerAbove3 - (Yi + Ling)).bit_count()

def getLeavesCreaseAnte(state: EliminationState, leaf: Leaf) -> Iterator[Leaf]:
    return iter(_getCreases(state, leaf, increase=False))

def getLeavesCreasePost(state: EliminationState, leaf: Leaf) -> Iterator[Leaf]:
    return iter(_getCreases(state, leaf, increase=True))

def _getCreases(state: EliminationState, leaf: Leaf, increase: bool=True) -> tuple[Leaf, ...]:
    return _makeCreases(leaf, state.dimensionsTotal)[increase]

@cache
def _makeCreases(leaf: Leaf, dimensionsTotal: int) -> tuple[tuple[Leaf, ...], tuple[Leaf, ...]]:
    listLeavesCrease: list[Leaf] = [int(bit_flip(leaf, dimension)) for dimension in range(dimensionsTotal)]
    if leaf == leafOrigin:
        listLeavesCreasePost: list[Leaf] = [1]
        listLeavesCreaseAnte: list[Leaf] = []
    else:
        slicingIndices: int = isOddMa(howManyDimensionsHaveOddParity(leaf))
        slicerAnte: slice = slice(slicingIndices, dimensionNearestShou(leaf) * bit_flip(slicingIndices, 0) or None)
        slicerPost: slice = slice(bit_flip(slicingIndices, 0), dimensionNearestShou(leaf) * slicingIndices or None)
        if isEvenMa(leaf):
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
    if mapShapeIs2ShangnDimensions(state.mapShape):
        originPinned: bool = leaf == leafOrigin
        return range(state.sumsOfProductsOfDimensions[dimensionNearestTail(leaf) + inclusive] + howManyDimensionsHaveOddParity(leaf) - originPinned, state.sumsOfProductsOfDimensionsNearestShou[dimensionNearestShou(leaf)] + 2 - howManyDimensionsHaveOddParity(leaf) - originPinned, 2 + 2 * (leaf == ShouLing(dimensionsTotal) + Ling))
    return range(leavesTotal)

def getDomainDimensionYi(state: EliminationState) -> tuple[tuple[int, int, int, int], ...]:
    domainYiLing: tuple[int, ...] = tuple(getLeafDomain(state, Yi + Ling))
    domainShouYi: tuple[int, ...] = tuple(getLeafDomain(state, ShouYi(state.dimensionsTotal)))
    return _getDomainDimensionYi(domainYiLing, domainShouYi, state.dimensionsTotal)

@cache
def _getDomainDimensionYi(domainYiLing: tuple[int, ...], domainShouYi: tuple[int, ...], dimensionsTotal: int) -> tuple[tuple[int, int, int, int], ...]:
    domainCombined: list[tuple[int, int, int, int]] = []
    for pileOfLeafYiLing in domainYiLing:
        domainOfLeafShouYi: tuple[int, ...] = domainShouYi
        pilesTotal: int = len(domainOfLeafShouYi)
        listIndicesPilesExcluded: list[int] = []
        if pileOfLeafYiLing <= ShouEr(dimensionsTotal):
            pass
        elif ShouEr(dimensionsTotal) < pileOfLeafYiLing < ShouYi(dimensionsTotal):
            listIndicesPilesExcluded.extend([*range(1, pilesTotal // 2), *range(1 + pilesTotal // 2, 3 * pilesTotal // 4)])
        elif pileOfLeafYiLing == ShouYi(dimensionsTotal):
            listIndicesPilesExcluded.extend([*range(1, pilesTotal // 2)])
        elif ShouYi(dimensionsTotal) < pileOfLeafYiLing < ShouLing(dimensionsTotal) - Yi:
            listIndicesPilesExcluded.extend([*range(3 * pilesTotal // 4)])
        elif pileOfLeafYiLing == ShouLing(dimensionsTotal) - Yi:
            listIndicesPilesExcluded.extend([*range(1, 3 * pilesTotal // 4)])
        elif pileOfLeafYiLing == ShouLing(dimensionsTotal):
            listIndicesPilesExcluded.extend([*range(2, pilesTotal // 2)])
        domainOfLeafShouYi = tuple(exclude(domainOfLeafShouYi, listIndicesPilesExcluded))
        domainCombined.extend([(pileOfLeafYiLing, pileOfLeafYiLing + 1, pileOfLeafShouYi, pileOfLeafShouYi + 1) for pileOfLeafShouYi in domainOfLeafShouYi])
    return tuple(filter(allUniqueMa, domainCombined))

def getDomainDimensionEr(state: EliminationState) -> tuple[tuple[int, int, int, int], ...]:
    domainErLingandEr: tuple[tuple[int, int], ...] = getDomainErLingandEr(state)
    domainErYiLingandErYi: tuple[tuple[int, int], ...] = getDomainErYiLingandErYi(state)
    return _getDomainDimensionEr(domainErLingandEr, domainErYiLingandErYi, state.dimensionsTotal)

@cache
def _getDomainDimensionEr(domainErLingandEr: tuple[tuple[int, int], ...], domainErYiLingandErYi: tuple[tuple[int, int], ...], dimensionsTotal: int) -> tuple[tuple[int, int, int, int], ...]:
    domain0corners: tuple[tuple[int, int], ...] = tuple(filter(consecutiveMa, domainErLingandEr))
    domainYicorners: tuple[tuple[int, int], ...] = tuple(filter(consecutiveMa, domainErYiLingandErYi))
    pilesTotal: int = len(domainYicorners)
    domainCombined: list[tuple[int, int, int, int]] = []
    productsOfDimensions: tuple[int, ...] = tuple((int(bit_flip(0, dimension)) for dimension in range(dimensionsTotal + 1)))
    for index, (pileOfLeafErYiLing, pileOfLeafErYi) in enumerate(domainYicorners):
        listIndicesPilesExcluded: list[int] = []
        dimensionTail: int = dimensionNearestTail(pileOfLeafErYi)
        excludeBelow: int = index
        listIndicesPilesExcluded.extend(range(excludeBelow))
        excludeAbove: int = pilesTotal
        if pileOfLeafErYi <= ShouYi(dimensionsTotal):
            if dimensionTail == 1:
                excludeAbove = pilesTotal // 2 + index
                if howManyDimensionsHaveOddParity(pileOfLeafErYi) == 2:
                    excludeAbove -= 1
                if howManyDimensionsHaveOddParity(pileOfLeafErYi) == 1 and 2 < dimensionNearestShou(pileOfLeafErYi):
                    excludeAbove += 2
                if howManyDimensionsHaveOddParity(pileOfLeafErYi) == 1 and dimensionNearestShou(pileOfLeafErYi) - raiseIfNone(dimensionSecondNearestShou(pileOfLeafErYi)) < 2:
                    addend: int = productsOfDimensions[dimensionsTotal - 2] + 4
                    excludeAbove = domain0corners.index((pileOfLeafErYi + addend, pileOfLeafErYiLing + addend))
            else:
                excludeAbove = 3 * pilesTotal // 4 + 2
                if index == 0:
                    excludeAbove = 1
                elif index <= 2:
                    addend = San + sum(productsOfDimensions[1:dimensionsTotal - 2])
                    excludeAbove = domain0corners.index((pileOfLeafErYi + addend, pileOfLeafErYiLing + addend))
        listIndicesPilesExcluded.extend(range(excludeAbove, pilesTotal))
        if pileOfLeafErYi < ShouYiEr(dimensionsTotal):
            if dimensionTail == 4:
                addend = int(bit_flip(0, dimensionTail))
                start: int = domain0corners.index((pileOfLeafErYi + addend, pileOfLeafErYiLing + addend))
                listIndicesPilesExcluded.extend([*range(start, start + dimensionTail)])
            if dimensionTail == 3:
                addend = int(bit_flip(0, dimensionTail))
                start = domain0corners.index((pileOfLeafErYi + addend, pileOfLeafErYiLing + addend))
                listIndicesPilesExcluded.extend([*range(start, start + dimensionTail - 1)])
                start = domain0corners.index((pileOfLeafErYi + addend * 2, pileOfLeafErYiLing + addend * 2))
                listIndicesPilesExcluded.extend([*range(start - 1, start + dimensionTail - 1)])
            if dimensionTail < 3 and 2 < dimensionNearestShou(pileOfLeafErYi):
                if 5 < dimensionsTotal:
                    addend = Si
                    start = domain0corners.index((pileOfLeafErYi + addend, pileOfLeafErYiLing + addend))
                    stop: int = start + addend
                    step: int = 2
                    if dimensionTail == 1 and dimensionNearestShou(pileOfLeafErYi) == 4:
                        start += 2
                        stop = start + 1
                    if dimensionTail == 2:
                        start += 3
                        if dimensionNearestShou(pileOfLeafErYi) == 4:
                            start -= 2
                        stop = start + dimensionTail + inclusive
                    if howManyDimensionsHaveOddParity(pileOfLeafErYi) == 2:
                        stop = start + 1
                    listIndicesPilesExcluded.extend([*range(start, stop, step)])
                if dimensionNearestShou(pileOfLeafErYi) == 3 and howManyDimensionsHaveOddParity(pileOfLeafErYi) == 1 or dimensionNearestShou(pileOfLeafErYi) - raiseIfNone(dimensionSecondNearestShou(pileOfLeafErYi)) == 3:
                    addend = pileOfLeafErYi
                    start = domain0corners.index((pileOfLeafErYi + addend, pileOfLeafErYiLing + addend))
                    stop = start + 2
                    if dimensionTail == 2:
                        start += 1
                        stop += 1
                    if dimensionNearestShou(pileOfLeafErYi) == 4:
                        start += 3
                        stop += 4
                    step = 1
                    listIndicesPilesExcluded.extend([*range(start, stop, step)])
            if dimensionNearestShou(pileOfLeafErYi) == 2:
                addend = San
                start = domain0corners.index((pileOfLeafErYi + addend, pileOfLeafErYiLing + addend))
                listIndicesPilesExcluded.extend([*range(start, start + addend, 2)])
        domainCombined.extend([(pileOfLeafErYi, pileOfLeafErYiLing, pileOfLeafErLing, pileOfLeafEr) for pileOfLeafErLing, pileOfLeafEr in exclude(domain0corners, listIndicesPilesExcluded)])
    domainYinonCorners: tuple[tuple[int, int], ...] = tuple(set(domainErYiLingandErYi).difference(set(domainYicorners)))
    domainCombined.extend([(pileOfLeafYiEr, pileOfLeafErYiLing, pileOfLeafErYiLing - 1, pileOfLeafYiEr + 1) for pileOfLeafErYiLing, pileOfLeafYiEr in domainYinonCorners])
    return tuple(sorted(filter(allUniqueMa, set(domainCombined))))

def getDomainDimensionShouEr(state: EliminationState) -> tuple[tuple[int, int, int, int], ...]:
    domainShouLingErandShouEr: tuple[tuple[int, int], ...] = getDomainShouLingErandShouEr(state)
    domainShouLingYiErandShouYiEr: tuple[tuple[int, int], ...] = getDomainShouLingYiErandShouYiEr(state)
    return _getDomainDimensionShouEr(state.dimensionsTotal, domainShouLingErandShouEr, domainShouLingYiErandShouYiEr)

@cache
def _getDomainDimensionShouEr(dimensionsTotal: int, domainShouLingErandShouEr: tuple[tuple[int, int], ...], domainShouLingYiErandShouYiEr: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int, int, int], ...]:
    domain0corners: tuple[tuple[int, int], ...] = tuple(filter(consecutiveMa, domainShouLingErandShouEr))
    domainYicorners: tuple[tuple[int, int], ...] = tuple(filter(consecutiveMa, domainShouLingYiErandShouYiEr))
    pilesTotal: Leaf = len(domainYicorners)
    domainCombined: list[tuple[int, int, int, int]] = []
    for index, (pileOfLeafShouLingEr, pileOfLeafShouEr) in enumerate(domain0corners):
        listIndicesPilesExcluded: list[int] = []
        dimensionTail: int = dimensionNearestTail(pileOfLeafShouLingEr)
        excludeBelow: int = index - 1
        listIndicesPilesExcluded.extend(range(excludeBelow))
        excludeAbove: int = pilesTotal
        if dimensionTail == 1:
            excludeAbove = pilesTotal - (int(pileOfLeafShouEr ^ bit_mask(dimensionsTotal)) // 4 - 1)
            if howManyDimensionsHaveOddParity(pileOfLeafShouEr) == 3 and dimensionsTotal - dimensionNearestShou(pileOfLeafShouEr) >= 2:
                excludeAbove += 2
            if howManyDimensionsHaveOddParity(pileOfLeafShouEr) == 1 and dimensionsTotal - dimensionNearestShou(pileOfLeafShouEr) >= 2 and (dimensionNearestShou(pileOfLeafShouEr) - raiseIfNone(dimensionSecondNearestShou(pileOfLeafShouEr)) > 3):
                excludeAbove += 2
            if howManyDimensionsHaveOddParity(pileOfLeafShouEr) == 1 and dimensionNearestShou(pileOfLeafShouEr) - raiseIfNone(dimensionSecondNearestShou(pileOfLeafShouEr)) > 4:
                excludeAbove += 2
            if howManyDimensionsHaveOddParity(pileOfLeafShouEr) == dimensionsTotal - dimensionNearestShou(pileOfLeafShouEr) and 4 <= dimensionNearestShou(pileOfLeafShouEr) and (howManyDimensionsHaveOddParity(pileOfLeafShouEr) > 1):
                excludeAbove -= 1
        else:
            if ShouLingEr(dimensionsTotal) <= pileOfLeafShouLingEr:
                excludeAbove = pilesTotal - 1
            if ShouLing(dimensionsTotal) < pileOfLeafShouLingEr < ShouLingEr(dimensionsTotal):
                excludeAbove = pilesTotal - (int(pileOfLeafShouLingEr ^ bit_mask(dimensionsTotal)) // 8 - 1)
            if ShouYiEr(dimensionsTotal) < pileOfLeafShouLingEr <= ShouLing(dimensionsTotal):
                excludeAbove = pilesTotal - int(bit_mask(dimensionsTotal - 4))
            if pileOfLeafShouLingEr == ShouYiEr(dimensionsTotal):
                excludeAbove = pilesTotal - int(bit_mask(dimensionsTotal - 4)) - 1
            if pileOfLeafShouLingEr < ShouYiEr(dimensionsTotal):
                excludeAbove = pilesTotal - int(bit_mask(dimensionsTotal - 3)) - (dimensionTail == 2)
        listIndicesPilesExcluded.extend(range(excludeAbove, pilesTotal))
        if dimensionTail == 1 and abs(pileOfLeafShouLingEr - ShouLing(dimensionsTotal)) == 2 and isEvenMa(dimensionsTotal):
            listIndicesPilesExcluded.extend([excludeAbove - 2])
        if dimensionTail != 1 and ShouYiEr(dimensionsTotal) <= pileOfLeafShouLingEr <= ShouLingYi(dimensionsTotal):
            if dimensionTail == 2 and howManyDimensionsHaveOddParity(pileOfLeafShouLingEr) + 1 != dimensionNearestShou(pileOfLeafShouLingEr) - raiseIfNone(dimensionSecondNearestShou(pileOfLeafShouLingEr)):
                listIndicesPilesExcluded.extend([pilesTotal - (int(pileOfLeafShouLingEr ^ bit_mask(dimensionsTotal)) // 8 + 2)])
                if pileOfLeafShouLingEr <= ShouLing(dimensionsTotal) and isEvenMa(dimensionsTotal):
                    listIndicesPilesExcluded.extend([pilesTotal - (int(pileOfLeafShouLingEr ^ bit_mask(dimensionsTotal)) // 4 - 1)])
            if dimensionTail == 3:
                listIndicesPilesExcluded.extend([excludeAbove - 2])
            if 3 < dimensionTail:
                listIndicesPilesExcluded.extend([pilesTotal - int(pileOfLeafShouLingEr ^ bit_mask(dimensionsTotal)) // 4])
        domainCombined.extend([(pileOfLeafShouEr, pileOfLeafShouLingEr, pileOfLeafShouLingYiEr, pileOfLeafShouYiEr) for pileOfLeafShouLingYiEr, pileOfLeafShouYiEr in exclude(domainYicorners, listIndicesPilesExcluded)])
    domain0nonCorners: tuple[tuple[int, int], ...] = tuple(set(domainShouLingErandShouEr).difference(set(domain0corners)))
    domainCombined.extend([(pileOfLeafShouEr, pileOfLeafShouLingEr, pileOfLeafShouLingEr - 1, pileOfLeafShouEr + 1) for pileOfLeafShouLingEr, pileOfLeafShouEr in domain0nonCorners])
    return tuple(sorted(filter(allUniqueMa, set(domainCombined))))

def getDomainErLingandEr(state: EliminationState) -> tuple[tuple[int, int], ...]:
    domainErLing: tuple[int, ...] = tuple(getLeafDomain(state, Er + Ling))
    domainEr: tuple[int, ...] = tuple(getLeafDomain(state, Er))
    direction: CallableFunction[[int, int], int] = add
    return _getDomainsErOrErYi(domainErLing, domainEr, direction, state.dimensionsTotal, state.sumsOfProductsOfDimensions)

def getDomainErYiLingandErYi(state: EliminationState) -> tuple[tuple[int, int], ...]:
    domainErYiLing: tuple[int, ...] = tuple(getLeafDomain(state, Er + Yi + Ling))
    domainErYi: tuple[int, ...] = tuple(getLeafDomain(state, Er + Yi))
    direction: CallableFunction[[int, int], int] = sub
    return _getDomainsErOrErYi(domainErYiLing, domainErYi, direction, state.dimensionsTotal, state.sumsOfProductsOfDimensions)

@cache
def _getDomainsErOrErYi(domainLing: tuple[int, ...], domain0: tuple[int, ...], direction: CallableFunction[[int, int], int], dimensionsTotal: int, sumsOfProductsOfDimensions: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    if direction(0, 6009) == 6009:
        ImaDomainErLingandEr: bool = True
        ImaDomainErYiLingandErYi: bool = False
    else:
        ImaDomainErLingandEr = False
        ImaDomainErYiLingandErYi = True
    domainCombined: list[tuple[int, int]] = []
    pilesTotal: int = len(domainLing)
    pilesFewerDomain0: int = pilesTotal - len(domain0)
    for indexDomainLing, pileOfLeafLing in enumerate(filter(betweenMa(pileOrigin, ShouLing(dimensionsTotal) - Ling), domainLing)):
        indicesDomain0ToExclude: list[int] = []
        dimensionTail: int = dimensionNearestTail(pileOfLeafLing - isOddMa(pileOfLeafLing))
        excludeBelowAddend: int = 0
        steppingBasisForUnknownReasons: int = indexDomainLing
        if ImaDomainErLingandEr:
            excludeBelowAddend = 0
            steppingBasisForUnknownReasons = int(bit_mask(dimensionTail - 1).bit_flip(0))
        elif ImaDomainErYiLingandErYi:
            excludeBelowAddend = int(isEvenMa(indexDomainLing) or dimensionTail)
            steppingBasisForUnknownReasons = indexDomainLing
        if ImaDomainErLingandEr:
            if pileOfLeafLing == Er:
                indicesDomain0ToExclude.extend([*range(indexDomainLing + 1)])
            if pileOfLeafLing == ShouYi(dimensionsTotal) + ShouEr(dimensionsTotal) + ShouSan(dimensionsTotal):
                indexDomain0: int = int(7 * pilesTotal / 8)
                indexDomain0 -= pilesFewerDomain0
                indicesDomain0ToExclude.extend([indexDomain0])
        excludeBelow: int = indexDomainLing + excludeBelowAddend
        excludeBelow -= pilesFewerDomain0
        indicesDomain0ToExclude.extend(range(excludeBelow))
        if pileOfLeafLing <= ShouYi(dimensionsTotal):
            excludeAbove: int = indexDomainLing + 3 * pilesTotal // 4
            excludeAbove -= pilesFewerDomain0
            indicesDomain0ToExclude.extend(range(excludeAbove, pilesTotal))
        if ShouYi(dimensionsTotal) < pileOfLeafLing < ShouLing(dimensionsTotal):
            excludeAbove = int(pileOfLeafLing ^ bit_mask(dimensionsTotal)) // 2
            indicesDomain0ToExclude.extend(range(excludeAbove, pilesTotal))
        for dimension in range(dimensionTail):
            indicesDomain0ToExclude.extend(range(steppingBasisForUnknownReasons + int(bit_mask(dimension)), pilesTotal, int(bit_flip(0, dimension + 1))))
        if dimensionTail == 1:
            if ShouEr(dimensionsTotal) < pileOfLeafLing < ShouLing(dimensionsTotal) - Ling and 2 < dimensionNearestShou(pileOfLeafLing):
                if dimensionSecondNearestShou(pileOfLeafLing) == Ling:
                    indexDomain0: int = pilesTotal // 2
                    indexDomain0 -= pilesFewerDomain0
                    if 4 < domain0[indexDomain0].bit_length():
                        indicesDomain0ToExclude.extend([indexDomain0])
                    if ShouYi(dimensionsTotal) < pileOfLeafLing:
                        indexDomain0 = -(pilesTotal // 4 - isOddMa(pileOfLeafLing))
                        indexDomain0 -= -pilesFewerDomain0
                        indicesDomain0ToExclude.extend([indexDomain0])
                if dimensionSecondNearestShou(pileOfLeafLing) == Yi:
                    indexDomain0 = pilesTotal // 2 + 2
                    indexDomain0 -= pilesFewerDomain0
                    if domain0[indexDomain0] < ShouLing(dimensionsTotal):
                        indicesDomain0ToExclude.extend([indexDomain0])
                    indexDomain0 = -(pilesTotal // 4 - 2)
                    indexDomain0 -= -pilesFewerDomain0
                    if ShouYi(dimensionsTotal) < pileOfLeafLing:
                        indicesDomain0ToExclude.extend([indexDomain0])
                if dimensionSecondNearestShou(pileOfLeafLing) == Yi + Ling:
                    indexDomain0 = -(pilesTotal // 4)
                    indexDomain0 -= -pilesFewerDomain0
                    indicesDomain0ToExclude.extend([indexDomain0])
                indexDomain0 = 3 * pilesTotal // 4
                indexDomain0 -= pilesFewerDomain0
                if pileOfLeafLing < ShouYiEr(dimensionsTotal):
                    dimensionIndexPartShou: int = dimensionsTotal
                    dimensionIndexPartYi: int = dimensionIndex(Yi)
                    dimensionIndexPartEr: int = dimensionIndex(Er)
                    indexSumsOfProductsOfDimensions: int = dimensionIndexPartShou - (dimensionIndexPartYi + dimensionIndexPartEr)
                    addend: int = sumsOfProductsOfDimensions[indexSumsOfProductsOfDimensions]
                    if ImaDomainErYiLingandErYi:
                        addend -= 1
                    pileOfLeaf0: int = addend + ShouLing(dimensionsTotal)
                    indexDomain0 = domain0.index(pileOfLeaf0)
                    indicesDomain0ToExclude.extend([indexDomain0])
                if dimensionThirdNearestShou(pileOfLeafLing) == Ling:
                    if dimensionSecondNearestShou(pileOfLeafLing) == Yi + Ling:
                        indicesDomain0ToExclude.extend([indexDomain0 - 2])
                    if dimensionNearestShou(pileOfLeafLing) == Yi + Ling:
                        indicesDomain0ToExclude.extend([indexDomain0 - 2])
        elif ShouYi(dimensionsTotal) + ShouSan(dimensionsTotal) + isOddMa(pileOfLeafLing) == pileOfLeafLing:
            indexDomain0 = 3 * pilesTotal // 4 - 1
            indexDomain0 -= pilesFewerDomain0
            indicesDomain0ToExclude.extend([indexDomain0])
        domainCombined.extend([(pileOfLeafLing, pileOfLeaf0) for pileOfLeaf0 in exclude(domain0, indicesDomain0ToExclude)])
    domainCombined.extend([(pile, direction(pile, Ling)) for pile in domainLing if direction(pile, Ling) in domain0])
    return tuple(sorted(set(domainCombined)))

def getDomainShouLingErandShouEr(state: EliminationState) -> tuple[tuple[int, int], ...]:
    domainShouLingEr: tuple[int, ...] = tuple(getLeafDomain(state, ShouLingEr(state.dimensionsTotal)))
    domainShouEr: tuple[int, ...] = tuple(getLeafDomain(state, ShouEr(state.dimensionsTotal)))
    return _getDomainShouLingErandShouEr(domainShouLingEr, domainShouEr, state.dimensionsTotal)

@cache
def _getDomainShouLingErandShouEr(domainShouLingEr: tuple[int, ...], domainShouEr: tuple[int, ...], dimensionsTotal: int) -> tuple[tuple[int, int], ...]:
    domainCombined: list[tuple[int, int]] = []
    domainLing: tuple[int, ...] = domainShouLingEr
    domain0: tuple[int, ...] = domainShouEr
    direction: CallableFunction[[int, int], int] = sub
    domainCombined.extend([(pile, direction(pile, Ling)) for pile in domainLing if direction(pile, Ling) in domain0])
    pilesTotal: int = len(domainLing)
    pilesFewerDomain0: int = pilesTotal - len(domain0)
    for index, pileOfLeafLing in enumerate(domainLing):
        if pileOfLeafLing < ShouLing(dimensionsTotal) + Ling:
            continue
        listIndicesPilesExcluded: list[int] = []
        dimensionTail: int = dimensionNearestTail(direction(pileOfLeafLing, isOddMa(pileOfLeafLing)))
        if ShouLingYi(dimensionsTotal) < pileOfLeafLing:
            excludeBelow: int = index + 3 - 3 * pilesTotal // 4
        else:
            excludeBelow = 2 + (ShouLingYi(dimensionsTotal) - direction(pileOfLeafLing, isOddMa(pileOfLeafLing))) // 2
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
            if dimensionThirdNearestShou(pileOfLeafLing) == Yi and Er + Ling <= dimensionNearestShou(pileOfLeafLing):
                indexDomain0: int = pilesTotal // 2 + 1
                indexDomain0 -= pilesFewerDomain0
                listIndicesPilesExcluded.extend([indexDomain0])
                indexDomain0: int = pilesTotal // 4 + 1
                indexDomain0 -= pilesFewerDomain0
                listIndicesPilesExcluded.extend([indexDomain0])
                if pileOfLeafLing < ShouLingYi(dimensionsTotal):
                    listIndicesPilesExcluded.extend([indexDomain0 - 2])
            if howManyDimensionsHaveOddParity(pileOfLeafLing) == Yi:
                indexDomain0 = pilesTotal // 4 + 3
                indexDomain0 -= pilesFewerDomain0
                if dimensionSecondNearestShou(pileOfLeafLing) == Yi:
                    listIndicesPilesExcluded.extend([indexDomain0])
                if dimensionSecondNearestShou(pileOfLeafLing) == Er:
                    listIndicesPilesExcluded.extend([indexDomain0])
                if dimensionNearestShou(pileOfLeafLing) == dimensionsTotal - 1 and dimensionSecondNearestShou(pileOfLeafLing) == dimensionsTotal - 3 or dimensionSecondNearestShou(pileOfLeafLing) == Er:
                    listIndicesPilesExcluded.extend([indexDomain0 - 2])
                    indexDomain0 = pilesTotal // 2 - 1
                    indexDomain0 -= pilesFewerDomain0
                    listIndicesPilesExcluded.extend([indexDomain0])
        elif ShouLingYi(dimensionsTotal) - direction(ShouSan(dimensionsTotal), isOddMa(pileOfLeafLing)) == pileOfLeafLing:
            indexDomain0 = pilesTotal // 4 + 2
            indexDomain0 -= pilesFewerDomain0
            listIndicesPilesExcluded.extend([indexDomain0])
        domainCombined.extend([(pileOfLeafLing, pileOfLeaf0) for pileOfLeaf0 in exclude(domain0, listIndicesPilesExcluded)])
    return tuple(sorted(set(domainCombined)))

def getDomainShouLingYiErandShouYiEr(state: EliminationState) -> tuple[tuple[int, int], ...]:
    domainShouLingYiEr: tuple[int, ...] = tuple(getLeafDomain(state, ShouLingYiEr(state.dimensionsTotal)))
    domainShouYiEr: tuple[int, ...] = tuple(getLeafDomain(state, ShouYiEr(state.dimensionsTotal)))
    direction: CallableFunction[[int, int], int] = add
    return _getDomainShouLingYiErandShouYiEr(domainShouLingYiEr, domainShouYiEr, direction, state.dimensionsTotal)

@cache
def _getDomainShouLingYiErandShouYiEr(domainLing: tuple[int, ...], domain0: tuple[int, ...], direction: CallableFunction[[int, int], int], dimensionsTotal: int) -> tuple[tuple[int, int], ...]:
    domainCombined: list[tuple[int, int]] = []
    pilesTotal: int = len(domainLing)
    pilesFewerDomain0: int = pilesTotal - len(domain0)
    for indexDomainLing, pileOfLeafLing in enumerate(domainLing):
        if pileOfLeafLing < ShouLing(dimensionsTotal):
            continue
        indicesDomain0ToExclude: list[int] = []
        dimensionTail: int = dimensionNearestTail(direction(pileOfLeafLing, isOddMa(pileOfLeafLing)))
        if ShouLingYi(dimensionsTotal) < pileOfLeafLing:
            excludeBelow: int = indexDomainLing + 1 - 3 * pilesTotal // 4
        else:
            excludeBelow = (ShouLingYi(dimensionsTotal) - direction(pileOfLeafLing, isOddMa(pileOfLeafLing))) // 2
        excludeBelow -= pilesFewerDomain0
        indicesDomain0ToExclude.extend(range(excludeBelow))
        excludeAbove: int = indexDomainLing + 1 - int(bit_mask(dimensionTail))
        excludeAbove -= pilesFewerDomain0
        indicesDomain0ToExclude.extend(range(excludeAbove, pilesTotal))
        steppingBasisForUnknownReasons: int = indexDomainLing
        for dimension in range(dimensionTail):
            indicesDomain0ToExclude.extend(range(steppingBasisForUnknownReasons - int(bit_mask(dimension)), decreasing, decreasing * int(bit_flip(0, dimension + 1))))
        if dimensionTail == 1:
            if dimensionThirdNearestShou(pileOfLeafLing) == Yi and Er + Ling <= dimensionNearestShou(pileOfLeafLing):
                indexDomain0: int = pilesTotal // 2
                indexDomain0 -= pilesFewerDomain0
                indicesDomain0ToExclude.extend([indexDomain0])
                indexDomain0: int = pilesTotal // 4
                indexDomain0 -= pilesFewerDomain0
                indicesDomain0ToExclude.extend([indexDomain0])
                if pileOfLeafLing < ShouLingYi(dimensionsTotal):
                    indicesDomain0ToExclude.extend([indexDomain0 - 2])
            if dimensionThirdNearestShou(pileOfLeafLing) == Yi + Ling:
                indexDomain0 = pilesTotal // 4
                indexDomain0 -= pilesFewerDomain0
                if dimensionFourthNearestShou(pileOfLeafLing) == Yi:
                    indicesDomain0ToExclude.extend([indexDomain0])
            if howManyDimensionsHaveOddParity(pileOfLeafLing) == Yi:
                indexDomain0 = pilesTotal // 4 + 2
                indexDomain0 -= pilesFewerDomain0
                if dimensionSecondNearestShou(pileOfLeafLing) == Yi:
                    indexDomain0 = domain0.index(ShouLing(dimensionsTotal) - Yi)
                    indicesDomain0ToExclude.extend([indexDomain0])
                if dimensionSecondNearestShou(pileOfLeafLing) == Er:
                    indicesDomain0ToExclude.extend([indexDomain0])
                if ShouLingEr(dimensionsTotal) < pileOfLeafLing and Er + Ling <= dimensionNearestShou(pileOfLeafLing):
                    indicesDomain0ToExclude.extend([indexDomain0 - 2])
                    indexDomain0 = pilesTotal // 2 - 2
                    indexDomain0 -= pilesFewerDomain0
                    indicesDomain0ToExclude.extend([indexDomain0])
        elif ShouLingYi(dimensionsTotal) - direction(ShouSan(dimensionsTotal), isOddMa(pileOfLeafLing)) == pileOfLeafLing:
            indexDomain0 = pilesTotal // 4 + 1
            indexDomain0 -= pilesFewerDomain0
            indicesDomain0ToExclude.extend([indexDomain0])
        domainCombined.extend([(pileOfLeafLing, pileOfLeaf0) for pileOfLeaf0 in exclude(domain0, indicesDomain0ToExclude)])
    domainCombined.extend([(pile, direction(pile, Ling)) for pile in domainLing if direction(pile, Ling) in domain0])
    return tuple(sorted(set(domainCombined)))

def getLeafShouLingPlusLingDomain(state: EliminationState, leaf: Leaf | None=None) -> tuple[Pile, ...]:
    if leaf is None:
        leaf = Ling + ShouLing(state.dimensionsTotal)
    domainShouLingPlusLing: tuple[Pile, ...] = tuple(getLeafDomain(state, leaf))
    leafYiLing: Leaf = Yi + Ling
    leafShouLingYi: Leaf = ShouLingYi(state.dimensionsTotal)
    if state.permutationSpace.leafPinnedMa(leafYiLing) and state.permutationSpace.leafPinnedMa(leafShouLingYi):
        pileOfLeafYiLing: Pile = raiseIfNone(reverseLookup(state.permutationSpace, leafYiLing))
        pileOfLeafShouLingYi: Pile = raiseIfNone(reverseLookup(state.permutationSpace, leafShouLingYi))
        domainShouLingPlusLing = _getLeafShouLingPlusLingDomain(domainShouLingPlusLing, pileOfLeafYiLing, pileOfLeafShouLingYi, state.dimensionsTotal, state.leavesTotal)
    return domainShouLingPlusLing

@cache
def _getLeafShouLingPlusLingDomain(domainShouLingPlusLing: tuple[Pile, ...], pileOfLeafYiLing: Pile, pileOfLeafShouLingYi: Pile, dimensionsTotal: int, leavesTotal: int) -> tuple[Pile, ...]:
    pilesTotal: int = ShouYi(dimensionsTotal)
    bump: int = 1 - int(pileOfLeafYiLing.bit_count() == 1)
    howMany: int = dimensionsTotal - (pileOfLeafYiLing.bit_length() + bump)
    onesInBinary: int = int(bit_mask(howMany))
    ImaPattern: int = pilesTotal - onesInBinary
    listIndicesPilesExcluded: list[int] = []
    if pileOfLeafYiLing == Er:
        listIndicesPilesExcluded.extend([Ling, Yi, Er])
    if Er < pileOfLeafYiLing <= ShouEr(dimensionsTotal):
        stop: int = pilesTotal // 2 - 1
        listIndicesPilesExcluded.extend(range(1, stop))
        aDimensionPropertyNotFullyUnderstood: int = 5
        for _dimension in loops(dimensionsTotal - aDimensionPropertyNotFullyUnderstood):
            start: int = 1 + stop
            stop += (stop + 1) // 2
            listIndicesPilesExcluded.extend([*range(start, stop)])
        listIndicesPilesExcluded.extend([*range(1 + stop, ImaPattern)])
    if ShouEr(dimensionsTotal) < pileOfLeafYiLing:
        listIndicesPilesExcluded.extend([*range(1, ImaPattern)])
    bump = 1 - int((leavesTotal - pileOfLeafShouLingYi).bit_count() == 1)
    howMany = dimensionsTotal - ((leavesTotal - pileOfLeafShouLingYi).bit_length() + bump)
    onesInBinary = int(bit_mask(howMany))
    ImaPattern = pilesTotal - onesInBinary
    aDimensionPropertyNotFullyUnderstood = 5
    if pileOfLeafShouLingYi == leavesTotal - Er:
        listIndicesPilesExcluded.extend([-Ling - 1, -Yi - 1])
        if aDimensionPropertyNotFullyUnderstood <= dimensionsTotal:
            listIndicesPilesExcluded.extend([-Er - 1])
    if ShouLingYiEr(dimensionsTotal) < pileOfLeafShouLingYi < leavesTotal - Er and ShouEr(dimensionsTotal) < pileOfLeafYiLing <= ShouLing(dimensionsTotal):
        listIndicesPilesExcluded.extend([-1])
    if ShouLingYiEr(dimensionsTotal) <= pileOfLeafShouLingYi < leavesTotal - Er:
        stop: int = pilesTotal // 2 - 1
        listIndicesPilesExcluded.extend(range((1 + inclusive) * decreasing, (stop + inclusive) * decreasing, decreasing))
        for _dimension in loops(dimensionsTotal - aDimensionPropertyNotFullyUnderstood):
            start: int = 1 + stop
            stop += (stop + 1) // 2
            listIndicesPilesExcluded.extend([*range((start + inclusive) * decreasing, (stop + inclusive) * decreasing, decreasing)])
        listIndicesPilesExcluded.extend([*range((1 + stop + inclusive) * decreasing, (ImaPattern + inclusive) * decreasing, decreasing)])
        if Er <= pileOfLeafYiLing <= ShouLing(dimensionsTotal):
            listIndicesPilesExcluded.extend([Ling, Yi, Er, pilesTotal // 2])
    if pileOfLeafShouLingYi == ShouLingYiEr(dimensionsTotal) and ShouYi(dimensionsTotal) < pileOfLeafYiLing <= ShouLing(dimensionsTotal):
        listIndicesPilesExcluded.extend([-1])
    if ShouLingYi(dimensionsTotal) < pileOfLeafShouLingYi < ShouLingYiEr(dimensionsTotal):
        if pileOfLeafYiLing in {ShouYi(dimensionsTotal), ShouLing(dimensionsTotal)}:
            listIndicesPilesExcluded.extend([-1])
        elif Er < pileOfLeafYiLing < ShouEr(dimensionsTotal):
            listIndicesPilesExcluded.extend([0])
    if pileOfLeafShouLingYi < ShouLingYiEr(dimensionsTotal):
        listIndicesPilesExcluded.extend([*range((1 + inclusive) * decreasing, (ImaPattern + inclusive) * decreasing, decreasing)])
    pileOfLeafYiLingARCHETYPICAL: int = ShouYi(dimensionsTotal)
    bump = 1 - int(pileOfLeafYiLingARCHETYPICAL.bit_count() == 1)
    howMany = dimensionsTotal - (pileOfLeafYiLingARCHETYPICAL.bit_length() + bump)
    onesInBinary = int(bit_mask(howMany))
    ImaPattern = pilesTotal - onesInBinary
    if pileOfLeafShouLingYi == leavesTotal - Er:
        if pileOfLeafYiLing == Er:
            listIndicesPilesExcluded.extend([Ling, Yi, Er, pilesTotal // 2 - 1, pilesTotal // 2])
        if Er < pileOfLeafYiLing <= ShouLing(dimensionsTotal):
            IDK: int = ImaPattern - 1
            listIndicesPilesExcluded.extend([*range(1, 3 * pilesTotal // 4), *range(1 + 3 * pilesTotal // 4, IDK)])
        if ShouYi(dimensionsTotal) < pileOfLeafYiLing <= ShouLing(dimensionsTotal):
            listIndicesPilesExcluded.extend([-1])
    if pileOfLeafShouLingYi == ShouLingYi(dimensionsTotal):
        if pileOfLeafYiLing == ShouLing(dimensionsTotal):
            listIndicesPilesExcluded.extend([-1])
        elif Er < pileOfLeafYiLing < ShouEr(dimensionsTotal) or ShouEr(dimensionsTotal) < pileOfLeafYiLing < ShouYi(dimensionsTotal):
            listIndicesPilesExcluded.extend([0])
    return tuple(exclude(domainShouLingPlusLing, listIndicesPilesExcluded))

def getDictionaryLeafDomains(state: EliminationState) -> dict[int, range]:
    return {leaf: getLeafDomain(state, leaf) for leaf in range(state.leavesTotal)}

def getDictionaryConditionalLeafPredecessors(state: EliminationState) -> dict[Leaf, dict[Pile, list[Leaf]]]:
    dictionaryConditionalLeafPredecessors: dict[Leaf, dict[Pile, list[Leaf]]] = {}
    if mapShapeIs2ShangnDimensions(state.mapShape, youMustBeDimensionsTallToPinThis=6):
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
                dictionaryPrecedence[leaf] = {aPile: [state.productsOfDimensions[dimensionNearestShou(leaf)] + state.productsOfDimensions[dimensionNearestTail(leaf)]] for aPile in list(dictionaryDomains[leaf])[0:getSumsOfProductsOfDimensionsNearestShou(state.productsOfDimensions, dimensionFromShou=dimension - 1)[dimension - 2 - countDown] // 2]}
    leaf = Ling + ShouYi(state.dimensionsTotal)
    dictionaryPrecedence[leaf] = {aPile: [2 * state.productsOfDimensions[dimensionNearestShou(leaf)] + state.productsOfDimensions[dimensionNearestTail(leaf)], 3 * state.productsOfDimensions[dimensionNearestShou(leaf)] + state.productsOfDimensions[dimensionNearestTail(leaf)]] for aPile in list(dictionaryDomains[leaf])[1:2]}
    del leaf
    leaf: Leaf = Ling + ShouLingYi(state.dimensionsTotal)
    listOfPiles = list(dictionaryDomains[leaf])
    dictionaryPrecedence[leaf] = {aPile: [] for aPile in list(dictionaryDomains[leaf])}
    sumsOfProductsOfDimensionsNearestShou: tuple[int, ...] = getSumsOfProductsOfDimensionsNearestShou(state.productsOfDimensions)
    sumsOfProductsOfDimensionsNearestShouInSubHyperplane: tuple[int, ...] = getSumsOfProductsOfDimensionsNearestShou(state.productsOfDimensions, dimensionFromShou=state.dimensionsTotal - 1)
    pileStepAbsolute = 2
    for aPile in listOfPiles[listOfPiles.index(Yi + Ling):listOfPiles.index(neg(Ling) + ShouLing(state.dimensionsTotal)) + inclusive]:
        dictionaryPrecedence[leaf][aPile].append(Ling + ShouLing(state.dimensionsTotal))
    for indexUniversal in range(state.dimensionsTotal - 2):
        leafPredecessorTheFirst: int = state.sumsOfProductsOfDimensions[indexUniversal + 2]
        leavesPredecessorInThisSeries: int = state.productsOfDimensions[howManyDimensionsHaveOddParity(leafPredecessorTheFirst)]
        for addend in range(leavesPredecessorInThisSeries):
            leafPredecessor = leafPredecessorTheFirst + addend * decreasing
            pileFirst: int = sumsOfProductsOfDimensionsNearestShou[indexUniversal] + state.sumsOfProductsOfDimensions[2] + state.productsOfDimensions[state.dimensionsTotal - (indexUniversal + 2)] - pileStepAbsolute * 2 * (howManyDimensionsHaveOddParity(leafPredecessor) - 1 + isEvenMa(leafPredecessor)) * (1 + (2 == howManyDimensionsHaveOddParity(leafPredecessor) + isEvenMa(leafPredecessor) == dimensionNearestShou(leafPredecessor)))
            for aPile in listOfPiles[listOfPiles.index(pileFirst):None]:
                dictionaryPrecedence[leaf][aPile].append(leafPredecessor)
            leafPredecessorShouLing: int = leafPredecessor + ShouLing(state.dimensionsTotal)
            if leafInSubHyperplane(leafPredecessor) == 0 and isOddMa(dimensionNearestTail(leafPredecessor)):
                dictionaryPrecedence[leaf][pileFirst].append(leafPredecessorShouLing)
            if leafPredecessorShouLing == leaf:
                continue
            pileFirst = listOfPiles[-1] - pileStepAbsolute * (howManyDimensionsHaveOddParity(leafPredecessorShouLing) - 1 + isEvenMa(leafPredecessorShouLing) - isOddMa(leafPredecessorShouLing) - int(dimensionNearestTail(leafPredecessorShouLing) == state.dimensionsTotal - 2) - int(leaf < leafPredecessorShouLing))
            for aPile in listOfPiles[listOfPiles.index(pileFirst):None]:
                dictionaryPrecedence[leaf][aPile].append(leafPredecessorShouLing)
            if indexUniversal < state.dimensionsTotal - 4 and isOddMa(dimensionNearestTail(leafPredecessor - isOddMa(leafPredecessor))):
                pileFirst = sumsOfProductsOfDimensionsNearestShouInSubHyperplane[indexUniversal] + state.sumsOfProductsOfDimensions[2 + 1 + indexUniversal] - pileStepAbsolute * 2 * (howManyDimensionsHaveOddParity(leafPredecessorShouLing) - 1 + isEvenMa(leafPredecessorShouLing) * indexUniversal - isEvenMa(leafPredecessorShouLing) * int(not bool(indexUniversal))) + state.productsOfDimensions[state.dimensionsTotal - 1 + addend * int(not bool(indexUniversal)) - (indexUniversal + 2)]
                for aPile in listOfPiles[listOfPiles.index(pileFirst) + indexUniversal:listOfPiles.index(neg(Ling) + ShouLing(state.dimensionsTotal)) - indexUniversal + inclusive]:
                    dictionaryPrecedence[leaf][aPile].append(leafPredecessorShouLing)
    del leaf, listOfPiles, sumsOfProductsOfDimensionsNearestShou, pileStepAbsolute, sumsOfProductsOfDimensionsNearestShouInSubHyperplane
    leaf: Leaf = Ling + ShouLing(state.dimensionsTotal)
    listOfPiles: list[Pile] = list(dictionaryDomains[leaf])[1:None]
    dictionaryPrecedence[leaf] = {aPile: [] for aPile in listOfPiles}
    sumsOfProductsOfDimensionsNearestShou: tuple[int, ...] = getSumsOfProductsOfDimensionsNearestShou(state.productsOfDimensions)
    pileStepAbsolute = 4
    for indexUniversal in range(state.dimensionsTotal - 2):
        leafPredecessorTheFirst: int = state.sumsOfProductsOfDimensions[indexUniversal + 2]
        leavesPredecessorInThisSeries = state.productsOfDimensions[howManyDimensionsHaveOddParity(leafPredecessorTheFirst)]
        for addend in range(leavesPredecessorInThisSeries):
            leafPredecessor: int = leafPredecessorTheFirst + addend * decreasing
            leafPredecessorShouLing: int = leafPredecessor + ShouLing(state.dimensionsTotal)
            pileFirst = sumsOfProductsOfDimensionsNearestShou[indexUniversal] + 6 - pileStepAbsolute * (howManyDimensionsHaveOddParity(leafPredecessor) - 1 + isEvenMa(leafPredecessor))
            for aPile in listOfPiles[listOfPiles.index(pileFirst):None]:
                dictionaryPrecedence[leaf][aPile].append(leafPredecessor)
                dictionaryPrecedence[leaf][aPile].append(leafPredecessorShouLing)
    del leaf, listOfPiles, sumsOfProductsOfDimensionsNearestShou, pileStepAbsolute
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
            listOptionalPiles: list[Pile] = sorted((pile for pile in tupleDomainLater if pile not in setPilesRequiring))
            for pileEarlier in tupleDomainEarlier:
                optionalLessEqualCount: int = bisect_right(listOptionalPiles, pileEarlier)
                if optionalLessEqualCount == 0:
                    listSuccessors: list[Leaf] = dictionarySuccessor.setdefault(leafEarlier, {}).setdefault(pileEarlier, [])
                    if leafLater not in listSuccessors:
                        listSuccessors.append(leafLater)
    return dictionarySuccessor

@syntacticCurry
def filterCeiling(pile: Pile, dimensionsTotal: int, leaf: Leaf) -> bool:
    return pile < int(bit_mask(dimensionsTotal) ^ bit_mask(dimensionsTotal - dimensionNearestShou(leaf))) - howManyDimensionsHaveOddParity(leaf) + 2 - (leaf == leafOrigin)

@syntacticCurry
def filterFloor(pile: Pile, leaf: Leaf) -> bool:
    return int(bit_flip(0, dimensionNearestTail(leaf) + 1)) + howManyDimensionsHaveOddParity(leaf) - 1 - (leaf == leafOrigin) <= pile

@syntacticCurry
def filterParity(pile: Pile, leaf: Leaf) -> bool:
    return pile & 1 == int(bit_flip(0, dimensionNearestTail(leaf) + 1)) + howManyDimensionsHaveOddParity(leaf) - 1 - (leaf == leafOrigin) & 1

@syntacticCurry
def filterDoubleParity(pile: Pile, dimensionsTotal: int, leaf: Leaf) -> bool:
    if leaf != ShouLing(dimensionsTotal) + Ling:
        return True
    return pile >> 1 & 1 == int(bit_flip(0, dimensionNearestTail(leaf) + 1)) + howManyDimensionsHaveOddParity(leaf) - 1 - (leaf == leafOrigin) >> 1 & 1

@cache
def _getLeafOptions(pile: Pile, dimensionsTotal: int, mapShape: tuple[int, ...], leavesTotal: int) -> LeafOptions:
    leafOptions: Iterable[Leaf] = range(leavesTotal)
    if mapShapeIs2ShangnDimensions(mapShape):
        parityMatch: Callable[[Leaf], bool] = filterParity(pile)
        pileAboveFloor: Callable[[Leaf], bool] = filterFloor(pile)
        pileBelowCeiling: Callable[[Leaf], bool] = filterCeiling(pile, dimensionsTotal)
        matchLargerStep: Callable[[Leaf], bool] = filterDoubleParity(pile, dimensionsTotal)
        leafOptions = filter(parityMatch, leafOptions)
        leafOptions = filter(pileAboveFloor, leafOptions)
        leafOptions = filter(pileBelowCeiling, leafOptions)
        leafOptions = filter(matchLargerStep, leafOptions)
    return makeLeafOptions(leavesTotal, leafOptions)

def notLeafOriginOrLeafLing(leaf: LeafSpace) -> bool:
    return Ling < leaf

@syntacticCurry
def oddLeaf2ShangnDimensionalMa(dimension: DimensionIndex, leaf: Leaf) -> bool:
    return isBit1Ma(leaf, dimension)

def creaseViolationMa(pile: Pile, pileComparand: Pile, pileCrease: Pile, pileComparandCrease: Pile) -> bool:
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

def foldingValidMa(folding: Folding, mapShape: tuple[int, ...]) -> bool:
    leavesPinned: PinnedLeaves = dict(enumerate(folding))
    leafToPile: dict[Leaf, Pile] = {leafValue: pileKey for pileKey, leafValue in DOTitems(leavesPinned)}
    for dimension in range(_dimensionsTotal(mapShape)):
        listPilePileCreaseByParity: list[deque[tuple[Pile, Pile]]] = [deque(), deque()]
        for pile, leaf in leavesPinned.items():
            crease: int | None = getCreasePost(mapShape, leaf, dimension)
            if crease:
                listPilePileCreaseByParity[oddLeafMa(mapShape, leaf, dimension)].append((pile, leafToPile[crease]))
        for groupedParity in listPilePileCreaseByParity:
            if any((creaseViolationMa(pile, pileComparand, pileCrease, pileComparandCrease) for (pile, pileCrease), (pileComparand, pileComparandCrease) in combinations(sorted(groupedParity), 2))):
                return False
    return True

def leavesPinnedValidMa(leavesPinned: PinnedLeaves, mapShape: tuple[int, ...]) -> bool:
    leafToPile: dict[Leaf, Pile] = {leafValue: pileKey for pileKey, leafValue in DOTitems(leavesPinned)}
    for dimension in range(_dimensionsTotal(mapShape)):
        listPilePileCreaseByParity: list[deque[tuple[Pile, Pile]]] = [deque(), deque()]
        for pile, leaf in leavesPinned.items():
            crease: int | None = getCreasePost(mapShape, leaf, dimension)
            if crease:
                listPilePileCreaseByParity[oddLeafMa(mapShape, leaf, dimension)].append((pile, leafToPile[crease]))
        for groupedParity in listPilePileCreaseByParity:
            if any((creaseViolationMa(pile, pileComparand, pileCrease, pileComparandCrease) for (pile, pileCrease), (pileComparand, pileComparandCrease) in combinations(sorted(groupedParity), 2))):
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
def oddLeafMa(mapShape: tuple[int, ...], leaf: Leaf, dimension: int) -> int:
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
        sherpa = sherpa.reduceAllPermutationSpace(listFunctionsReduction2ShangnDimensional).removeCreaseViolations().moveToListFolding()
        listFolding.extend(sherpa.listFolding)
        state.listPermutationSpace.extend(sherpa.listPermutationSpace)
    state.listFolding.extend(listFolding)
    return state

def doTheNeedful(state: EliminationState, workersMaximum: int) -> EliminationState:
    if not mapShapeIs2ShangnDimensions(state.mapShape):
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
    state = pinLeavesDimensionShouEr(state)
    state = pinLeavesDimensions0LingYi(state)
    workersMaximum: int = defineProcessorLimit(CPUlimit)
    print(doTheNeedful(state, workersMaximum).foldsTotal)
