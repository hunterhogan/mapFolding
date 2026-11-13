from __future__ import annotations

from mapFolding.oeis._theSSOT import settingsOEIS
from sympy import ceiling, Eq, factorial, floor, Mod, oo, Piecewise, Range, Sum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from sympy import Equality, Symbol
	from sympy.logic.boolalg import BooleanFalse, BooleanTrue

type 形Relation = tuple[Equality | BooleanFalse | BooleanTrue, Range]

n: Symbol = settingsOEIS.n
k: Symbol = settingsOEIS.k

A000108 = settingsOEIS.lookupSymbolic['A000108']
A001246 = settingsOEIS.lookupSymbolic['A001246']
A001263 = settingsOEIS.lookupSymbolic['A001263']
A001405 = settingsOEIS.lookupSymbolic['A001405']
A002415 = settingsOEIS.lookupSymbolic['A002415']
A014137 = settingsOEIS.lookupSymbolic['A014137']
A014140 = settingsOEIS.lookupSymbolic['A014140']
A053121 = settingsOEIS.lookupSymbolic['A053121']
A401693 = settingsOEIS.lookupSymbolic['A401693']

boxOfRelationsCatalan: tuple[形Relation, ...] = (
	(Eq(A000108(n), factorial(2 * n) / (factorial(n) * factorial(n + 1))), Range(0, oo))
	, (Eq((n + 1) * A000108(n), 2 * (2 * n - 1) * A000108(n - 1)), Range(1, oo))
	, (Eq(A001246(n), A000108(n) ** 2), Range(0, oo))
	, (Eq(A002415(n), n ** 2 * (n ** 2 - 1) / 12), Range(0, oo))
	, (Eq(A001263(n, 1), 1), Range(1, oo))
	, (Eq(A001263(n, n), 1), Range(1, oo))
	, (Eq(A001263(n, 2), n * (n - 1) / 2), Range(2, oo))
	, (Eq(A001263(n, 3), A002415(n - 1)), Range(3, oo))
	, (Eq(A000108(n), Sum(A001263(n, k), (k, 1, n))), Range(1, oo))
	, (Eq(A053121(2 * n, 0), A000108(n)), Range(0, oo))
	, (Eq(A053121(2 * n + 1, 0), 0), Range(0, oo))
	, (Eq(A053121(n, n), 1), Range(0, oo))
	, (Eq(A000108(n), Sum(A053121(n, k) ** 2, (k, 0, n))), Range(0, oo))
	, (Eq(Sum(A053121(n, k), (k, 0, n)), factorial(n) / (factorial(floor(n / 2)) * factorial(ceiling(n / 2)))), Range(0, oo))
	, (Eq(A014137(n), Sum(A000108(k), (k, 0, n))), Range(0, oo))
	, (Eq(A014137(n), A014137(n - 1) + A000108(n)), Range(1, oo))
	, (Eq(A014140(n), Sum((n + 1 - k) * A000108(k), (k, 0, n))), Range(0, oo))
	, (Eq(A014140(n), A014140(n - 1) + A014137(n)), Range(1, oo))
	, (Eq(A401693(2 * n), A014140(2 * n) - 2 * A014140(n - 1) - 1), Range(1, oo))
	, (Eq(A401693(2 * n + 1), A014140(2 * n + 1) - 2 * A014140(n)), Range(0, oo))
	, (Eq(A401693(2 * n), A401693(2 * n - 2) + 2 * A014137(2 * n) - 2 * A014137(n - 1) - A000108(2 * n)), Range(2, oo))
	, (Eq(A401693(2 * n + 1), A401693(2 * n - 1) + 2 * A014137(2 * n + 1) - 2 * A014137(n) - A000108(2 * n + 1)), Range(1, oo))
)

boxOfRelationsBallot: tuple[形Relation, ...] = (
	(Eq(A001405(n), factorial(n) / (factorial(floor(n / 2)) * factorial(ceiling(n / 2)))), Range(0, oo))
	, (Eq(A001405(n), Sum(A053121(n, k), (k, 0, n))), Range(0, oo))
	, (Eq(A001405(2 * n), (n + 1) * A000108(n)), Range(0, oo))
	, (Eq(A001405(2 * n), 2 * A001405(2 * n - 1)), Range(1, oo))
)

A045445 = settingsOEIS.lookupSymbolic['A045445']

boxOfRelationsCatafusenes: tuple[形Relation, ...] = (
	(Eq((n - 2) * (n + 2) * A045445(n), 3 * (n - 1) * (2 * n - 1) * A045445(n - 1)
		- 5 * (n - 2) * (n - 1) * A045445(n - 2)), Range(3, oo))
	,
)

A000682 = settingsOEIS.lookupSymbolic['A000682']
A005315 = settingsOEIS.lookupSymbolic['A005315']
A005316 = settingsOEIS.lookupSymbolic['A005316']
A060206 = settingsOEIS.lookupSymbolic['A060206']
A077014 = settingsOEIS.lookupSymbolic['A077014']
A077054 = settingsOEIS.lookupSymbolic['A077054']
A077460 = settingsOEIS.lookupSymbolic['A077460']
A078591 = settingsOEIS.lookupSymbolic['A078591']
A078592 = settingsOEIS.lookupSymbolic['A078592']
A085973 = settingsOEIS.lookupSymbolic['A085973']
A208357 = settingsOEIS.lookupSymbolic['A208357']

boxOfRelationsMeanders: tuple[形Relation, ...] = (
	(Eq(A005315(n), A005316(2 * n - 1)), Range(1, oo))
	, (Eq(A077054(n), A005316(2 * n)), Range(0, oo))
	, (Eq(A060206(n), A000682(2 * n + 1)), Range(0, oo))
	, (Eq(A077014(n), (2 - Mod(n, 2)) * A005316(n)), Range(0, oo))
	, (Eq(A085973(n), A077054(n) + A005315(n)), Range(1, oo))
	, (Eq(4 * A077460(n), A005315(n) + Piecewise((A005316(n) + A060206((n - 1) / 2), Eq(Mod(n, 2), 1)), (2 * A005316(n), True))), Range(2, oo))
	, (Eq(2 * A078591(n), A005315(n)), Range(2, oo))
	, (Eq(2 * A078592(n), A005316(2 * n) + A005316(n)), Range(0, oo))
	, (Eq(A208357(n), A005315(n + 1) ** 2), Range(1, oo))
	, (Eq(A000682(n), 1), Range(1, 3))
	, (Eq(A005315(n), 1), Range(0, 2))
	, (Eq(A005316(n), 1), Range(0, 2))
	, (Eq(A077460(n), 1), Range(0, 2))
	, (Eq(A078591(n), 1), Range(0, 2))
	, (Eq(A085973(n), 3), Range(0, 1))
)

A000136 = settingsOEIS.lookupSymbolic['A000136']
A000560 = settingsOEIS.lookupSymbolic['A000560']
A001010 = settingsOEIS.lookupSymbolic['A001010']
A001011 = settingsOEIS.lookupSymbolic['A001011']
A007822 = settingsOEIS.lookupSymbolic['A007822']
A217310 = settingsOEIS.lookupSymbolic['A217310']
A217318 = settingsOEIS.lookupSymbolic['A217318']
A223093 = settingsOEIS.lookupSymbolic['A223093']
A223094 = settingsOEIS.lookupSymbolic['A223094']
A223095 = settingsOEIS.lookupSymbolic['A223095']
A227167 = settingsOEIS.lookupSymbolic['A227167']

boxOfRelationsStamps: tuple[形Relation, ...] = (
	(Eq(A000136(n), n * A000682(n)), Range(1, oo))
	, (Eq(2 * A000560(n), A000682(n + 1)), Range(2, oo))
	, (Eq(A001010(n), Piecewise((2 * A007822((n + 1) / 2), Eq(Mod(n, 2), 1)), (2 * A000682(n / 2 + 1), True))), Range(2, oo))
	, (Eq(4 * A001011(n), A001010(n) + n * A000682(n)), Range(2, oo))
	, (Eq(A217310(n), (1 + Mod(n, 2)) * A223093(n)), Range(1, oo))
	, (Eq(2 * A217318(n), (1 + Mod(n, 2)) * A223095(n)), Range(1, oo))
	, (Eq(A223093(n), A000682(n + 1) - (2 - Mod(n, 2)) * A005316(n)), Range(1, oo))
	, (Eq(A223094(n), n * A000682(n) - A000682(n + 1)), Range(1, oo))
	, (Eq(A223095(n), A223094(n) - A223093(n)), Range(1, oo))
	, (Eq((2 - Mod(n, 2)) * A227167(n), n * A000682(n)), Range(1, oo))
	, (Eq(A227167(n), A217310(n) + A217318(n) + A005316(n)), Range(1, oo))
	, (Eq(A000682(n + 1), factorial(n) - Sum(A223094(k) * factorial(n) / factorial(k), (k, 3, n))), Range(3, oo))
	, (Eq(A001010(n), 1), Range(1, 2))
	, (Eq(A001011(n), 1), Range(1, 2))
	, (Eq(A007822(n), 1), Range(1, 2))
)

A287548 = settingsOEIS.lookupSymbolic['A287548']
A301620 = settingsOEIS.lookupSymbolic['A301620']
A322507 = settingsOEIS.lookupSymbolic['A322507']
A331499 = settingsOEIS.lookupSymbolic['A331499']
A333971 = settingsOEIS.lookupSymbolic['A333971']
A334615 = settingsOEIS.lookupSymbolic['A334615']
A337581 = settingsOEIS.lookupSymbolic['A337581']
A339179 = settingsOEIS.lookupSymbolic['A339179']

boxOfRelationsArches: tuple[形Relation, ...] = (
	(Eq(A301620(n), A000682(n + 2) - 2 * A000682(n + 1)), Range(1, oo))
	, (Eq(A333971(n), 4 * (A000682(n - 1) - A000682(n - 2))), Range(4, oo))
	, (Eq(A334615(n), A000682(n) - 4 * A000682(n - 1) + 4 * A000682(n - 2)), Range(4, oo))
	, (Eq(A337581(n), 4 * A000682(n - 2)), Range(4, oo))
	, (Eq(A334615(n + 2), A301620(n) - 2 * A301620(n - 1)), Range(2, oo))
	, (Eq(A000682(n), 2 ** (n - 2) + Sum(2 ** (n - k - 2) * A301620(k), (k, 3, n - 2))), Range(3, oo))
	, (Eq(A000682(n - 1), Sum(A339179(n, k), (k, 2, floor(n / 2) + 1))), Range(2, oo))
	, (Eq(A339179(n, 2), A000682(n - 2)), Range(4, oo))
	, (Eq(A339179(n, 3), A000682(n - 2)), Range(4, oo))
	, (Eq(A301620(n - 3), Sum(A339179(n, k), (k, 4, floor((n + 2) / 2)))), Range(4, oo))
	, (Eq(A322507(n), 0), Range(1, 2))
	, (Eq(A287548(n, 1), A000108(n)), Range(1, oo))
	, (Eq(A287548(n, n), A000682(n)), Range(1, oo))
	, (Eq(A331499(n, 1), A000682(n)), Range(1, oo))
	, (Eq(A331499(n, n), A000108(n - 1)), Range(1, oo))
	, (Eq(A331499(n, n - 1), A000108(n - 2)), Range(2, oo))
	, (Eq(A000108(n), Sum(A331499(n, k), (k, 1, n))), Range(1, oo))
	, (Eq(A333971(n), n - 1), Range(2, 4))
	, (Eq(A334615(n), 0), Range(2, 4))
	, (Eq(A337581(n), n - 1), Range(2, 4))
)

A259689 = settingsOEIS.lookupSymbolic['A259689']
A259698 = settingsOEIS.lookupSymbolic['A259698']
A259700 = settingsOEIS.lookupSymbolic['A259700']
A259701 = settingsOEIS.lookupSymbolic['A259701']
A259702 = settingsOEIS.lookupSymbolic['A259702']
A259703 = settingsOEIS.lookupSymbolic['A259703']
A260785 = settingsOEIS.lookupSymbolic['A260785']

boxOfRelationsOverlaps: tuple[形Relation, ...] = (
	(Eq(2 * A259702(n), A000682(n) - 2 * A000682(n - 1)), Range(3, oo))
	, (Eq(A301620(n), 2 * A259702(n + 2)), Range(1, oo))
	, (Eq(A000682(n + 1), Sum(k * A259689(n, k), (k, 2, floor(n / 2) + 1))), Range(2, oo))
	, (Eq(A301620(n), Sum((k - 2) * A259689(n + 1, k), (k, 2, floor((n + 1) / 2) + 1))), Range(1, oo))
	, (Eq(A259689(n, floor(n / 2) + 1), 2 ** floor((n - 1) / 2)), Range(3, oo))
	, (Eq(A259689(n, floor(n / 2)), 2 ** floor((n - 1) / 2) * (n - 4) + 2), Range(4, oo))
	, (Eq(A000682(n), Sum(A259698(n, k), (k, 1, n - 1))), Range(2, oo))
	, (Eq(A000682(n), Sum(A259703(n, k), (k, 1, n - 1))), Range(2, oo))
	, (Eq(A259702(n), Sum(A259701(n, k), (k, 2, n - 1))), Range(2, oo))
	, (Eq(A259701(n, 1), A000560(n - 1)), Range(3, oo))
	, (Eq(A259703(n, 1), A000560(n - 1)), Range(3, oo))
	, (Eq(A000682(n), Sum(A259689(n, k), (k, 2, floor(n / 2) + 1))), Range(2, oo))
	, (Eq(A259689(n, 2), A260785(n)), Range(2, oo))
	, (Eq(A259700(n), Sum(k * A259698(n, k), (k, 1, n - 1))), Range(2, oo))
	, (Eq(A322507(n), A301620(n + 1) - Sum(k * (k - 1) * A259689(n, k), (k, 2, floor(n / 2) + 1))), Range(2, oo))
	, (Eq(A259702(n), 0), Range(2, 3))
)

A006661 = settingsOEIS.lookupSymbolic['A006661']
A076876 = settingsOEIS.lookupSymbolic['A076876']
A400429 = settingsOEIS.lookupSymbolic['A400429']
A400630 = settingsOEIS.lookupSymbolic['A400630']
A400631 = settingsOEIS.lookupSymbolic['A400631']
A400632 = settingsOEIS.lookupSymbolic['A400632']
A400633 = settingsOEIS.lookupSymbolic['A400633']
A400634 = settingsOEIS.lookupSymbolic['A400634']
A400635 = settingsOEIS.lookupSymbolic['A400635']
A400636 = settingsOEIS.lookupSymbolic['A400636']
A400637 = settingsOEIS.lookupSymbolic['A400637']

boxOfRelationsWinding: tuple[形Relation, ...] = (
	(Eq(A000682(n), Sum(A400429(n, k), (k, 1, floor(n / 2)))), Range(2, oo))
	, (Eq(A400429(2 * n, 1), A005315(n)), Range(1, oo))
	, (Eq(A400429(2 * n - 1, 1), A005315(n)), Range(2, oo))
	, (Eq(A400429(2 * n, 2), A005315(n + 1) - 2 * A005316(2 * n)), Range(2, oo))
	, (Eq(A400429(n, floor(n / 2)), 2), Range(3, oo))
	, (Eq(A400429(n, floor(n / 2) - 1), (n ** 2 + 2 * n + Mod(n, 2) - 20) / 2), Range(4, oo))
	, (Eq(A400429(n, floor(n / 2) - 3), A400632(n)), Range(8, oo))
	, (Eq(A400429(n, floor(n / 2) - 4), A400633(n)), Range(10, oo))
	, (Eq(2 * A076876(n), A400429(n, 3) - A400429(n, 2) + A400429(n + 1, 1) - 2 * A400429(n + 1, 2) + A400429(n + 2, 2)), Range(6, oo, 2))
	, (Eq(2 * A006661(n + 4), A400429(n, 3) - 2 * A400429(n, 2) + 2 * A400429(n + 1, 1) - 2 * A400429(n + 1, 2) + A400429(n + 2, 2)), Range(6, oo, 2))
	, (Eq(2 * A077054(n), A400429(2 * n + 1, 1) - A400429(2 * n, 2)), Range(2, oo))
	, (Eq(A400429(n, floor(n / 2) - 1), A400630(n)), Range(4, oo))
	, (Eq(A400630(n), (n ** 2 + 2 * n + Mod(n, 2) - 20) / 2), Range(4, oo))
	, (Eq(A400630(n), A400630(n - 2) + 2 * n), Range(6, oo))
	, (Eq(A400429(n, floor(n / 2) - 2), A400631(n)), Range(6, oo))
	, (Eq(A400631(n), (13 * n ** 4 + 24 * n ** 3 - 284 * n ** 2 - 4560 * n + 17920
		+ Mod(n, 2) * (18 * n ** 2 - 72 * n + 189) - 64 * 0 ** Mod(n, 3)) / 288), Range(6, oo))
	, (Eq(A400429(n, floor(n / 2) - 5), A400634(n)), Range(12, oo))
	, (Eq(A400429(n, floor(n / 2) - 6), A400635(n)), Range(14, oo))
	, (Eq(A400429(n, floor(n / 2) - 7), A400636(n)), Range(16, oo))
	, (Eq(A400429(n, floor(n / 2) - 8), A400637(n)), Range(18, oo))
)

A002057 = settingsOEIS.lookupSymbolic['A002057']
A002694 = settingsOEIS.lookupSymbolic['A002694']
A006657 = settingsOEIS.lookupSymbolic['A006657']
A006658 = settingsOEIS.lookupSymbolic['A006658']
A006659 = settingsOEIS.lookupSymbolic['A006659']
A007746 = settingsOEIS.lookupSymbolic['A007746']
A008828 = settingsOEIS.lookupSymbolic['A008828']
A060972 = settingsOEIS.lookupSymbolic['A060972']

boxOfRelationsSystems: tuple[形Relation, ...] = (
	(Eq(A008828(n, 1), A005315(n)), Range(1, oo))
	, (Eq(A008828(n, 2), A006657(n)), Range(2, oo))
	, (Eq(A008828(n, 3), A006658(n)), Range(3, oo))
	, (Eq(A008828(n, n), A000108(n)), Range(1, oo))
	, (Eq(A008828(n + 1, n), A006659(n)), Range(1, oo))
	, (Eq(A008828(n, n - 3), A007746(n)), Range(4, oo))
	, (Eq(A001246(n), Sum(A008828(n, k), (k, 1, n))), Range(1, oo))
	, (Eq(A006659(n), 2 * factorial(2 * n + 2) / (factorial(n - 1) * factorial(n + 3))), Range(1, oo))
	, (Eq(A007746(n), 4 * factorial(2 * n) * (n ** 4 + 20 * n ** 3 + 107 * n ** 2 - 107 * n + 15)
		/ (3 * factorial(n - 4) * factorial(n + 6))), Range(4, oo))
	, (Eq(A002057(n), 4 * factorial(2 * n + 3) / (factorial(n) * factorial(n + 4))), Range(0, oo))
	, (Eq(A002057(n), A000108(n + 3) - 2 * A000108(n + 2)), Range(0, oo))
	, (Eq(A006659(n), (n + 1) * A002057(n - 1)), Range(1, oo))
	, (Eq(A002694(n), factorial(2 * n) / (factorial(n - 2) * factorial(n + 2))), Range(2, oo))
	, (Eq(A006659(n), 2 * A002694(n + 1)), Range(1, oo))
	, (Eq(A060972(n, 1), 1), Range(1, oo))
	, (Eq(A060972(n, n), 1), Range(1, oo))
	, (Eq(A060972(n, 2), A002415(n)), Range(2, oo))
	, (Eq(A060972(n, n - 1), A002415(n)), Range(2, oo))
	, (Eq(A005315(n), Sum(A060972(n, k), (k, 1, n))), Range(1, oo))
)

A006660 = settingsOEIS.lookupSymbolic['A006660']
A006662 = settingsOEIS.lookupSymbolic['A006662']
A259974 = settingsOEIS.lookupSymbolic['A259974']

boxOfRelationsBridges: tuple[形Relation, ...] = (
	(Eq(A259974(n, 1), A005316(n - 1)), Range(1, oo))
	, (Eq(A259974(n, 2), A006660(n)), Range(3, oo))
	, (Eq(A259974(n, 3), A006661(n)), Range(5, oo))
	, (Eq(A259974(n, 4), A006662(n)), Range(7, oo))
	, (Eq(A005316(n), Sum(A259974(n, k), (k, 1, ceiling(n / 2)))), Range(1, oo))
	, (Eq(A006661(n), A076876(n - 4) + A005316(n - 4)), Range(6, oo, 2))
)

A077056 = settingsOEIS.lookupSymbolic['A077056']
A380369 = settingsOEIS.lookupSymbolic['A380369']

boxOfRelationsHumps: tuple[形Relation, ...] = (
	(Eq(A380369(n, 0), 0), Range(1, oo))
	, (Eq(A380369(n, n), 1), Range(0, oo))
	, (Eq(A380369(n, n - 1), n * (n - 1)), Range(1, oo))
	, (Eq(A380369(n, 1), A006660(2 * n + 1)), Range(1, oo))
	, (Eq(A077054(n), Sum(A380369(n, k), (k, 0, n))), Range(0, oo))
	, (Eq(A077056(n), Sum(k * A380369(n, k), (k, 1, n))), Range(1, oo))
	, (Eq(A077056(n), 0), Range(0, 1))
)

boxOfRelations: tuple[形Relation, ...] = (*boxOfRelationsCatalan, *boxOfRelationsBallot, *boxOfRelationsCatafusenes, *boxOfRelationsMeanders, *boxOfRelationsStamps
	, *boxOfRelationsArches, *boxOfRelationsOverlaps, *boxOfRelationsWinding, *boxOfRelationsSystems, *boxOfRelationsBridges, *boxOfRelationsHumps)
