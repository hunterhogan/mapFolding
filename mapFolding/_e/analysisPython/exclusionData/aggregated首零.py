from collections.abc import Callable
from fractions import Fraction

type FractionAddend = tuple[Callable[[int], int], Fraction, int]

dictionaryExclusions: dict[str, dict[str, list[FractionAddend]]] = {'首一': {},
'首一1': {},
'首一三': {},
'首一三1': {},
'首一二': {},
'首一二1': {},
'首一二三': {},
'首一二三1': {},
'首三': {},
'首三1': {},
'首二': {},
'首二1': {},
'首二三': {},
'首二三1': {},
'首零': {},
'首零1': {},
'首零一': {},
'首零一1': {},
'首零一三': {},
'首零一三1': {},
'首零一二': {},
'首零一二1': {},
'首零一二三': {},
'首零一二三1': {},
'首零三': {},
'首零三1': {},
'首零二': {},
'首零二1': {},
'首零二三': {},
'首零二三1': {}}

