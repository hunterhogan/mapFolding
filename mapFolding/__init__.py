"""Prototype concept: Import priority levels. Larger priority values should be imported before smaller priority values.

`import` acts on the entire module, of course, so priority levels are per-module, not per-statement.

Value system based on ~40 years of varied experiences starting with line basic in 1985, but also used to order/prioritize/enumerate a dynamic set of objects/tasks.
- Only positive integer values.
- Do not reuse values.
- Avoid contiguous values.
- Prefer values that end with 0 because it makes it easier to skim.
- "Normal" range: 10-100; linear relationships between values.
- "Subnormal" range: After you set something at 10 and you discover something else that needs to be lower: 1-9.
- "I effed up" value: 0.
- "Supernormal" values: don't use these; The value means, "No, really, this is mega-important;" exponential growth.

I can already tell this implementation is, at best, untenably inefficient.
"""
from .theSSOT import * # Priority 10,000
from .beDRY import getLeavesTotal, getTaskDivisions, makeConnectionGraph, outfitFoldings, setCPUlimit # Priority 1,000
from .beDRY import parseListDimensions, validateListDimensions
from .lola import countFolds # Priority 70. NOTE `countFolds` is the point of the package. Two things should be very stable: 1) the name of the function and 2) the first parameter will accept a `list` of integers representing the dimensions of a map.
from .oeis import oeisIDfor_n, getOEISids, clearOEIScache # Priority 30

__all__ = [
    'clearOEIScache',
    'countFolds',
    'getOEISids',
    'oeisIDfor_n',
]
