# Static test data for mapFolding/_semiotics.py and mapFolding/_e/_semiotics.py
# This module contains only static data—dictionaries, lists, tuples, constants, and type aliases.

from typing import Final

#======== Expected values from mapFolding/_semiotics.py =======

expectedDecreasing: Final[int] = -1
expectedInclusive: Final[int] = 1

#======== Expected values from mapFolding/_e/_semiotics.py =======

# Dimension index constants (powers of 2)
expected零: Final[int] = 1
expected一: Final[int] = 2
expected二: Final[int] = 4
expected三: Final[int] = 8
expected四: Final[int] = 16
expected五: Final[int] = 32
expected六: Final[int] = 64
expected七: Final[int] = 128
expected八: Final[int] = 256
expected九: Final[int] = 512

# Origin constants
expectedLeafOrigin: Final[int] = 0
expectedPileOrigin: Final[int] = 0

# Function expected outputs for dimensionsTotal in range(5, 9)
# Each tuple is (dimensionsTotal, expectedResult)

expected首零: tuple[tuple[int, int], ...] = (
	(5, 16),
	(6, 32),
	(7, 64),
	(8, 128),
)

expected首零一: tuple[tuple[int, int], ...] = (
	(5, 24),
	(6, 48),
	(7, 96),
	(8, 192),
)

expected首零一二: tuple[tuple[int, int], ...] = (
	(5, 28),
	(6, 56),
	(7, 112),
	(8, 224),
)

expected首零二: tuple[tuple[int, int], ...] = (
	(5, 20),
	(6, 40),
	(7, 80),
	(8, 160),
)

expected首一: tuple[tuple[int, int], ...] = (
	(5, 8),
	(6, 16),
	(7, 32),
	(8, 64),
)

expected首一二: tuple[tuple[int, int], ...] = (
	(5, 12),
	(6, 24),
	(7, 48),
	(8, 96),
)

expected首二: tuple[tuple[int, int], ...] = (
	(5, 4),
	(6, 8),
	(7, 16),
	(8, 32),
)

expected首三: tuple[tuple[int, int], ...] = (
	(5, 2),
	(6, 4),
	(7, 8),
	(8, 16),
)

expected首零一二三: tuple[tuple[int, int], ...] = (
	(5, 30),
	(6, 60),
	(7, 120),
	(8, 240),
)

expected首零一三: tuple[tuple[int, int], ...] = (
	(5, 26),
	(6, 52),
	(7, 104),
	(8, 208),
)

expected首零二三: tuple[tuple[int, int], ...] = (
	(5, 22),
	(6, 44),
	(7, 88),
	(8, 176),
)

expected首零三: tuple[tuple[int, int], ...] = (
	(5, 18),
	(6, 36),
	(7, 72),
	(8, 144),
)

expected首一二三: tuple[tuple[int, int], ...] = (
	(5, 14),
	(6, 28),
	(7, 56),
	(8, 112),
)

expected首一三: tuple[tuple[int, int], ...] = (
	(5, 10),
	(6, 20),
	(7, 40),
	(8, 80),
)

expected首二三: tuple[tuple[int, int], ...] = (
	(5, 6),
	(6, 12),
	(7, 24),
	(8, 48),
)
