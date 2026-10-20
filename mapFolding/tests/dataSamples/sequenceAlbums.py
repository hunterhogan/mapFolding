from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from _pytest.mark.structures import ParameterSet
	from mapFolding.theTypes import Folding
albumStamps4: list[Folding] = [
	(1, 4, 3, 2),
	(4, 3, 2, 1),
	(3, 2, 1, 4),
	(2, 1, 4, 3),
	(1, 3, 4, 2),
	(3, 4, 2, 1),
	(4, 2, 1, 3),
	(2, 1, 3, 4),
	(1, 2, 4, 3),
	(2, 4, 3, 1),
	(4, 3, 1, 2),
	(3, 1, 2, 4),
	(1, 2, 3, 4),
	(2, 3, 4, 1),
	(3, 4, 1, 2),
	(4, 1, 2, 3),
]

albumStampsBlank4: list[Folding] = [(1, 4, 3, 2), (1, 2, 3, 4), (2, 1, 4, 3), (1, 3, 4, 2), (1, 2, 4, 3)]

albumSemi4: list[Folding] = [(1, 4, 3, 2), (1, 3, 4, 2), (1, 2, 4, 3), (1, 2, 3, 4)]

albumSemi6: list[Folding] = [
	(1, 6, 5, 4, 3, 2),
	(1, 5, 4, 3, 6, 2),
	(1, 5, 6, 4, 3, 2),
	(1, 4, 3, 2, 6, 5),
	(1, 4, 3, 2, 5, 6),
	(1, 4, 6, 5, 3, 2),
	(1, 4, 5, 6, 3, 2),
	(1, 3, 6, 5, 4, 2),
	(1, 3, 5, 6, 4, 2),
	(1, 3, 4, 6, 5, 2),
	(1, 6, 3, 4, 5, 2),
	(1, 3, 4, 5, 6, 2),
	(1, 2, 6, 5, 4, 3),
	(1, 2, 5, 4, 3, 6),
	(1, 2, 5, 6, 4, 3),
	(1, 2, 4, 6, 5, 3),
	(1, 2, 4, 5, 6, 3),
	(1, 2, 3, 6, 5, 4),
	(1, 2, 3, 5, 6, 4),
	(1, 6, 5, 2, 3, 4),
	(1, 5, 6, 2, 3, 4),
	(1, 2, 3, 4, 6, 5),
	(1, 2, 6, 3, 4, 5),
	(1, 2, 3, 4, 5, 6),
]

albumSemi7: list[Folding] = [
	(1, 7, 6, 5, 4, 3, 2),
	(1, 6, 5, 4, 3, 2, 7),
	(1, 6, 5, 4, 7, 3, 2),
	(1, 6, 7, 5, 4, 3, 2),
	(1, 5, 4, 3, 7, 6, 2),
	(1, 5, 4, 3, 6, 7, 2),
	(1, 5, 7, 6, 4, 3, 2),
	(1, 5, 6, 7, 4, 3, 2),
	(1, 4, 3, 2, 7, 6, 5),
	(1, 4, 7, 3, 2, 6, 5),
	(1, 4, 3, 2, 6, 7, 5),
	(1, 4, 3, 2, 5, 7, 6),
	(1, 7, 4, 3, 2, 5, 6),
	(1, 4, 3, 2, 5, 6, 7),
	(1, 4, 7, 6, 5, 3, 2),
	(1, 4, 6, 7, 5, 3, 2),
	(1, 4, 5, 7, 6, 3, 2),
	(1, 7, 4, 5, 6, 3, 2),
	(1, 4, 5, 6, 3, 2, 7),
	(1, 4, 5, 6, 7, 3, 2),
	(1, 3, 7, 6, 5, 4, 2),
	(1, 3, 6, 5, 4, 7, 2),
	(1, 3, 6, 7, 5, 4, 2),
	(1, 3, 5, 7, 6, 4, 2),
	(1, 3, 5, 6, 7, 4, 2),
	(1, 3, 4, 7, 6, 5, 2),
	(1, 3, 4, 6, 7, 5, 2),
	(1, 7, 6, 3, 4, 5, 2),
	(1, 6, 3, 4, 5, 2, 7),
	(1, 6, 7, 3, 4, 5, 2),
	(1, 3, 4, 5, 7, 6, 2),
	(1, 3, 7, 4, 5, 6, 2),
	(1, 3, 4, 5, 6, 7, 2),
	(1, 2, 7, 6, 5, 4, 3),
	(1, 2, 6, 5, 4, 7, 3),
	(1, 2, 6, 7, 5, 4, 3),
	(1, 2, 5, 4, 3, 7, 6),
	(1, 7, 2, 5, 4, 3, 6),
	(1, 2, 5, 4, 3, 6, 7),
	(1, 2, 5, 7, 6, 4, 3),
	(1, 2, 5, 6, 7, 4, 3),
	(1, 2, 4, 7, 6, 5, 3),
	(1, 2, 4, 6, 7, 5, 3),
	(1, 2, 4, 5, 7, 6, 3),
	(1, 2, 7, 4, 5, 6, 3),
	(1, 2, 4, 5, 6, 7, 3),
	(1, 2, 3, 7, 6, 5, 4),
	(1, 7, 2, 3, 6, 5, 4),
	(1, 2, 3, 6, 5, 4, 7),
	(1, 2, 3, 6, 7, 5, 4),
	(1, 2, 3, 5, 7, 6, 4),
	(1, 2, 3, 5, 6, 7, 4),
	(1, 7, 6, 5, 2, 3, 4),
	(1, 6, 5, 2, 3, 4, 7),
	(1, 6, 7, 5, 2, 3, 4),
	(1, 5, 7, 6, 2, 3, 4),
	(1, 5, 6, 2, 3, 7, 4),
	(1, 5, 6, 7, 2, 3, 4),
	(1, 2, 3, 4, 7, 6, 5),
	(1, 2, 3, 4, 6, 7, 5),
	(1, 2, 7, 6, 3, 4, 5),
	(1, 2, 6, 7, 3, 4, 5),
	(1, 2, 3, 4, 5, 7, 6),
	(1, 2, 3, 7, 4, 5, 6),
	(1, 7, 2, 3, 4, 5, 6),
	(1, 2, 3, 4, 5, 6, 7),
]

albumSemiSymmetric7: list[Folding] = [
	(7, 6, 5, 4, 3, 2, 1),
	(6, 5, 4, 3, 2, 7, 1),
	(6, 5, 4, 7, 3, 2, 1),
	(6, 7, 5, 4, 3, 2, 1),
	(5, 4, 3, 2, 1, 7, 6),
	(5, 4, 3, 2, 1, 6, 7),
	(5, 4, 3, 7, 6, 2, 1),
	(5, 4, 3, 6, 7, 2, 1),
	(5, 7, 6, 4, 3, 2, 1),
	(5, 6, 7, 4, 3, 2, 1),
	(4, 3, 2, 7, 6, 5, 1),
	(4, 7, 3, 2, 6, 5, 1),
	(4, 3, 2, 6, 7, 5, 1),
	(4, 3, 2, 5, 7, 6, 1),
	(7, 4, 3, 2, 5, 6, 1),
	(4, 3, 2, 5, 6, 7, 1),
	(4, 7, 6, 5, 3, 2, 1),
	(4, 6, 7, 5, 3, 2, 1),
	(4, 5, 7, 6, 3, 2, 1),
	(7, 4, 5, 6, 3, 2, 1),
	(4, 5, 6, 3, 2, 7, 1),
	(4, 5, 6, 7, 3, 2, 1),
	(3, 2, 1, 7, 6, 5, 4),
	(3, 2, 1, 6, 5, 4, 7),
	(3, 2, 1, 6, 7, 5, 4),
	(3, 7, 6, 2, 1, 5, 4),
	(3, 6, 7, 2, 1, 5, 4),
	(3, 2, 1, 5, 7, 6, 4),
	(3, 2, 1, 5, 6, 7, 4),
	(3, 2, 1, 4, 7, 6, 5),
	(3, 2, 1, 4, 6, 7, 5),
	(7, 6, 3, 2, 1, 4, 5),
	(6, 3, 2, 7, 1, 4, 5),
	(6, 7, 3, 2, 1, 4, 5),
	(3, 2, 1, 4, 5, 7, 6),
	(3, 2, 1, 7, 4, 5, 6),
	(3, 2, 1, 4, 5, 6, 7),
	(3, 7, 6, 5, 4, 2, 1),
	(3, 6, 5, 4, 7, 2, 1),
	(3, 6, 7, 5, 4, 2, 1),
	(3, 5, 7, 6, 4, 2, 1),
	(3, 5, 6, 7, 4, 2, 1),
	(3, 4, 7, 6, 5, 2, 1),
	(3, 4, 6, 7, 5, 2, 1),
	(7, 6, 3, 4, 5, 2, 1),
	(6, 3, 4, 5, 2, 7, 1),
	(6, 7, 3, 4, 5, 2, 1),
	(3, 4, 5, 2, 1, 7, 6),
	(3, 4, 5, 2, 1, 6, 7),
	(3, 4, 5, 7, 6, 2, 1),
	(3, 7, 4, 5, 6, 2, 1),
	(3, 4, 5, 6, 7, 2, 1),
	(2, 7, 6, 5, 4, 3, 1),
	(2, 6, 5, 4, 7, 3, 1),
	(2, 6, 7, 5, 4, 3, 1),
	(2, 5, 4, 3, 7, 6, 1),
	(7, 2, 5, 4, 3, 6, 1),
	(2, 5, 4, 3, 6, 7, 1),
	(2, 5, 7, 6, 4, 3, 1),
	(2, 5, 6, 7, 4, 3, 1),
	(2, 4, 7, 6, 5, 3, 1),
	(2, 4, 6, 7, 5, 3, 1),
	(2, 4, 5, 7, 6, 3, 1),
	(2, 7, 4, 5, 6, 3, 1),
	(2, 4, 5, 6, 7, 3, 1),
	(2, 3, 7, 6, 5, 4, 1),
	(7, 2, 3, 6, 5, 4, 1),
	(2, 3, 6, 5, 4, 7, 1),
	(2, 3, 6, 7, 5, 4, 1),
	(2, 3, 5, 7, 6, 4, 1),
	(2, 3, 5, 6, 7, 4, 1),
	(7, 6, 5, 2, 3, 4, 1),
	(6, 5, 2, 3, 4, 7, 1),
	(6, 7, 5, 2, 3, 4, 1),
	(5, 2, 3, 4, 1, 7, 6),
	(5, 2, 3, 4, 1, 6, 7),
	(5, 7, 6, 2, 3, 4, 1),
	(5, 6, 2, 3, 7, 4, 1),
	(5, 6, 7, 2, 3, 4, 1),
	(2, 3, 4, 7, 6, 5, 1),
	(2, 3, 4, 6, 7, 5, 1),
	(2, 7, 6, 3, 4, 5, 1),
	(2, 6, 7, 3, 4, 5, 1),
	(2, 3, 4, 5, 7, 6, 1),
	(2, 3, 7, 4, 5, 6, 1),
	(7, 2, 3, 4, 5, 6, 1),
	(2, 3, 4, 5, 6, 7, 1),
]

albumSemiSymmetric4: list[Folding] = [(4, 3, 2, 1), (3, 2, 1, 4), (3, 4, 2, 1), (2, 4, 3, 1), (2, 3, 4, 1)]
albumMeanders3: list[Folding] = [(1, 2, 3), (3, 2, 1)]
albumMeanders4: list[Folding] = [(1, 2, 3, 4), (1, 4, 3, 2), (3, 2, 1, 4)]
albumMeandersClasses4: list[Folding] = [(1, 2, 3, 4, 5), (1, 2, 5, 4, 3), (1, 4, 3, 2, 5)]

parametersEnumeration: tuple[ParameterSet, ...] = (
	pytest.param(4, 'stamps', None, None, 'A000136', id='default-stamps')
	, pytest.param(4, 'stamps blank', None, None, 'A001011', id='default-stamps-blank')
	, pytest.param(4, 'semi', None, None, 'A000682', id='default-semi')
	, pytest.param(4, 'semi-meanders symmetric', None, None, 'A000560', id='default-semi-symmetric')
	, pytest.param(4, 'meanders', None, None, 'A005316', id='default-meanders')
	, pytest.param(4, 'meanders symmetric', None, None, 'A077055', id='default-meanders-symmetric')
	, pytest.param(4, None, None, None, 'A000682', id='default-kind')
	, pytest.param(6, 'semi', None, None, 'A000682', id='default-semi-six')
	, pytest.param(7, 'semi', None, None, 'A000682', id='default-semi-frontier')
	, pytest.param(7, 'semi-meanders symmetric', None, None, 'A000560', id='default-symmetric-frontier')
	, pytest.param(3, 'meanders', None, None, 'A005316', id='default-meanders-odd')
	, pytest.param(1, 'stamps', None, None, 'A000136', id='default-stamps-offset')
	, pytest.param(1, 'stamps blank', None, None, 'A001011', id='default-blank-offset')
	, pytest.param(1, None, None, None, 'A000682', id='default-semi-offset')
	, pytest.param(2, 'semi', None, None, 'A000682', id='default-semi-two')
	, pytest.param(2, 'semi-meanders symmetric', None, None, 'A000560', id='default-symmetric-offset')
	, pytest.param(0, 'meanders', None, None, 'A005316', id='default-meanders-offset')
	, pytest.param(1, 'meanders', None, None, 'A005316', id='default-meanders-one')
	, pytest.param(0, 'meanders symmetric', None, None, 'A077055', id='default-meanders-symmetric-offset')
	, pytest.param(1, 'meanders symmetric', None, None, 'A077055', id='default-meanders-symmetric-one')
	, pytest.param(4, 'stamps', 'permutations', None, 'A000136', id='permutations-stamps')
	, pytest.param(4, 'stamps blank', 'permutations', None, 'A001011', id='permutations-stamps-blank')
	, pytest.param(4, 'semi', 'permutations', None, 'A000682', id='permutations-semi')
	, pytest.param(4, 'semi-meanders symmetric', 'permutations', None, 'A000560', id='permutations-semi-symmetric')
	, pytest.param(4, 'meanders', 'permutations', None, 'A005316', id='permutations-meanders')
	, pytest.param(4, 'meanders symmetric', 'permutations', None, 'A077055', id='permutations-meanders-symmetric')
	, pytest.param(4, None, 'permutations', None, 'A000682', id='permutations-default-kind')
	, pytest.param(6, 'semi', 'permutations', None, 'A000682', id='permutations-semi-six')
	, pytest.param(7, 'semi', 'permutations', None, 'A000682', id='permutations-semi-frontier')
	, pytest.param(7, 'semi-meanders symmetric', 'permutations', None, 'A000560', id='permutations-symmetric-frontier')
	, pytest.param(3, 'meanders', 'permutations', None, 'A005316', id='permutations-meanders-odd')
	, pytest.param(1, 'stamps', 'permutations', None, 'A000136', id='permutations-stamps-offset')
	, pytest.param(1, 'stamps blank', 'permutations', None, 'A001011', id='permutations-blank-offset')
	, pytest.param(1, None, 'permutations', None, 'A000682', id='permutations-semi-offset')
	, pytest.param(2, 'semi', 'permutations', None, 'A000682', id='permutations-semi-two')
	, pytest.param(2, 'semi-meanders symmetric', 'permutations', None, 'A000560', id='permutations-symmetric-offset')
	, pytest.param(0, 'meanders', 'permutations', None, 'A005316', id='permutations-meanders-offset')
	, pytest.param(1, 'meanders', 'permutations', None, 'A005316', id='permutations-meanders-one')
	, pytest.param(0, 'meanders symmetric', 'permutations', None, 'A077055', id='permutations-meanders-symmetric-offset')
	, pytest.param(1, 'meanders symmetric', 'permutations', None, 'A077055', id='permutations-meanders-symmetric-one')
	, pytest.param(4, 'semi', 'bilateral', None, 'A000682', id='bilateral-semi')
	, pytest.param(4, 'semi-meanders symmetric', 'bilateral', None, 'A000560', id='bilateral-semi-symmetric')
	, pytest.param(7, 'semi', 'bilateral', None, 'A000682', id='bilateral-semi-frontier')
	, pytest.param(7, 'semi-meanders symmetric', 'bilateral', None, 'A000560', id='bilateral-symmetric-frontier')
	, pytest.param(1, 'semi', 'bilateral', None, 'A000682', id='bilateral-semi-offset')
	, pytest.param(2, 'semi', 'bilateral', None, 'A000682', id='bilateral-semi-two')
	, pytest.param(2, 'semi-meanders symmetric', 'bilateral', None, 'A000560', id='bilateral-symmetric-offset')
	, pytest.param(4, 'semi', 'bilateralConcurrent', 1, 'A000682', id='concurrent-one-semi')
	, pytest.param(4, 'semi-meanders symmetric', 'bilateralConcurrent', 1, 'A000560', id='concurrent-one-semi-symmetric')
	, pytest.param(7, 'semi', 'bilateralConcurrent', 1, 'A000682', id='concurrent-one-semi-frontier')
	, pytest.param(7, 'semi-meanders symmetric', 'bilateralConcurrent', 1, 'A000560', id='concurrent-one-symmetric-frontier')
	, pytest.param(1, 'semi', 'bilateralConcurrent', 1, 'A000682', id='concurrent-one-semi-offset')
	, pytest.param(2, 'semi', 'bilateralConcurrent', 1, 'A000682', id='concurrent-one-semi-two')
	, pytest.param(2, 'semi-meanders symmetric', 'bilateralConcurrent', 1, 'A000560', id='concurrent-one-symmetric-offset')
	, pytest.param(4, 'semi', 'bilateralConcurrent', 2, 'A000682', id='concurrent-two-semi')
	, pytest.param(4, 'semi-meanders symmetric', 'bilateralConcurrent', 2, 'A000560', id='concurrent-two-semi-symmetric')
	, pytest.param(7, 'semi', 'bilateralConcurrent', 2, 'A000682', id='concurrent-two-semi-frontier')
	, pytest.param(7, 'semi-meanders symmetric', 'bilateralConcurrent', 2, 'A000560', id='concurrent-two-symmetric-frontier')
	, pytest.param(1, 'semi', 'bilateralConcurrent', 2, 'A000682', id='concurrent-two-semi-offset')
	, pytest.param(2, 'semi', 'bilateralConcurrent', 2, 'A000682', id='concurrent-two-semi-two')
	, pytest.param(2, 'semi-meanders symmetric', 'bilateralConcurrent', 2, 'A000560', id='concurrent-two-symmetric-offset')
)

parametersErrors: tuple[ParameterSet, ...] = (
	pytest.param(6, 'unknown', 'permutations', None, None, KeyError, id='unknown-kind')
	, pytest.param(4, '', 'permutations', None, None, KeyError, id='empty-kind')
	, pytest.param(4, 'semi', 'permutations', Path('existing.album'), None, FileExistsError, id='existing-album')
	, pytest.param(4, 'semi', 'bilateral', Path('existing.album'), None, FileExistsError, id='existing-album-bilateral')
	, pytest.param(4, 'semi', 'bilateralConcurrent', Path('existing.album'), 2, FileExistsError, id='existing-album-concurrent')
	, pytest.param(4, 'stamps', 'bilateral', None, None, KeyError, id='bilateral-stamps-unimplemented')
	, pytest.param(4, 'stamps blank', 'bilateral', None, None, KeyError, id='bilateral-blank-unimplemented')
	, pytest.param(4, 'meanders', 'bilateral', None, None, KeyError, id='bilateral-meanders-unimplemented')
	, pytest.param(4, 'meanders symmetric', 'bilateral', None, None, KeyError, id='bilateral-symmetric-meanders-unimplemented')
	, pytest.param(4, 'stamps', 'bilateralConcurrent', None, 2, KeyError, id='concurrent-stamps-unimplemented')
	, pytest.param(4, 'stamps blank', 'bilateralConcurrent', None, 2, KeyError, id='concurrent-blank-unimplemented')
	, pytest.param(4, 'meanders', 'bilateralConcurrent', None, 2, KeyError, id='concurrent-meanders-unimplemented')
	, pytest.param(4, 'meanders symmetric', 'bilateralConcurrent', None, 2, KeyError, id='concurrent-symmetric-meanders-unimplemented')
)
