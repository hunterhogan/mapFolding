from __future__ import annotations

import pytest

parameterRowsGeneratingFunctionCoefficients = (
	pytest.param(0, (0, 1), (1, -1, -1), 0, id='fibonacci-constant')
	, pytest.param(1, (0, 1), (1, -1, -1), 1, id='fibonacci-first')
	, pytest.param(11, (0, 1), (1, -1, -1), 89, id='fibonacci-eleventh')
	, pytest.param(17, (0, 1), (1, -1, -1), 1597, id='fibonacci-seventeenth')
	, pytest.param(12, (5, 7), (1, 0, -1), 5, id='alternating-even')
	, pytest.param(13, (5, 7), (1, 0, -1), 7, id='alternating-odd')
	, pytest.param(17, (1,), (1, -2, 1), 18, id='repeated-pole')
	, pytest.param(17, (1,), (1, 1), -1, id='alternating-sign')
	, pytest.param(2, (0, 0, 5), (1, -1), 5, id='shifted-numerator')
	, pytest.param(11, (0, 0, 5), (1, -1), 5, id='shifted-geometric')
	, pytest.param(1, (5, 7), (1,), 7, id='polynomial-coefficient')
	, pytest.param(11, (5, 7), (1,), 0, id='past-polynomial-degree')
	, pytest.param(11, (0,), (1, -1), 0, id='zero-numerator')
	, pytest.param(11, (), (1, -1), 0, id='empty-numerator')
)

parameterRowsGeneratingFunctionCoefficientErrors = (
	pytest.param(-3, (5, 7), (1, -1), ValueError, id='negative-coefficient-index'),
)

parameterRowsGeneratingFunctionFractions = (
	pytest.param((1, 1), (), (2,), ((1,), (1, -1)), id='cancel-part-of-binomial')
	, pytest.param((1,), (1,), (2,), ((1,), (1, 1)), id='cancel-numerator-step')
	, pytest.param((5, 7), (2,), (1,), ((5, 12, 7), (1,)), id='polynomial')
	, pytest.param((2, 3), (), (2,), ((2, 3), (1, 0, -1)), id='already-coprime')
	, pytest.param((1, 1), (), (2, 2), ((1,), (1, -1, -1, 1)), id='repeated-denominator-factor')
	, pytest.param((1, 1, 1), (), (6,), ((1,), (1, -1, 0, 1, -1)), id='cancel-third-root-factor')
	, pytest.param((1, -1), (1,), (1, 2), ((1,), (1, 1)), id='repeated-numerator-factor')
	, pytest.param((5, 7), (4, 2), (2, 4), ((5, 7), (1,)), id='cancel-identical-steps')
	, pytest.param((0, 0, 5, 0, 0), (), (), ((0, 0, 5), (1,)), id='preserve-shift-trim-padding')
	, pytest.param((-5, -5), (), (2,), ((-5,), (1, -1)), id='preserve-sign')
	, pytest.param((0, 0), (3,), (2, 6), ((0,), (1,)), id='zero-polynomial')
	, pytest.param((), (), (2,), ((0,), (1,)), id='empty-numerator')
	, pytest.param((5, 7), (), (3, 1, 2), ((5, 7), (1, -1, -1, 0, 1, 1, -1)), id='unsorted-steps')
)

parameterRowsGeneratingFunctionSteps = (
	pytest.param((1, 1), (), (2,), ((1,), (), (1,)), id='cancel-part-of-binomial')
	, pytest.param((1,), (1,), (2,), ((1,), (1,), (2,)), id='cyclotomic-requires-numerator-step')
	, pytest.param((5, 7), (2,), (1,), ((5, 12, 7), (), ()), id='polynomial')
	, pytest.param((2, 3), (), (2,), ((2, 3), (), (2,)), id='already-coprime')
	, pytest.param((1, 1), (), (2, 2), ((1,), (), (1, 2)), id='repeated-denominator-factor')
	, pytest.param((1, 1, 1), (), (6,), ((1,), (3,), (1, 6)), id='cancel-third-root-factor')
	, pytest.param((1, -1), (1,), (1, 2), ((1,), (1,), (2,)), id='repeated-numerator-factor')
	, pytest.param((5, 7), (4, 2), (2, 4), ((5, 7), (), ()), id='cancel-identical-steps')
	, pytest.param((0, 0, 5, 0, 0), (), (), ((0, 0, 5), (), ()), id='preserve-shift-trim-padding')
	, pytest.param((-5, -5), (), (2,), ((-5,), (), (1,)), id='preserve-sign')
	, pytest.param((0, 0), (3,), (2, 6), ((0,), (), ()), id='zero-polynomial')
	, pytest.param((), (), (2,), ((0,), (), ()), id='empty-numerator')
	, pytest.param((5, 7), (), (3, 1, 2), ((5, 7), (), (1, 2, 3)), id='sort-steps')
	, pytest.param((4, 17, 17, -5, -22, -13, 6, 9, 1, -1), (), (1, 2, 2, 2, 3),
		((4, 17, 17, -5, -22, -13, 6, 9, 1, -1), (), (1, 2, 2, 2, 3)), id='A005315of3')
)

parameterRowsGeneratingFunctionErrors = (
	pytest.param((5, 7), (0,), (2,), ValueError, id='zero-numerator-step')
	, pytest.param((5, 7), (-3,), (2,), ValueError, id='negative-numerator-step')
	, pytest.param((5, 7), (), (0,), ValueError, id='zero-denominator-step')
	, pytest.param((5, 7), (), (-3,), ValueError, id='negative-denominator-step')
)
