# DEVELOPMENT
# ruff: file-ignore[implicit-namespace-package, undocumented-public-module, import-private-name, undocumented-public-function, batched-without-explicit-strict, print]
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
from __future__ import annotations

from fractions import Fraction
from functools import reduce
from itertools import batched, starmap
from math import comb, lcm
from research.matrixMeanders.formulasTriangle._A005315 import A005315of10
from research.matrixMeanders.infoBooth import pathFilenameFormulaA005315
from sympy import Matrix
import json

function_name = 'A005315of10'
next_function_name = '_crunchCoefficients'
encoded_input_start = 20
multiplicities = ((2, 18), (3, 8), (4, 7), (5, 6), (6, 5), (7, 4), (8, 7), (9, 2), (10, 6), (12, 5), (14, 4), (16, 3), (18, 2), (20, 1))
labels: list[tuple[int, int, int] | tuple[str, int, int]] = [('constant', 0, 0)] + [(period, shift, degree) for period, maximum_degree in multiplicities for degree in range(1, maximum_degree + 1) for shift in range(period)]

def basis(offset: int, label: tuple[int, int, int] | tuple[str, int, int]) -> int:
    period, shift, degree = label
    return 1 if period == 'constant' else comb((offset + shift) // int(period), degree)

matrix: Matrix = Matrix([[basis(offset, label) for label in labels] for offset in range(410)])
values: Matrix = Matrix([A005315of10(offset + encoded_input_start) for offset in range(410)])
solution, parameters = matrix.gauss_jordan_solve(values)
solution = solution.subs(dict.fromkeys(parameters, 0))
terms = [(label, Fraction(coefficient)) for label, coefficient in zip(labels, solution, strict=True) if coefficient]
denominator = reduce(lcm, (coefficient.denominator for _, coefficient in terms), 1)

def candidate(x: int) -> Fraction:
    offset: int = x - encoded_input_start
    return Fraction(sum(coefficient * basis(offset, label) for label, coefficient in terms))

for x in range(encoded_input_start, 601):
    value = candidate(x)
    if value.denominator != 1 or value.numerator != A005315of10(x):
        raise AssertionError((x, value, A005315of10(x)))

scaled_terms: list[tuple[tuple[int, int, int] | tuple[str, int, int], int]] = [(label, coefficient.numerator * (denominator // coefficient.denominator)) for label, coefficient in terms]
constant: int = next(coefficient for (label, coefficient) in scaled_terms if label[0] == 'constant')
nonconstant_terms: list[tuple[tuple[int, int, int] | tuple[str, int, int], int]] = [(label, coefficient) for label, coefficient in scaled_terms if label[0] != 'constant']

def render_term(label: tuple[int, int, int] | tuple[str, int, int], coefficient: int) -> str:
    period, shift, degree = label
    argument = f'x // {period}' if shift == 0 else f'(x + {shift}) // {period}'
    operator = '+' if coefficient > 0 else '-'
    return f'{operator} {abs(coefficient)} * comb({argument}, {degree})'

lines = [
    f'def {function_name}(x: int) -> int:',
    f'\tx -= {encoded_input_start}',
    '\treturn (',
    f'\t\t{constant}',
]
lines.extend('\t\t' + ' '.join(starmap(render_term, pair)) for pair in batched(nonconstant_terms, 2))
lines.extend((f'\t) // {denominator}', '""'))
new_block = '\n'.join(lines)

source = pathFilenameFormulaA005315.read_text(encoding='utf-8')
start = source.index(f'def {function_name}')
end = source.index(f'\ndef {next_function_name}', start)
old_block = source[start:end]
print(json.dumps({
    'old': old_block,
    'new': new_block,
    'summary': f'{len(nonconstant_terms)} binomial terms, denominator {denominator}, matched x={encoded_input_start}..600',
}))
