# ruff: file-ignore[implicit-namespace-package, undocumented-public-module, undocumented-public-function, ambiguous-variable-name, print]
# pyright: reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false, reportMissingParameterType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportArgumentType=false
# ruff: file-ignore[missing-return-type-private-function, missing-type-function-argument, suspicious-eval-usage, missing-return-type-undocumented-public-function]
# ty: ignore[unresolved-attribute, invalid-argument-type]
# ruff: file-ignore[assert]
from __future__ import annotations

from hunterMakesPy import raiseIfNone
from research.matrixMeanders.infoBooth import pathFilenameArchiveA005315Binomial
from typing import Any
import ast
import sympy

src: str = pathFilenameArchiveA005315Binomial.read_text(encoding='utf-8')
tr: ast.Module = ast.parse(src)
z, u = sympy.symbols('z u')

def get_info(n: int) -> Any:
	fn: ast.FunctionDef = next(node for node in tr.body if isinstance(node, ast.FunctionDef) and node.name == f'A005315of{n}')
	ass: ast.AnnAssign = next(s for s in fn.body if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name) and s.target.id == 'numerator')
	num: int = ast.literal_eval(raiseIfNone(ass.value))
	ret: ast.expr = raiseIfNone(next(s for s in fn.body if isinstance(s, ast.Return)).value)
	call = ret.right

	def ev(e):
		return eval(compile(ast.Expression(e), '', 'eval'), {})
	sn = ev(call.args[2])
	sd = ev(call.args[3])
	P = 2 * sum(c * z**i for i, c in enumerate(num))
	for d in sn:
		P *= 1 - z**d
	Q = 1
	for d in sd:
		Q *= 1 - z**d
	return sympy.cancel(P / Q)

def zTOu(expr):
	num, den = map(sympy.expand, sympy.fraction(sympy.cancel(expr)))

	def conv(p):
		P = sympy.Poly(p, z)
		out = 0
		for (e,), c in P.terms():
			assert e % 2 == 0, (e, c)
			out += c * u**(e // 2)
		return sympy.expand(out)
	return sympy.cancel(conv(num) / conv(den))

for n in range(4, 11):
	G = get_info(n)
	E = zTOu(sympy.cancel((G + G.subs(z, -z)) / 2))
	O = zTOu(sympy.cancel((G - G.subs(z, -z)) / (2 * z)))
	print('\nn=', n)
	for label, H in [('E', E), ('O', O)]:
		num, den = sympy.fraction(H)
		print(label, 'degnum', sympy.degree(num, u), 'degden', sympy.degree(den, u))
		print('den=', sympy.factor(den))
		if n <= 5:
			print('num factor=', sympy.factor(num))

n = 4
# E degnum 13 degden 14
den = (u - 1)**7 * (u + 1) * (u**2 + 1) * (u**2 + u + 1)**2
num_factor = 2 * (2 * u**13 - 6 * u**12 + 68 * u**11 - 81 * u**10 - 16 * u**9 - 132 * u**8 + 60 * u**7 + 23 * u**6 - 136 * u**5 - 281 * u**4 - 572 * u**3 - 380 * u**2 - 292 * u - 21)
# O degnum 13 degden 14
den = (u - 1)**7 * (u + 1) * (u**2 + 1) * (u**2 + u + 1)**2
num_factor = 2 * (4 * u**13 - 19 * u**12 + 62 * u**11 - 29 * u**10 - 24 * u**9 - 114 * u**8 - 26 * u**7 + 68 * u**6 - 58 * u**5 - 179 * u**4 - 474 * u**3 - 482 * u**2 - 362 * u - 131)

n = 5
# E degnum 24 degden 25
den = (u - 1)**9 * (u + 1)**2 * (u**2 + 1)**2 * (u**2 + u + 1)**3 * (u**4 + u**3 + u**2 + u + 1)
num_factor = 2 * (10 * u**24 - 48 * u**23 + 40 * u**22 - 297 * u**21 + 21 * u**20 + 64 * u**19 + 966 * u**18 + 728 * u**17 - 672 * u**16 - 4224 * u**15 - 10595 * u**14 - 19067 * u**13 - 31632 * u**12 - 47035 * u**11 - 66324 * u**10 - 83912 * u**9 - 96420 * u**8 - 97764 * u**7 - 87087 * u**6 - 67031 * u**5 - 43266 * u**4 - 23129 * u**3 - 9056 * u**2 - 2591 * u - 131)
# O degnum 24 degden 25
den = (u - 1)**9 * (u + 1)**2 * (u**2 + 1)**2 * (u**2 + u + 1)**3 * (u**4 + u**3 + u**2 + u + 1)
num_factor = 2 * (21 * u**24 - 76 * u**23 + 91 * u**22 - 232 * u**21 - 27 * u**20 - 134 * u**19 + 617 * u**18 + 902 * u**17 + 211 * u**16 - 2068 * u**15 - 7277 * u**14 - 14275 * u**13 - 25096 * u**12 - 38673 * u**11 - 56657 * u**10 - 75283 * u**9 - 91302 * u**8 - 98484 * u**7 - 93836 * u**6 - 78025 * u**5 - 55021 * u**4 - 32645 * u**3 - 15102 * u**2 - 5167 * u - 914)

n = 6
# E degnum 37 degden 38
den = (u - 1)**11 * (u + 1)**3 * (u**2 + 1)**3 * (u**2 - u + 1) * (u**2 + u + 1)**4 * (u**4 + u**3 + u**2 + u + 1)**2
# O degnum 37 degden 38
den = (u - 1)**11 * (u + 1)**3 * (u**2 + 1)**3 * (u**2 - u + 1) * (u**2 + u + 1)**4 * (u**4 + u**3 + u**2 + u + 1)**2

n = 7
# E degnum 56 degden 57
den = (u - 1)**13 * (u + 1)**4 * (u**2 + 1)**4 * (u**2 - u + 1)**2 * (u**2 + u + 1)**5 * (u**4 + u**3 + u**2 + u + 1)**3 * (u**6 + u**5 + u**4 + u**3 + u**2 + u + 1)
# O degnum 56 degden 57
den = (u - 1)**13 * (u + 1)**4 * (u**2 + 1)**4 * (u**2 - u + 1)**2 * (u**2 + u + 1)**5 * (u**4 + u**3 + u**2 + u + 1)**3 * (u**6 + u**5 + u**4 + u**3 + u**2 + u + 1)

n = 8
# E degnum 79 degden 80
den = (u - 1)**15 * (u + 1)**5 * (u**2 + 1)**5 * (u**4 + 1) * (u**2 - u + 1)**3 * (u**2 + u + 1)**6 * (u**4 + u**3 + u**2 + u + 1)**4 * (u**6 + u**5 + u**4 + u**3 + u**2 + u + 1)**2
# O degnum 79 degden 80
den = (u - 1)**15 * (u + 1)**5 * (u**2 + 1)**5 * (u**4 + 1) * (u**2 - u + 1)**3 * (u**2 + u + 1)**6 * (u**4 + u**3 + u**2 + u + 1)**4 * (u**6 + u**5 + u**4 + u**3 + u**2 + u + 1)**2

n = 9
# E degnum 108 degden 109
den = (u - 1)**17 * (u + 1)**6 * (u**2 + 1)**6 * (u**4 + 1)**2 * (u**2 - u + 1)**4 * (u**2 + u + 1)**7 * (u**6 + u**3 + 1) * (u**4 + u**3 + u**2 + u + 1)**5 * (u**6 + u**5 + u**4 + u**3 + u**2 + u + 1)**3
# O degnum 108 degden 109
den = (u - 1)**17 * (u + 1)**6 * (u**2 + 1)**6 * (u**4 + 1)**2 * (u**2 - u + 1)**4 * (u**2 + u + 1)**7 * (u**6 + u**3 + 1) * (u**4 + u**3 + u**2 + u + 1)**5 * (u**6 + u**5 + u**4 + u**3 + u**2 + u + 1)**3

n = 10
# E degnum 141 degden 142
den = (u - 1)**19 * (u + 1)**7 * (u**2 + 1)**7 * (u**4 + 1)**3 * (u**2 - u + 1)**5 * (u**2 + u + 1)**8 * (u**6 + u**3 + 1)**2 * (u**4 - u**3 + u**2 - u + 1) * (u**4 + u**3 + u**2 + u + 1)**6 * (u**6 + u**5 + u**4 + u**3 + u**2 + u + 1)**4
# O degnum 141 degden 142
den = (u - 1)**19 * (u + 1)**7 * (u**2 + 1)**7 * (u**4 + 1)**3 * (u**2 - u + 1)**5 * (u**2 + u + 1)**8 * (u**6 + u**3 + 1)**2 * (u**4 - u**3 + u**2 - u + 1) * (u**4 + u**3 + u**2 + u + 1)**6 * (u**6 + u**5 + u**4 + u**3 + u**2 + u + 1)**4
