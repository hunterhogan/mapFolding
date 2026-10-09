# ruff: file-ignore[undocumented-public-function,ambiguous-variable-name,undocumented-public-module]
from __future__ import annotations

from collections import defaultdict
from mapFolding.oeis import printEasyRunBenchmark, printEasyRunHeader
from tqdm import tqdm
import time

State = tuple[int, int, int, tuple[int, ...]]

def successors(x: int, state: State) -> list[State | None]:
	dl, dr, a, old = state
	b: int = len(old) - a
	out: list[State | None] = []

	if a <= x - dl and b <= x - dr:
		if a < x - dl:
			if b < x - dr:
				p: dict[int, int] = dict(enumerate(old))
				l: int = a + b
				r: int = a + b + 1
				p[l] = r
				p[r] = l
				record(out, dl, dr, p, (*range(a), l, *range(a, a + b), r), a + 1)

			if b:
				p = dict(enumerate(old))
				l = a + b
				partner: int = p.pop(a + b - 1)
				p[partner] = l
				p[l] = partner
				record(out, dl, dr + 1, p, (*range(a), l, *range(a, a + b - 1)), a + 1)

		if a:
			if b < x - dr:
				p = dict(enumerate(old))
				r = a + b + 1
				partner = p.pop(a - 1)
				p[partner] = r
				p[r] = partner
				record(out, dl + 1, dr, p, (*range(a - 1), *range(a, a + b), r), a - 1)

			if b:
				if old[a - 1] != a + b - 1:
					p = dict(enumerate(old))
					lp: int = p.pop(a - 1)
					rp: int = p.pop(a + b - 1)
					p[lp] = rp
					p[rp] = lp
					record(out, dl + 1, dr + 1, p, (*range(a - 1), *range(a, a + b - 1)), a - 1)
				elif a == b == 1 and dl == dr == x - 1:
					out.append(None)
	return out

def record(out: list[State | None], nl: int, nr: int, p: dict[int, int], order: tuple[int, ...], len_left: int) -> None:
    index: dict[int, int] = {v: i for i, v in enumerate(order)}
    out.append((nl, nr, len_left, tuple(index[p[v]] for v in order)))

def count(x: int) -> int:
	nxt: dict[State, int] = {(0, 0, 0, ()): 1}
	for _step in tqdm(range(2 * x - 2)):
		layer: dict[State, int] = nxt
		nxt = defaultdict(int)
		for state, weight in layer.items():  # Could be concurrent.
			for successor in filter(None, successors(x, state)):
				nxt[successor] += weight
	total: int = 0
	for state, weight in nxt.items():
		total += weight * len(successors(x, state))
	return total

if __name__ == '__main__':
	oeisID = 'A005315'
	printEasyRunHeader(oeisID, 'count')
	timeStart = time.perf_counter()
	for x in range(18, 19):
		printEasyRunBenchmark(oeisID, x, count(x), timeStart)
