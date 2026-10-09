# ruff: file-ignore[print,undocumented-public-function,ambiguous-variable-name,loop-iterator-mutation]
"""Exact fixed-deficit arch transfer, using safe premature rainbow closure.

No conjectured coefficients enter the transfer.  See note.tex for its proof.
State: (left closes used, right closes used, left stack size, pairing).
The pairing lists partners on left bottom-to-top then right bottom-to-top.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import argparse
import csv
import json
import os
import pathlib
import time

def successors(j: int, state: tuple[int, int, int, tuple[int, ...]]) -> list[tuple[tuple[int, int], None] | tuple[tuple[int, int], tuple[int, int, int, tuple[int, ...]]]]:
	dl, dr, a, old = state
	b: int = len(old) - a
	out: list[tuple[tuple[int, int], None] | tuple[tuple[int, int], tuple[int, int, int, tuple[int, ...]]]] = []
	for cl in (0, 1):
		for cr in (0, 1):
			if cl and (dl == j or a == 0):
				continue
			if cr and (dr == j or b == 0):
				continue
			# Distinct labels avoid dependence on how indices are removed.
			p: dict[int, int] = dict(enumerate(old))
			left: list[int] = list(range(a))
			right: list[int] = list(range(a, a + b))
			l: int = left.pop() if cl else a + b
			r: int = right.pop() if cr else a + b + 1
			cycles = 0
			if not cl and not cr:
				p[l] = r
				p[r] = l
			elif cl and cr:
				lp, rp = p.pop(l), p.pop(r)
				if lp == r:
					cycles += 1
				else:
					p[lp] = rp
					p[rp] = lp
			elif cl:
				partner: int = p.pop(l)
				p[partner] = r
				p[r] = partner
			else:
				partner = p.pop(r)
				p[partner] = l
				p[l] = partner
			if not cl:
				left.append(l)
			if not cr:
				right.append(r)
			nl, nr = dl + cl, dr + cr
			safe: int = max(0, min(len(left) - (j - nl), len(right) - (j - nr)))
			for s, t in zip(left[0:safe], right[0:safe], strict=False):
				sp, tp = p.pop(s), p.pop(t)
				if sp == t:
					cycles += 1
				else:
					p[sp] = tp
					p[tp] = sp
			left, right = left[safe:], right[safe:]
			if cycles:
				if cycles == 1 and nl == nr == j and not left and not right:
					out.append(((cl, cr), None))  # terminal connected component
				continue
			order: list[int] = left + right
			index: dict[int, int] = {v: i for i, v in enumerate(order)}
			pairing: tuple[int, ...] = tuple(index[p[v]] for v in order)
			out.append(((cl, cr), (nl, nr, len(left), pairing)))
	return out

def automaton(j: int) -> tuple[list[tuple[int, int, int, tuple[int, ...]]], list[list[tuple[tuple[int, int], int]]]]:
	initial = (0, 0, 0, ())
	states: list[tuple[int, int, int, tuple[int, ...]]] = [initial]
	ids: dict[tuple[int, int, int, tuple[int, ...]], int] = {initial: 0}
	edges: list[list[tuple[tuple[int, int], int]]] = []
	for s in states:
		row: list[tuple[tuple[int, int], int]] = []
		for label, t in successors(j, s):
			if t is not None and t not in ids:
				ids[t] = len(states)
				states.append(t)
			row.append((label, -1 if t is None else ids[t]))
		edges.append(row)
	return states, edges

def counts(edges: list[list[tuple[tuple[int, int], int]]], nmax: int) -> list[int]:
	active = {0: 1}
	out = [0]
	for _n in range(1, nmax + 1):
		nxt: dict[int, int] = defaultdict(int)
		accepted = 0
		for s, v in active.items():
			for _, t in edges[s]:
				if t == -1:
					accepted += v
				else:
					nxt[t] += v
		out.append(accepted)
		active = nxt
	return out

def brute(n: int, j: int) -> int:
	@lru_cache(None)
	def matchings(m: int) -> tuple[tuple[int, ...], ...]:
		if not m:
			return ((),)
		out: list[tuple[int, ...]] = []
		for k in range(m):
			r: int = 2 * k + 1
			for x in matchings(k):
				out.extend((r, *tuple(v + 1 for v in x), 0, *tuple(v + r + 1 for v in y)) for y in matchings(m - k - 1))
		return tuple(out)
	total = 0
	for p in matchings(n):
		if sum(p[i] >= n for i in range(n)) != n - 2 * j:
			continue
		v, length = 0, 0
		while True:
			length += 1
			v = 2 * n - 1 - p[v]
			if v == 0:
				break
		total += length == n
	return total

def main() -> None:
	j: int = 7
	n: int = 2 * j
	parser = argparse.ArgumentParser()
	parser.add_argument('--j', type=int, default=j)
	parser.add_argument('--nmax', type=int, default=n)
	parser.add_argument('--brute-max', type=int, default=n)
	args = parser.parse_args()
	with pathlib.Path('jobs.txt').open('a', encoding='utf-8') as f:
		f.write(f'{time.strftime("%Y-%m-%dT%H:%M:%S%z")} pid={os.getpid()} transfer j={args.j} nmax={args.nmax}\n')
	start: float = time.time()
	states, edges = automaton(args.j)
	print(f'j={args.j}: {len(states)} states, {sum(map(len, edges))} edges', flush=True)
	values = counts(edges, args.nmax)
	with pathlib.Path(f'diagonal{args.j}.csv').open('w', encoding='utf-8') as f:
		w = csv.writer(f)
		w.writerow(('n', 'count'))
		w.writerows(enumerate(values))
	with pathlib.Path(f'automaton{args.j}.json').open('w', encoding='utf-8') as f:
		json.dump({'j': args.j, 'states': states, 'edges': edges}, f)
	for n in range(2 * args.j, min(args.nmax, args.brute_max) + 1):
		v: int = brute(n, args.j)
		assert v == values[n], (n, v, values[n])  # ruff: ignore[assert]
		print(f'brute n={n}: {v}', flush=True)
	print(f'finished {time.time() - start:.3f}s', flush=True)

if __name__ == '__main__':
	# A000682(n) = sum j(n), j = 1..n//2
	# if D_j is the j-th diagonal of A400429, then 2*j is its least index.
	# j(2*j  ) = A005315(j  ) = A005316(2*j-1).
	# j(2*j+1) = A005315(j+1) = A005316(2*j+1).

	# main()
	j = 7
	for j in range(1, 12):
		print(counts(automaton(j)[1], 2 * j)[2 * j])
