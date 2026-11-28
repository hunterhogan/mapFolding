"""Validate and save contiguous overlap-permutation statistics from the C enumerator.

Run from the activated repository environment after building permutationStatistics.c:
    python -m research.collectPermutationStatistics 15 23 --executable /tmp/permutationStatistics

The upper bound is inclusive. Existing reference cells, semi-meander totals, and
restricted row sums must all agree before any terms from a row are saved.
Use --partitions 64 --workers 8 --checkpoint-dir PATH to distribute disjoint
prefix subtrees and retain each completed partition for later resumption.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import count, repeat
from mapFolding.oeis import getTriangle, getTriangleCells, getValuesKnown, updateTriangle, updateValuesKnown
from pathlib import Path
from time import perf_counter
from typing import Literal, NotRequired, TYPE_CHECKING, TypedDict
import argparse
import hashlib
import json
#=SIN= `subprocess`: only `subprocess` runs the compiled C enumerator and captures its JSON output.
# ruff: ignore[suspicious-subprocess-import]
import subprocess

if TYPE_CHECKING:
    from mapFolding.oeis import OEISid

type Statistic = Literal['runs', 'first', 'restricted']

class PermutationStatistics(TypedDict):
    n: int
    total: int
    runs: list[int]
    first: list[int]
    restricted: list[int]
    partition: NotRequired[int]
    partitions: NotRequired[int]
    workers: NotRequired[int]
    seconds: NotRequired[float]
    weighted_runs: NotRequired[int]

def run_partition(args: argparse.Namespace, n: int, partition: int) -> PermutationStatistics:
    cache = args.checkpoint_dir / f'n{n}_part{partition}_of{args.partitions}.json' if args.checkpoint_dir else None
    if cache and cache.exists():
        partial = json.loads(cache.read_text())
    else:
        command = [str(args.executable), str(n)]
        if args.partitions > 1:
            command.extend([str(partition), str(args.partitions)])
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        partial = json.loads(result.stdout)
        if cache:
            temporary = cache.with_suffix('.tmp')
            temporary.write_text(json.dumps(partial) + '\n')
            temporary.replace(cache)
    assert partial['n'] == n
    assert partial.get('partition', 0) == partition
    assert partial.get('partitions', 1) == args.partitions
    return partial

def sum_partials(partials: list[PermutationStatistics], statistic: Statistic) -> list[int]:
    return [sum(values) for values in zip(*(partial[statistic] for partial in partials), strict=True)]

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('start', type=int)
    parser.add_argument('stop', type=int)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--partitions', type=int, default=1)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--checkpoint-dir', type=Path)
    args = parser.parse_args()
    if args.partitions < 1 or args.partitions > 1024 or args.workers < 1:
        parser.error('Use 1..1024 partitions and a positive worker count.')
    if args.checkpoint_dir:
        args.checkpoint_dir /= hashlib.sha256(args.executable.read_bytes()).hexdigest()
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    families: dict[OEISid, Statistic] = {'A259698': 'runs', 'A259703': 'first', 'A259701': 'restricted'}
    for n in range(args.start, args.stop + 1):
        print(f'Enumerating n={n}', flush=True)
        start = perf_counter()

        pool = ThreadPoolExecutor(max_workers=min(args.workers, args.partitions))
        try:
            partials = list(pool.map(run_partition, repeat(args), repeat(n), range(args.partitions)))
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        record: PermutationStatistics = {
            'n': n,
            'total': sum(partial['total'] for partial in partials),
            'runs': sum_partials(partials, 'runs'),
            'first': sum_partials(partials, 'first'),
            'restricted': sum_partials(partials, 'restricted'),
        }
        if args.partitions > 1:
            record['partitions'] = args.partitions
            record['workers'] = args.workers
        assert record['n'] == n
        assert record['total'] == getValuesKnown('A000682')[n]
        assert sum(record['runs']) == sum(record['first']) == record['total']
        assert sum(record['restricted'][1:]) == getValuesKnown('A259702')[n]
        weighted = sum(k * value for k, value in enumerate(record['runs'], start=1))
        known_weighted = getValuesKnown('A259700').get(n)
        assert known_weighted is None or known_weighted == weighted
        for oeis_id, key in families.items():
            row = record[key]
            assert len(row) == n - 1
            assert all(isinstance(x, int) and x >= 0 for x in row)
            assert list(map(getTriangleCells(oeis_id).get, zip(repeat(n), count(1)), row, strict=False)) == row
            # Each row must touch the existing prefix; never persist isolated future rows.
            triangle = getTriangle(oeis_id)
            assert n in triangle or len(triangle.get(n - 1, ())) == n - 2
        record['seconds'] = round(perf_counter() - start, 6)
        record['weighted_runs'] = weighted
        with Path(__file__).with_name('permutation_statistics.jsonl').open('a') as output:
            output.write(json.dumps(record) + '\n')
        for oeis_id, key in families.items():
            updateTriangle(oeis_id, {n: record[key]})
        updateValuesKnown('A259700', {n: weighted})
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
