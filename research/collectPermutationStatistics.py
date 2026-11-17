"""Validate and save contiguous overlap-permutation statistics from the C enumerator.

Run from the activated repository environment after building permutationStatistics.c:
    python -m research.collectPermutationStatistics 15 23 --executable /tmp/permutationStatistics

The upper bound is inclusive. Existing reference cells, semi-meander totals, and
restricted row sums must all agree before any terms from a row are saved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from time import perf_counter

from mapFolding.dataStructures import flattenTriangle
from mapFolding.oeis import getMetadata, getValuesKnown, updateTriangle, updateValuesKnown
from mapFolding.oeis._metadata import getMetadataTriangle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('start', type=int)
    parser.add_argument('stop', type=int)
    parser.add_argument('--executable', type=Path, required=True)
    args = parser.parse_args()
    families = {'A259698': 'runs', 'A259703': 'first', 'A259701': 'restricted'}
    for n in range(args.start, args.stop + 1):
        print(f'Enumerating n={n}', flush=True)
        start = perf_counter()
        result = subprocess.run([str(args.executable), str(n)], check=True, capture_output=True, text=True)
        record = json.loads(result.stdout)
        assert record['n'] == n
        assert record['total'] == getValuesKnown('A000682')[n]
        assert sum(record['runs']) == sum(record['first']) == record['total']
        assert sum(record['restricted'][1:]) == getValuesKnown('A259702')[n]
        weighted = sum(k * value for k, value in enumerate(record['runs'], start=1))
        known_weighted = getValuesKnown('A259700').get(n)
        assert known_weighted is None or known_weighted == weighted
        for oeis_id, key in families.items():
            row = record[key]
            assert len(row) == n - 1 and all(isinstance(x, int) and x >= 0 for x in row)
            metadata = getMetadataTriangle(oeis_id)
            flattened = flattenTriangle({n: row}, metadata['rowLength'], metadata['rowStart'], metadata['offset'])
            known = getValuesKnown(oeis_id)
            assert all(known[index] == value for index, value in flattened.items() if index in known)
            # Each row must touch the existing prefix; never persist isolated future rows.
            first_missing = next(i for i in range(getMetadata(oeis_id)['offset'], max(known) + 2) if i not in known)
            assert min(flattened) <= first_missing
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
