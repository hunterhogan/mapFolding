"""Extend A007822 and A001010 contiguously using the C++ strip transfer.

    python -m research.matrixMeanders.validate_strip /tmp/axis_transfer-strip 24 30

Arguments are strip-state indices; state n counts A007822(n+1).
Each result must agree with a second run without reflection-orbit merging,
using a different number of streamed layers. Existing reference values are
checked before any additions are saved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from mapFolding.oeis import getValuesKnown, updateValuesKnown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable', type=Path)
    parser.add_argument('start', type=int)
    parser.add_argument('stop', type=int)
    args = parser.parse_args()
    for n in range(args.start, args.stop + 1):
        print(f'Start A007822({n+1})', flush=True)
        records = []
        for flow, tail in [('strip', 3), ('strip-plain', 5)]:
            result = subprocess.run([str(args.executable), flow, str(n), str(tail)],
                                    check=True, capture_output=True, text=True)
            records.append(json.loads(result.stdout))
        value = records[0]['value']
        assert records[1]['value'] == value, records
        known = getValuesKnown('A007822')
        assert n + 1 not in known or known[n + 1] == value, records
        assert all(k in known for k in range(1, n + 1))
        record = {'sequence': 'A007822', 'n': n + 1, 'value': value, 'runs': records}
        with Path(__file__).with_name('strip_validation.jsonl').open('a') as output:
            output.write(json.dumps(record) + '\n')
        updateValuesKnown('A007822', {n + 1: value})
        # A001010(2*m-1)=2*A007822(m); A001010(2*m)=2*A000682(m+1).
        symmetric = getValuesKnown('A001010')
        candidates = {2 * n + 1: 2 * value, 2 * n + 2: 2 * getValuesKnown('A000682')[n + 2]}
        assert all(index not in symmetric or symmetric[index] == term for index, term in candidates.items())
        assert all(k in symmetric for k in range(1, 2 * n + 1))
        updateValuesKnown('A001010', candidates)
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
