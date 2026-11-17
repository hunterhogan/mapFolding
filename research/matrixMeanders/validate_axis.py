"""Benchmark axis-meander flows against stored terms; log candidates without accepting them.

Run from the repository root, for example:
    python -m research.matrixMeanders.validate_axis matrixAxisPileNumPy 26 27
    python -m research.matrixMeanders.validate_axis pile 26 27 --cpp /tmp/axis_transfer

Results append to axis_validation.jsonl beside this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import resource
import subprocess
from time import perf_counter

from mapFolding.basecamp import countMeanders
from mapFolding.oeis import getValuesKnown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    python_flows = ('matrixAxisPile', 'matrixAxisPileOrbit', 'matrixAxisPileNumPy',
                    'matrixAxisPileDeadEnd', 'matrixAxisPileDeadEndOrbit',
                    'matrixAxisStripGap', 'matrixAxisStripGapOrbit')
    cpp_flows = ('pile', 'pile-wide', 'gap', 'gap-compact', 'pile-plain', 'gap-plain')
    parser.add_argument('flow', choices=python_flows + cpp_flows)
    parser.add_argument('indices', type=int, nargs='+')
    parser.add_argument('--cpp', type=Path, help='Use a compiled axis_transfer executable.')
    parser.add_argument('--tail', type=int, default=1, help='Stream this many final layers.')
    parser.add_argument('--save', action='store_true', help='Save newly computed terms immediately.')
    args = parser.parse_args()
    if args.flow not in (cpp_flows if args.cpp else python_flows):
        parser.error('Use --cpp with a C++ flow; omit it with a Python flow.')
    known_axis = dict(getValuesKnown('A223096'))
    known_shapes = dict(getValuesKnown('A077055'))
    known_open = dict(getValuesKnown('A005316'))
    path = Path(__file__).with_name('axis_validation.jsonl')
    for n in args.indices:
        print(f'Start {args.flow} A223096({n})', flush=True)
        start = perf_counter()
        if args.cpp:
            result = subprocess.run([str(args.cpp), args.flow, str(n), str(args.tail)],
                                    check=True, stdout=subprocess.PIPE, text=True)
            record = json.loads(result.stdout)
            value = record['value']
        else:
            value = int(countMeanders('symmetric', n, args.flow).totalMeanders)
            record = dict(flow=args.flow, n=n, value=value,
                          seconds=round(perf_counter() - start, 6),
                          max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          python=platform.python_version())
        record['expected'] = known_axis.get(n)
        if n >= 1 and 2 * n + 1 in known_open:
            numerator = known_open[2 * n + 1] + 2 * value
            record['shape_n'] = 2 * n + 1
            record['shape_numerator_remainder'] = numerator % 4
            record['shape_value'] = numerator // 4
            record['shape_expected'] = known_shapes.get(2 * n + 1)
        with path.open('a') as output:
            output.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        if n in known_axis:
            assert value == known_axis[n], record
        if 'shape_value' in record:
            assert record['shape_numerator_remainder'] == 0, record
            if 2 * n + 1 in known_shapes:
                assert record['shape_value'] == known_shapes[2 * n + 1], record
        if args.save:
            from mapFolding.oeis import updateValuesKnown
            updateValuesKnown('A223096', {n: value})
            if 'shape_value' in record:
                updateValuesKnown('A077055', {record['shape_n']: record['shape_value']})


if __name__ == '__main__':
    main()
