# pyright: reportMissingParameterType=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
# ruff: file-ignore[missing-return-type-undocumented-public-function, undocumented-public-function, multiple-statements-on-one-line-semicolon, collection-literal-concatenation, print]
# ruff: file-ignore[missing-type-function-argument]
"""Semi-meanders with n crossings and winding number n - 2h, for a FIXED h.

Contributed by Alper Ferudun (https://oeis.org/wiki/User:Alper_Ferudun), October 2026.

Picture: fold the semi-meander along the ray.  Positions 1..n on the ray; each
position joins its top half-edge to its bottom half-edge.  On each side (top,
bottom) there is a noncrossing partial matching with exactly h arches; the
remaining "free" points lie at depth 0 and wind around the end of the ray: the
j-th free top point (from the left) is joined to the j-th free bottom point.
The winding number is the number of free points per side, n - 2h.  We count
configurations that form a single loop.

Why this is cheap: with h fixed, at most h arches are ever open on each side,
and the free points only form a FIFO queue that is tagged with the side that
is waiting.  So the transfer matrix has a bounded number of states,
independent of n: a(n) for h = 5 up to n = 4000 takes about 45 s in plain
Python with 2685 states.  For the full semi-meander count (all h at once, i.e.
A000682) there is no such cap, and the state space grows exponentially.

Checks: h = 3, 4, 5 reproduce the data of A400631, A400632, A400633 (offsets
6, 8, 10), and summing over h = 0..floor(n/2) reproduces A000682(n) for
n = 1..16.

Usage:  python fixed_deficit_semimeanders.py h N   (prints n a(n) for n <= N)
"""
from __future__ import annotations

from collections import defaultdict
import sys

def canon(ts, bs, qs, q):
    # Relabel loop-segment ids in order of first appearance so equal states merge.
    m = {}
    out = []
    for seq in (ts, bs, q):
        o = []
        for x in seq:
            if x not in m:
                m[x] = len(m)
            o.append(m[x])
        out.append(tuple(o))
    return (out[0], out[1], qs if q else 0, out[2])


def run(h, N):
    # State: (top arch stack, bottom arch stack, queue side 'T'/'B'/0, queue of
    # free points waiting for a partner, arches opened on top, on bottom) -> count.
    # Stack/queue entries are ids of the open loop segments they belong to.
    states = defaultdict(int)
    states[(), (), 0, (), 0, 0] = 1
    res = {}
    X = -1  # id of the segment through the current position
    for i in range(1, N + 1):
        new = defaultdict(int)
        done = 0
        for (ts, bs, qs, q, at, ab), c in states.items():
            # top half-edge at position i: open an arch, close one, or be free
            tops = []
            if at < h:
                tops.append('open')
            if ts:
                tops.append('close')
            if not ts:
                tops.append('free')  # free points only at depth 0
            for ta in tops:
                ts1 = list(ts); bs1 = list(bs); q1 = list(q); qs1 = qs; at1 = at
                Y = None
                if ta == 'open':
                    ts1.append(X); at1 += 1
                elif ta == 'close':
                    Y = ts1.pop()
                elif qs1 == 'B' and q1:
                    Y = q1.pop(0)      # meets the oldest waiting bottom free point
                else:
                    qs1 = 'T'; q1.append(X)
                if Y is not None:          # merge segment Y into the current one
                    ts1 = [X if v == Y else v for v in ts1]
                    bs1 = [X if v == Y else v for v in bs1]
                    q1 = [X if v == Y else v for v in q1]
                if not q1:
                    qs1 = 0
                # bottom half-edge at position i
                bots = []
                if ab < h:
                    bots.append('open')
                if bs1:
                    bots.append('close')
                if not bs1:
                    bots.append('free')
                for ba in bots:
                    ts2 = list(ts1); bs2 = list(bs1); q2 = list(q1); qs2 = qs1; ab2 = ab
                    Y = None
                    if ba == 'open':
                        bs2.append(X); ab2 += 1
                    elif ba == 'close':
                        Y = bs2.pop()
                    elif qs2 == 'T' and q2:
                        Y = q2.pop(0)
                    else:
                        qs2 = 'B'; q2.append(X)
                    if Y is not None:
                        if Y == X and (X not in ts2) and (X not in bs2) and (X not in q2):
                            # the current segment closed on itself: a loop
                            if not ts2 and not bs2 and not q2 and at1 == h and ab2 == h:
                                done += c   # single loop using everything: count it
                            continue        # otherwise a stray loop: discard
                        ts2 = [X if v == Y else v for v in ts2]
                        bs2 = [X if v == Y else v for v in bs2]
                        q2 = [X if v == Y else v for v in q2]
                    if not q2:
                        qs2 = 0
                    new[canon(ts2, bs2, qs2, q2) + (at1, ab2)] += c
        res[i] = done
        states = new
    return res, len(states)


if __name__ == '__main__':
    h = int(sys.argv[1]); N = int(sys.argv[2])
    res, ns = run(h, N)
    for i in range(1, N + 1):
        if res[i]:
            print(i, res[i])
    print('#states', ns, file=sys.stderr)
