from typing import NamedTuple
import sys

class limitLocators(NamedTuple):
	bifurcationAlphaLocator: int
	bifurcationZuluLocator: int
	curveLocationsMAXIMUM: int

curveMaximum: dict[int, limitLocators] = {
0: limitLocators(0x15, 0x2a, 0x10),
1: limitLocators(0x55, 0xaa, 0x40),
2: limitLocators(0x155, 0x2aa, 0x100),
3: limitLocators(0x555, 0xaaa, 0x400),
4: limitLocators(0x1555, 0x2aaa, 0x1000),
5: limitLocators(0x5555, 0xaaaa, 0x4000),
6: limitLocators(0x15555, 0x2aaaa, 0x10000),
7: limitLocators(0x55555, 0xaaaaa, 0x40000),
8: limitLocators(0x155555, 0x2aaaaa, 0x100000),
9: limitLocators(0x555555, 0xaaaaaa, 0x400000),
10: limitLocators(0x1555555, 0x2aaaaaa, 0x1000000),
11: limitLocators(0x5555555, 0xaaaaaaa, 0x4000000),
12: limitLocators(0x15555555, 0x2aaaaaaa, 0x10000000),
13: limitLocators(0x55555555, 0xaaaaaaaa, 0x40000000),
14: limitLocators(0x155555555, 0x2aaaaaaaa, 0x100000000),
15: limitLocators(0x555555555, 0xaaaaaaaaa, 0x400000000),
16: limitLocators(0x1555555555, 0x2aaaaaaaaa, 0x1000000000),
17: limitLocators(0x5555555555, 0xaaaaaaaaaa, 0x4000000000),
18: limitLocators(0x15555555555, 0x2aaaaaaaaaa, 0x10000000000),
19: limitLocators(0x55555555555, 0xaaaaaaaaaaa, 0x40000000000),
20: limitLocators(0x155555555555, 0x2aaaaaaaaaaa, 0x100000000000),
21: limitLocators(0x555555555555, 0xaaaaaaaaaaaa, 0x400000000000),
22: limitLocators(0x1555555555555, 0x2aaaaaaaaaaaa, 0x1000000000000),
23: limitLocators(0x5555555555555, 0xaaaaaaaaaaaaa, 0x4000000000000),
24: limitLocators(0x15555555555555, 0x2aaaaaaaaaaaaa, 0x10000000000000),
25: limitLocators(0x55555555555555, 0xaaaaaaaaaaaaaa, 0x40000000000000),
26: limitLocators(0x155555555555555, 0x2aaaaaaaaaaaaaa, 0x100000000000000),
27: limitLocators(0x555555555555555, 0xaaaaaaaaaaaaaaa, 0x400000000000000),
28: limitLocators(0x1555555555555555, 0x2aaaaaaaaaaaaaaa, 0x1000000000000000),
29: limitLocators(0x5555555555555555, 0xaaaaaaaaaaaaaaaa, 0x4000000000000000),
30: limitLocators(0x15555555555555555, 0x2aaaaaaaaaaaaaaaa, 0x10000000000000000),
31: limitLocators(0x55555555555555555, 0xaaaaaaaaaaaaaaaaa, 0x40000000000000000),
32: limitLocators(0x155555555555555555, 0x2aaaaaaaaaaaaaaaaa, 0x100000000000000000),
33: limitLocators(0x555555555555555555, 0xaaaaaaaaaaaaaaaaaa, 0x400000000000000000),
34: limitLocators(0x1555555555555555555, 0x2aaaaaaaaaaaaaaaaaa, 0x1000000000000000000),
35: limitLocators(0x5555555555555555555, 0xaaaaaaaaaaaaaaaaaaa, 0x4000000000000000000),
36: limitLocators(0x15555555555555555555, 0x2aaaaaaaaaaaaaaaaaaa, 0x10000000000000000000),
37: limitLocators(0x55555555555555555555, 0xaaaaaaaaaaaaaaaaaaaa, 0x40000000000000000000),
38: limitLocators(0x155555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaa, 0x100000000000000000000),
39: limitLocators(0x555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaa, 0x400000000000000000000),
40: limitLocators(0x1555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaa, 0x1000000000000000000000),
41: limitLocators(0x5555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaa, 0x4000000000000000000000),
42: limitLocators(0x15555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaaa, 0x10000000000000000000000),
43: limitLocators(0x55555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaaa, 0x40000000000000000000000),
44: limitLocators(0x155555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaaaa, 0x100000000000000000000000),
45: limitLocators(0x555555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaaaa, 0x400000000000000000000000),
46: limitLocators(0x1555555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaaaaa, 0x1000000000000000000000000),
47: limitLocators(0x5555555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaaaaa, 0x4000000000000000000000000),
48: limitLocators(0x15555555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaaaaaa, 0x10000000000000000000000000),
49: limitLocators(0x55555555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x40000000000000000000000000),
50: limitLocators(0x155555555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaaaaaaa, 0x100000000000000000000000000),
51: limitLocators(0x555555555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x400000000000000000000000000),
52: limitLocators(0x1555555555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x1000000000000000000000000000),
53: limitLocators(0x5555555555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x4000000000000000000000000000),
54: limitLocators(0x15555555555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x10000000000000000000000000000),
55: limitLocators(0x55555555555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x40000000000000000000000000000),
56: limitLocators(0x155555555555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x100000000000000000000000000000),
57: limitLocators(0x555555555555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x400000000000000000000000000000),
58: limitLocators(0x1555555555555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x1000000000000000000000000000000),
59: limitLocators(0x5555555555555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x4000000000000000000000000000000),
60: limitLocators(0x15555555555555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x10000000000000000000000000000000),
61: limitLocators(0x55555555555555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x40000000000000000000000000000000),
}

def makeCurveMaximum() -> None:
	sys.stdout.write("curveMaximum: dict[int, limitLocators] = {\n")
	for n in range(62):
		curveLocationsMAXIMUM = 1 << (2 * n + 4)
		bifurcationAlphaLocator = int('01' * ((curveLocationsMAXIMUM.bit_length() + 1) // 2), 2)
		sys.stdout.write(f"{n}: limitLocators({hex(bifurcationAlphaLocator)}, {hex(bifurcationAlphaLocator << 1)}, {hex(curveLocationsMAXIMUM)}),\n")
	sys.stdout.write("}\n")

if __name__ == '__main__':
	makeCurveMaximum()

