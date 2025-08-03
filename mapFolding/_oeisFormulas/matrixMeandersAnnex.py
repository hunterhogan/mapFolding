from typing import NamedTuple
import sys

class limitLocators(NamedTuple):
	curveLocationsMAXIMUM: int
	bifurcationEvenLocator: int
	bifurcationOddLocator: int

curveMaximum: dict[int, limitLocators] = {
0: limitLocators(16, 0x2a, 0x15),
1: limitLocators(64, 0xaa, 0x55),
2: limitLocators(256, 0x2aa, 0x155),
3: limitLocators(1024, 0xaaa, 0x555),
4: limitLocators(4096, 0x2aaa, 0x1555),
5: limitLocators(16384, 0xaaaa, 0x5555),
6: limitLocators(65536, 0x2aaaa, 0x15555),
7: limitLocators(262144, 0xaaaaa, 0x55555),
8: limitLocators(1048576, 0x2aaaaa, 0x155555),
9: limitLocators(4194304, 0xaaaaaa, 0x555555),
10: limitLocators(16777216, 0x2aaaaaa, 0x1555555),
11: limitLocators(67108864, 0xaaaaaaa, 0x5555555),
12: limitLocators(268435456, 0x2aaaaaaa, 0x15555555),
13: limitLocators(1073741824, 0xaaaaaaaa, 0x55555555),
14: limitLocators(4294967296, 0x2aaaaaaaa, 0x155555555),
15: limitLocators(17179869184, 0xaaaaaaaaa, 0x555555555),
16: limitLocators(68719476736, 0x2aaaaaaaaa, 0x1555555555),
17: limitLocators(274877906944, 0xaaaaaaaaaa, 0x5555555555),
18: limitLocators(1099511627776, 0x2aaaaaaaaaa, 0x15555555555),
19: limitLocators(4398046511104, 0xaaaaaaaaaaa, 0x55555555555),
20: limitLocators(17592186044416, 0x2aaaaaaaaaaa, 0x155555555555),
21: limitLocators(70368744177664, 0xaaaaaaaaaaaa, 0x555555555555),
22: limitLocators(281474976710656, 0x2aaaaaaaaaaaa, 0x1555555555555),
23: limitLocators(1125899906842624, 0xaaaaaaaaaaaaa, 0x5555555555555),
24: limitLocators(4503599627370496, 0x2aaaaaaaaaaaaa, 0x15555555555555),
25: limitLocators(18014398509481984, 0xaaaaaaaaaaaaaa, 0x55555555555555),
26: limitLocators(72057594037927936, 0x2aaaaaaaaaaaaaa, 0x155555555555555),
27: limitLocators(288230376151711744, 0xaaaaaaaaaaaaaaa, 0x555555555555555),
28: limitLocators(1152921504606846976, 0x2aaaaaaaaaaaaaaa, 0x1555555555555555), # 0x2aaaaaaaaaaaaaaa.bit_length() = 62
29: limitLocators(4611686018427387904, 0xaaaaaaaaaaaaaaaa, 0x5555555555555555),
30: limitLocators(18446744073709551616, 0x2aaaaaaaaaaaaaaaa, 0x15555555555555555),
31: limitLocators(73786976294838206464, 0xaaaaaaaaaaaaaaaaa, 0x55555555555555555),
32: limitLocators(295147905179352825856, 0x2aaaaaaaaaaaaaaaaa, 0x155555555555555555),
33: limitLocators(1180591620717411303424, 0xaaaaaaaaaaaaaaaaaa, 0x555555555555555555),
34: limitLocators(4722366482869645213696, 0x2aaaaaaaaaaaaaaaaaa, 0x1555555555555555555),
35: limitLocators(18889465931478580854784, 0xaaaaaaaaaaaaaaaaaaa, 0x5555555555555555555),
36: limitLocators(75557863725914323419136, 0x2aaaaaaaaaaaaaaaaaaa, 0x15555555555555555555),
37: limitLocators(302231454903657293676544, 0xaaaaaaaaaaaaaaaaaaaa, 0x55555555555555555555),
38: limitLocators(1208925819614629174706176, 0x2aaaaaaaaaaaaaaaaaaaa, 0x155555555555555555555),
39: limitLocators(4835703278458516698824704, 0xaaaaaaaaaaaaaaaaaaaaa, 0x555555555555555555555),
40: limitLocators(19342813113834066795298816, 0x2aaaaaaaaaaaaaaaaaaaaa, 0x1555555555555555555555),
41: limitLocators(77371252455336267181195264, 0xaaaaaaaaaaaaaaaaaaaaaa, 0x5555555555555555555555),
42: limitLocators(309485009821345068724781056, 0x2aaaaaaaaaaaaaaaaaaaaaa, 0x15555555555555555555555),
43: limitLocators(1237940039285380274899124224, 0xaaaaaaaaaaaaaaaaaaaaaaa, 0x55555555555555555555555),
44: limitLocators(4951760157141521099596496896, 0x2aaaaaaaaaaaaaaaaaaaaaaa, 0x155555555555555555555555),
45: limitLocators(19807040628566084398385987584, 0xaaaaaaaaaaaaaaaaaaaaaaaa, 0x555555555555555555555555),
46: limitLocators(79228162514264337593543950336, 0x2aaaaaaaaaaaaaaaaaaaaaaaa, 0x1555555555555555555555555),
47: limitLocators(316912650057057350374175801344, 0xaaaaaaaaaaaaaaaaaaaaaaaaa, 0x5555555555555555555555555),
48: limitLocators(1267650600228229401496703205376, 0x2aaaaaaaaaaaaaaaaaaaaaaaaa, 0x15555555555555555555555555),
49: limitLocators(5070602400912917605986812821504, 0xaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x55555555555555555555555555),
50: limitLocators(20282409603651670423947251286016, 0x2aaaaaaaaaaaaaaaaaaaaaaaaaa, 0x155555555555555555555555555),
51: limitLocators(81129638414606681695789005144064, 0xaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x555555555555555555555555555),
52: limitLocators(324518553658426726783156020576256, 0x2aaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x1555555555555555555555555555),
53: limitLocators(1298074214633706907132624082305024, 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x5555555555555555555555555555),
54: limitLocators(5192296858534827628530496329220096, 0x2aaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x15555555555555555555555555555),
55: limitLocators(20769187434139310514121985316880384, 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x55555555555555555555555555555),
56: limitLocators(83076749736557242056487941267521536, 0x2aaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x155555555555555555555555555555),
57: limitLocators(332306998946228968225951765070086144, 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x555555555555555555555555555555),
58: limitLocators(1329227995784915872903807060280344576, 0x2aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x1555555555555555555555555555555),
59: limitLocators(5316911983139663491615228241121378304, 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x5555555555555555555555555555555),
60: limitLocators(21267647932558653966460912964485513216, 0x2aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x15555555555555555555555555555555),
61: limitLocators(85070591730234615865843651857942052864, 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x55555555555555555555555555555555),
}

def makeCurveMaximum() -> None:
	sys.stdout.write("curveMaximum: dict[int, limitLocators] = {\n")
	for n in range(62):
		curveLocationsMAXIMUM = 1 << (2 * n + 4)
		bifurcationOddLocator = int('01' * ((curveLocationsMAXIMUM.bit_length() + 1) // 2), 2)
		sys.stdout.write(f"{n}: limitLocators({curveLocationsMAXIMUM}, {hex(bifurcationOddLocator << 1)}, {hex(bifurcationOddLocator)}),\n")
	sys.stdout.write("}\n")

if __name__ == '__main__':
	makeCurveMaximum()

