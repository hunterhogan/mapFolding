"""A000682 facts."""

from __future__ import annotations

nToBitWidthTotalMeanders: dict[int, int] = {
				1: 1, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 7, 8: 8, 9: 9
	, 10: 11, 11: 13, 12: 14, 13: 16, 14: 17, 15: 19, 16: 21, 17: 22, 18: 24, 19: 25
	, 20: 27, 21: 29, 22: 30, 23: 32, 24: 34, 25: 36, 26: 37, 27: 39, 28: 41, 29: 42
	, 30: 44, 31: 46, 32: 47, 33: 49, 34: 51, 35: 53, 36: 54, 37: 56, 38: 58, 39: 60
	, 40: 61, 41: 63, 42: 65, 43: 67, 44: 68, 45: 70}

bitWidthTotalMeandersIncreasesPer_n: list[int] = [0, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 2
, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2]
bitWidthTotalMeandersIncreaseMaximum = 2
bitWidthTotalMeandersIncreaseMean = 1.57
bitWidthTotalMeandersIncreaseMedian = 2

nToNumberOfInitialArcCodes: dict[int, int] = {  # numberOfInitialArcCodes = (n // 2)
	2:	1,
	3:	1,
	4:	2,
	5:	2,
	6:	3,
	7:	3,
	8:	4,
	9:	4,
	10:	5,
	11:	5,
	12:	6,
	13:	6,
	14:	7,
	15:	7,
	16:	8,
	17:	8,
	18:	9,
	19:	9,
	20:	10,
	21:	10,
	22:	11,
	23:	11,
	24:	12,
	25:	12,
	26:	13,
	27:	13,
	28:	14,
	29:	14,
	30:	15,
	31:	15,
	32:	16,
	33:	16,
	34:	17,
	35:	17,
	36:	18,
	37:	18,
	38:	19,
	39:	19,
	40:	20,
	41:	20,
	42:	21,
	43:	21,
	44:	22,
	45:	22,
}

nToInitialArcCodes: dict[int, dict[int, int]] = {
	2:	{0x3: 1},
	4:	{0x3: 1, 0x3f: 1},
	6:	{0x3: 1, 0x3f: 1, 0x3ff: 1},  # {0b 11: 1, 0b 11 1111: 1, 0b 11 1111 1111: 1}
	8:	{0x3: 1, 0x3f: 1, 0x3ff: 1, 0x3fff: 1},
	10:	{0x3: 1, 0x3f: 1, 0x3ff: 1, 0x3fff: 1, 0x3ffff: 1},
	12:	{0x3: 1, 0x3f: 1, 0x3ff: 1, 0x3fff: 1, 0x3ffff: 1, 0x3fffff: 1},
	14:	{0x3: 1, 0x3f: 1, 0x3ff: 1, 0x3fff: 1, 0x3ffff: 1, 0x3fffff: 1, 0x3ffffff: 1},
	16:	{0x3: 1, 0x3f: 1, 0x3ff: 1, 0x3fff: 1, 0x3ffff: 1, 0x3fffff: 1, 0x3ffffff: 1, 0x3fffffff: 1},
	18:	{0x3: 1, 0x3f: 1, 0x3ff: 1, 0x3fff: 1, 0x3ffff: 1, 0x3fffff: 1, 0x3ffffff: 1, 0x3fffffff: 1, 0x3ffffffff: 1},
	20:	{0x3: 1, 0x3f: 1, 0x3ff: 1, 0x3fff: 1, 0x3ffff: 1, 0x3fffff: 1, 0x3ffffff: 1, 0x3fffffff: 1, 0x3ffffffff: 1, 0x3fffffffff: 1},
	22:	{0x3: 1, 0x3f: 1, 0x3ff: 1, 0x3fff: 1, 0x3ffff: 1, 0x3fffff: 1, 0x3ffffff: 1, 0x3fffffff: 1, 0x3ffffffff: 1, 0x3fffffffff: 1, 0x3ffffffffff: 1},
	24:	{0x3: 1, 0x3f: 1, 0x3ff: 1, 0x3fff: 1, 0x3ffff: 1, 0x3fffff: 1, 0x3ffffff: 1, 0x3fffffff: 1, 0x3ffffffff: 1, 0x3fffffffff: 1, 0x3ffffffffff: 1, 0x3fffffffffff: 1},
	26:	{0x3: 1, 0x3f: 1, 0x3ff: 1, 0x3fff: 1, 0x3ffff: 1, 0x3fffff: 1, 0x3ffffff: 1, 0x3fffffff: 1, 0x3ffffffff: 1, 0x3fffffffff: 1, 0x3ffffffffff: 1, 0x3fffffffffff: 1, 0x3ffffffffffff: 1},
	28:	{0x3: 1, 0x3f: 1, 0x3ff: 1, 0x3fff: 1, 0x3ffff: 1, 0x3fffff: 1, 0x3ffffff: 1, 0x3fffffff: 1, 0x3ffffffff: 1, 0x3fffffffff: 1, 0x3ffffffffff: 1, 0x3fffffffffff: 1, 0x3ffffffffffff: 1, 0x3fffffffffffff: 1},
	30:	{0x3: 1, 0x3f: 1, 0x3ff: 1, 0x3fff: 1, 0x3ffff: 1, 0x3fffff: 1, 0x3ffffff: 1, 0x3fffffff: 1, 0x3ffffffff: 1, 0x3fffffffff: 1, 0x3ffffffffff: 1, 0x3fffffffffff: 1, 0x3ffffffffffff: 1, 0x3fffffffffffff: 1, 0x3ffffffffffffff: 1},
	32:	{0x3: 1, 0x3f: 1, 0x3ff: 1, 0x3fff: 1, 0x3ffff: 1, 0x3fffff: 1, 0x3ffffff: 1, 0x3fffffff: 1, 0x3ffffffff: 1, 0x3fffffffff: 1, 0x3ffffffffff: 1, 0x3fffffffffff: 1, 0x3ffffffffffff: 1, 0x3fffffffffffff: 1, 0x3ffffffffffffff: 1, 0x3fffffffffffffff: 1},
	34:	{0x3: 1, 0x3f: 1, 0x3ff: 1, 0x3fff: 1, 0x3ffff: 1, 0x3fffff: 1, 0x3ffffff: 1, 0x3fffffff: 1, 0x3ffffffff: 1, 0x3fffffffff: 1, 0x3ffffffffff: 1, 0x3fffffffffff: 1, 0x3ffffffffffff: 1, 0x3fffffffffffff: 1, 0x3ffffffffffffff: 1, 0x3fffffffffffffff: 1, 0x3ffffffffffffffff: 1},
	36:	{0x3: 1, 0x3f: 1, 0x3ff: 1, 0x3fff: 1, 0x3ffff: 1, 0x3fffff: 1, 0x3ffffff: 1, 0x3fffffff: 1, 0x3ffffffff: 1, 0x3fffffffff: 1, 0x3ffffffffff: 1, 0x3fffffffffff: 1, 0x3ffffffffffff: 1, 0x3fffffffffffff: 1, 0x3ffffffffffffff: 1, 0x3fffffffffffffff: 1, 0x3ffffffffffffffff: 1, 0x3fffffffffffffffff: 1},
	38:	{0x3: 1, 0x3f: 1, 0x3ff: 1, 0x3fff: 1, 0x3ffff: 1, 0x3fffff: 1, 0x3ffffff: 1, 0x3fffffff: 1, 0x3ffffffff: 1, 0x3fffffffff: 1, 0x3ffffffffff: 1, 0x3fffffffffff: 1, 0x3ffffffffffff: 1, 0x3fffffffffffff: 1, 0x3ffffffffffffff: 1, 0x3fffffffffffffff: 1, 0x3ffffffffffffffff: 1, 0x3fffffffffffffffff: 1, 0x3ffffffffffffffffff: 1},
	40:	{0x3: 1, 0x3f: 1, 0x3ff: 1, 0x3fff: 1, 0x3ffff: 1, 0x3fffff: 1, 0x3ffffff: 1, 0x3fffffff: 1, 0x3ffffffff: 1, 0x3fffffffff: 1, 0x3ffffffffff: 1, 0x3fffffffffff: 1, 0x3ffffffffffff: 1, 0x3fffffffffffff: 1, 0x3ffffffffffffff: 1, 0x3fffffffffffffff: 1, 0x3ffffffffffffffff: 1, 0x3fffffffffffffffff: 1, 0x3ffffffffffffffffff: 1, 0x3fffffffffffffffffff: 1},
	42:	{0x3: 1, 0x3f: 1, 0x3ff: 1, 0x3fff: 1, 0x3ffff: 1, 0x3fffff: 1, 0x3ffffff: 1, 0x3fffffff: 1, 0x3ffffffff: 1, 0x3fffffffff: 1, 0x3ffffffffff: 1, 0x3fffffffffff: 1, 0x3ffffffffffff: 1, 0x3fffffffffffff: 1, 0x3ffffffffffffff: 1, 0x3fffffffffffffff: 1, 0x3ffffffffffffffff: 1, 0x3fffffffffffffffff: 1, 0x3ffffffffffffffffff: 1, 0x3fffffffffffffffffff: 1, 0x3ffffffffffffffffffff: 1},
	44:	{0x3: 1, 0x3f: 1, 0x3ff: 1, 0x3fff: 1, 0x3ffff: 1, 0x3fffff: 1, 0x3ffffff: 1, 0x3fffffff: 1, 0x3ffffffff: 1, 0x3fffffffff: 1, 0x3ffffffffff: 1, 0x3fffffffffff: 1, 0x3ffffffffffff: 1, 0x3fffffffffffff: 1, 0x3ffffffffffffff: 1, 0x3fffffffffffffff: 1, 0x3ffffffffffffffff: 1, 0x3fffffffffffffffff: 1, 0x3ffffffffffffffffff: 1, 0x3fffffffffffffffffff: 1, 0x3ffffffffffffffffffff: 1, 0x3fffffffffffffffffffff: 1},

	3:	{0xf: 1},
	5:	{0xf: 1, 0xff: 1},
	7:	{0xf: 1, 0xff: 1, 0xfff: 1},  # {0b 1111: 1, 0b 1111 1111: 1, 0b 1111 1111 1111: 1}
	9:	{0xf: 1, 0xff: 1, 0xfff: 1, 0xffff: 1},
	11:	{0xf: 1, 0xff: 1, 0xfff: 1, 0xffff: 1, 0xfffff: 1},
	13:	{0xf: 1, 0xff: 1, 0xfff: 1, 0xffff: 1, 0xfffff: 1, 0xffffff: 1},
	15:	{0xf: 1, 0xff: 1, 0xfff: 1, 0xffff: 1, 0xfffff: 1, 0xffffff: 1, 0xfffffff: 1},
	17:	{0xf: 1, 0xff: 1, 0xfff: 1, 0xffff: 1, 0xfffff: 1, 0xffffff: 1, 0xfffffff: 1, 0xffffffff: 1},
	19:	{0xf: 1, 0xff: 1, 0xfff: 1, 0xffff: 1, 0xfffff: 1, 0xffffff: 1, 0xfffffff: 1, 0xffffffff: 1, 0xfffffffff: 1},
	21:	{0xf: 1, 0xff: 1, 0xfff: 1, 0xffff: 1, 0xfffff: 1, 0xffffff: 1, 0xfffffff: 1, 0xffffffff: 1, 0xfffffffff: 1, 0xffffffffff: 1},
	23:	{0xf: 1, 0xff: 1, 0xfff: 1, 0xffff: 1, 0xfffff: 1, 0xffffff: 1, 0xfffffff: 1, 0xffffffff: 1, 0xfffffffff: 1, 0xffffffffff: 1, 0xfffffffffff: 1},
	25:	{0xf: 1, 0xff: 1, 0xfff: 1, 0xffff: 1, 0xfffff: 1, 0xffffff: 1, 0xfffffff: 1, 0xffffffff: 1, 0xfffffffff: 1, 0xffffffffff: 1, 0xfffffffffff: 1, 0xffffffffffff: 1},
	27:	{0xf: 1, 0xff: 1, 0xfff: 1, 0xffff: 1, 0xfffff: 1, 0xffffff: 1, 0xfffffff: 1, 0xffffffff: 1, 0xfffffffff: 1, 0xffffffffff: 1, 0xfffffffffff: 1, 0xffffffffffff: 1, 0xfffffffffffff: 1},
	29:	{0xf: 1, 0xff: 1, 0xfff: 1, 0xffff: 1, 0xfffff: 1, 0xffffff: 1, 0xfffffff: 1, 0xffffffff: 1, 0xfffffffff: 1, 0xffffffffff: 1, 0xfffffffffff: 1, 0xffffffffffff: 1, 0xfffffffffffff: 1, 0xffffffffffffff: 1},
	31:	{0xf: 1, 0xff: 1, 0xfff: 1, 0xffff: 1, 0xfffff: 1, 0xffffff: 1, 0xfffffff: 1, 0xffffffff: 1, 0xfffffffff: 1, 0xffffffffff: 1, 0xfffffffffff: 1, 0xffffffffffff: 1, 0xfffffffffffff: 1, 0xffffffffffffff: 1, 0xfffffffffffffff: 1},
	33:	{0xf: 1, 0xff: 1, 0xfff: 1, 0xffff: 1, 0xfffff: 1, 0xffffff: 1, 0xfffffff: 1, 0xffffffff: 1, 0xfffffffff: 1, 0xffffffffff: 1, 0xfffffffffff: 1, 0xffffffffffff: 1, 0xfffffffffffff: 1, 0xffffffffffffff: 1, 0xfffffffffffffff: 1, 0xffffffffffffffff: 1},
	35:	{0xf: 1, 0xff: 1, 0xfff: 1, 0xffff: 1, 0xfffff: 1, 0xffffff: 1, 0xfffffff: 1, 0xffffffff: 1, 0xfffffffff: 1, 0xffffffffff: 1, 0xfffffffffff: 1, 0xffffffffffff: 1, 0xfffffffffffff: 1, 0xffffffffffffff: 1, 0xfffffffffffffff: 1, 0xffffffffffffffff: 1, 0xfffffffffffffffff: 1},
	37:	{0xf: 1, 0xff: 1, 0xfff: 1, 0xffff: 1, 0xfffff: 1, 0xffffff: 1, 0xfffffff: 1, 0xffffffff: 1, 0xfffffffff: 1, 0xffffffffff: 1, 0xfffffffffff: 1, 0xffffffffffff: 1, 0xfffffffffffff: 1, 0xffffffffffffff: 1, 0xfffffffffffffff: 1, 0xffffffffffffffff: 1, 0xfffffffffffffffff: 1, 0xffffffffffffffffff: 1},
	39:	{0xf: 1, 0xff: 1, 0xfff: 1, 0xffff: 1, 0xfffff: 1, 0xffffff: 1, 0xfffffff: 1, 0xffffffff: 1, 0xfffffffff: 1, 0xffffffffff: 1, 0xfffffffffff: 1, 0xffffffffffff: 1, 0xfffffffffffff: 1, 0xffffffffffffff: 1, 0xfffffffffffffff: 1, 0xffffffffffffffff: 1, 0xfffffffffffffffff: 1, 0xffffffffffffffffff: 1, 0xfffffffffffffffffff: 1},
	41:	{0xf: 1, 0xff: 1, 0xfff: 1, 0xffff: 1, 0xfffff: 1, 0xffffff: 1, 0xfffffff: 1, 0xffffffff: 1, 0xfffffffff: 1, 0xffffffffff: 1, 0xfffffffffff: 1, 0xffffffffffff: 1, 0xfffffffffffff: 1, 0xffffffffffffff: 1, 0xfffffffffffffff: 1, 0xffffffffffffffff: 1, 0xfffffffffffffffff: 1, 0xffffffffffffffffff: 1, 0xfffffffffffffffffff: 1, 0xffffffffffffffffffff: 1},
	43:	{0xf: 1, 0xff: 1, 0xfff: 1, 0xffff: 1, 0xfffff: 1, 0xffffff: 1, 0xfffffff: 1, 0xffffffff: 1, 0xfffffffff: 1, 0xffffffffff: 1, 0xfffffffffff: 1, 0xffffffffffff: 1, 0xfffffffffffff: 1, 0xffffffffffffff: 1, 0xfffffffffffffff: 1, 0xffffffffffffffff: 1, 0xfffffffffffffffff: 1, 0xffffffffffffffffff: 1, 0xfffffffffffffffffff: 1, 0xffffffffffffffffffff: 1, 0xfffffffffffffffffffff: 1},
	45:	{0xf: 1, 0xff: 1, 0xfff: 1, 0xffff: 1, 0xfffff: 1, 0xffffff: 1, 0xfffffff: 1, 0xffffffff: 1, 0xfffffffff: 1, 0xffffffffff: 1, 0xfffffffffff: 1, 0xffffffffffff: 1, 0xfffffffffffff: 1, 0xffffffffffffff: 1, 0xfffffffffffffff: 1, 0xffffffffffffffff: 1, 0xfffffffffffffffff: 1, 0xffffffffffffffffff: 1, 0xfffffffffffffffffff: 1, 0xffffffffffffffffffff: 1, 0xfffffffffffffffffffff: 1, 0xffffffffffffffffffffff: 1},
}

lookupTotalMeandersArcCode0: dict[int, int] = {  # cf. https://oeis.org/A005315
	6:	8
	, 7:	42
	, 8:	42
	, 9:	262
	, 10:	262
	, 11:	1828
	, 12:	1828
	, 13:	13820
	, 14:	13820
	, 15:	110954
	, 16:	110954
	, 17:	933458
	, 18:	933458
	, 19:	8152860
	, 20:	8152860
	, 21:	73424650
	, 22:	73424650
	, 23:	678390116
	, 24:	678390116
	, 25:	6405031050
	, 26:	6405031050
	, 27:	61606881612
	, 28:	61606881612
	, 29:	602188541928
	, 30:	602188541928
	, 31:	5969806669034
	, 32:	5969806669034
	, 33:	59923200729046
	, 34:	59923200729046
	, 35:	608188709574124
	, 36:	608188709574124
	, 37:	6234277838531806
	, 38:	6234277838531806
	, 39:	64477712119584604
	, 40:	64477712119584604
	, 41:	672265814872772972
	, 42:	672265814872772972
}

lookupTotalMeandersArcCode1: dict[int, int] = {
	8:	100,
	10:	752,
	12:	5968,
	14:	49566,
	16:	427580,
	18:	3807200,
	20:	34816270,
	22:	325703336,
	24:	3107014138,
	26:	30145730504,
	28:	296861174940,
	30:	2961880759174,
	32:	29897469262344,
	34:	304943404747736,
	36:	3139547681462650,
	38:	32597767360886248,
	40:	341070909551229752,
	42:	3593723811002124408,

	9:	200,
	11:	1772,
	13:	15818,
	15:	143610,
	17:	1328456,
	19:	12513610,
	21:	119865856,
	23:	1165777458,
	25:	11494641496,
	27:	114747571176,
	29:	1158331933170,
	31:	11811351133710,
	33:	121544501379440,
	35:	1261194943541742,
	37:	13186378279251504,
	39:	138831428468855740,
	41:	1471037277291733368,
}

lookupTotalMeandersArcCode2: dict[int, int] = {
	10:	340,
	12:	3516,
	14:	35230,
	16:	349822,
	18:	3476110,
	20:	34717772,
	22:	349184078,
	24:	3539266774,
	26:	36155933236,
	28:	372197114904,
	30:	3859662829474,
	32:	40302920432370,
	34:	423597903947194,
	36:	4479427609071960,
	38:	47639957311742744,
	40:	509376635753344588,
	42:	5473627306050831240,

	11:	546,
	13:	6480,
	15:	72174,
	17:	779502,
	19:	8294446,
	21:	87687178,
	23:	925302318,
	25:	9771953456,
	27:	103440529038,
	29:	1098464948628,
	31:	11707647153102,
	33:	125267070807626,
	35:	1345607568375674,
	37:	14511214968621068,
	39:	157093306797219472,
	41:	1706985881466060624,
}

lookupTotalMeandersArcCode3: dict[int, int] = {
	12:	810,
	14:	10948,
	16:	135092,
	18:	1584952,
	20:	18057566,
	22:	202174358,
	24:	2240280798,
	26:	24677860682,
	28:	271000641518,
	30:	2972311647308,
	32:	32599555093292,
	34:	357829999061408,
	36:	3933011308048856,
	38:	43302385285887082,
	40:	477678505701324520,
	42:	5280281840800535268,

	13:	1170,
	15:	17808,
	17:	241670,
	19:	3065586,
	21:	37282532,
	23:	441207224,
	25:	5127673790,
	27:	58874151598,
	29:	670478793876,
	31:	7594310493766,
	33:	85716694410306,
	35:	965399477888626,
	37:	10860215868275764,
	39:	122114776687508044,
	41:	1373161335398602250,
}
