from __future__ import annotations

import polars

dataframePacked6: polars.DataFrame = polars.DataFrame({'arcCode': [0x3F], 'meanders': [7]}, schema={'arcCode': polars.UInt8, 'meanders': polars.UInt8})
dataframePacked14: polars.DataFrame = polars.DataFrame({'arcCode': [0x3FFF], 'meanders': [7]}, schema={'arcCode': polars.UInt16, 'meanders': polars.UInt8})
dataframePacked30: polars.DataFrame = polars.DataFrame({'arcCode': [0x3FFFFFFF], 'meanders': [7]}, schema={'arcCode': polars.UInt32, 'meanders': polars.UInt8})
dataframePacked62: polars.DataFrame = polars.DataFrame({'arcCode': [0x3FFFFFFFFFFFFFFF], 'meanders': [7]}, schema={'arcCode': polars.UInt64, 'meanders': polars.UInt8})
dataframePacked126: polars.DataFrame = polars.DataFrame({'arcCode': [0x3FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF], 'meanders': [7]}, schema={'arcCode': polars.UInt128, 'meanders': polars.UInt8})
dataframePackedMixed6: polars.DataFrame = polars.DataFrame({'arcCode': [22, 41, 63], 'meanders': [13, 11, 7]}, schema={'arcCode': polars.UInt8, 'meanders': polars.UInt8})

dataframeTransition8: polars.DataFrame = polars.DataFrame({'arcCode': [0xFF, 0xAD, 0x5E], 'meanders': [7, 7, 7]}, schema={'arcCode': polars.UInt8, 'meanders': polars.UInt8})
dataframeTransition16: polars.DataFrame = polars.DataFrame({'arcCode': [0xFFFF, 0xAFFD, 0x5FFE], 'meanders': [7, 7, 7]}, schema={'arcCode': polars.UInt16, 'meanders': polars.UInt8})
dataframeTransition32: polars.DataFrame = polars.DataFrame({'arcCode': [0xFFFFFFFF, 0xAFFFFFFD, 0x5FFFFFFE], 'meanders': [7, 7, 7]}, schema={'arcCode': polars.UInt32, 'meanders': polars.UInt8})
dataframeTransition64: polars.DataFrame = polars.DataFrame(
	{'arcCode': [0xFFFFFFFFFFFFFFFF, 0xAFFFFFFFFFFFFFFD, 0x5FFFFFFFFFFFFFFE], 'meanders': [7, 7, 7]}, schema={'arcCode': polars.UInt64, 'meanders': polars.UInt8}
)
dataframeTransition128: polars.DataFrame = polars.DataFrame(
	{'arcCode': [0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF, 0xAFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFD, 0x5FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFE], 'meanders': [7, 7, 7]},
	schema={'arcCode': polars.UInt128, 'meanders': polars.UInt8},
)
dataframeTransitionMixed6: polars.DataFrame = polars.DataFrame({'arcCode': [15, 15], 'meanders': [13, 11]}, schema={'arcCode': polars.UInt8, 'meanders': polars.UInt8})
dataframeUnsigned8Empty: polars.DataFrame = polars.DataFrame(schema={'arcCode': polars.UInt8, 'meanders': polars.UInt8})

tupleDataframesOverlapping8: tuple[polars.DataFrame, ...] = (
	dataframePackedMixed6, dataframeTransitionMixed6, dataframeUnsigned8Empty, dataframePackedMixed6)
dataframeCountsOverlapping8: polars.DataFrame = polars.DataFrame(
	{'arcCode': [15, 22, 41, 63], 'meanders': [24, 26, 22, 14]}, schema={'arcCode': polars.UInt8, 'meanders': polars.UInt8})
tupleDataframesOverlapping128: tuple[polars.DataFrame, ...] = (dataframePacked126, dataframeTransition128, dataframePacked126)
dataframeCountsOverlapping128: polars.DataFrame = polars.DataFrame(
	{'arcCode': [0x3FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF, 0x5FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFE,
		0xAFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFD, 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF], 'meanders': [14, 7, 7, 7]},
	schema={'arcCode': polars.UInt128, 'meanders': polars.UInt8})
tupleDataframesEmpty8: tuple[polars.DataFrame, ...] = (dataframeUnsigned8Empty, dataframeUnsigned8Empty)
dataframePackedCount149: polars.DataFrame = polars.DataFrame(
	{'arcCode': [0x3F], 'meanders': [149]}, schema={'arcCode': polars.UInt8, 'meanders': polars.UInt8})
tupleDataframesCount298: tuple[polars.DataFrame, ...] = (dataframePackedCount149, dataframePackedCount149)
dataframeNegativeArcCode: polars.LazyFrame = polars.LazyFrame().select(
	polars.lit(-1).cast(polars.UInt8).alias('arcCode'), polars.lit(7, dtype=polars.UInt8).alias('meanders'))
tupleAnalysesNegativeArcCode: tuple[polars.LazyFrame, ...] = (dataframePacked6.lazy(), dataframeNegativeArcCode)

dataframeClosures8: polars.DataFrame = polars.DataFrame({'arcCode': [38, 25, 114, 177], 'meanders': [7, 11, 13, 17]}, schema={'arcCode': polars.UInt8, 'meanders': polars.UInt8})
dataframeAligned8: polars.DataFrame = polars.DataFrame({'arcCode': [34, 17, 50, 49], 'meanders': [7, 11, 13, 17]}, schema={'arcCode': polars.UInt8, 'meanders': polars.UInt8})
dataframeClosures128: polars.DataFrame = polars.DataFrame(
	{'arcCode': [0x5555555555555555000000000000000A, 0xAAAAAAAAAAAAAAAA0000000000000005], 'meanders': [19, 23]}, schema={'arcCode': polars.UInt128, 'meanders': polars.UInt8}
)
dataframeAligned128: polars.DataFrame = polars.DataFrame(
	{'arcCode': [0x1555555555555555000000000000000A, 0x2AAAAAAAAAAAAAAA0000000000000005], 'meanders': [19, 23]}, schema={'arcCode': polars.UInt128, 'meanders': polars.UInt8}
)
