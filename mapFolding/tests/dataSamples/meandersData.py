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

dataframeClosures8: polars.DataFrame = polars.DataFrame({'arcCode': [38, 25, 114, 177], 'meanders': [7, 11, 13, 17]}, schema={'arcCode': polars.UInt8, 'meanders': polars.UInt8})
dataframeAligned8: polars.DataFrame = polars.DataFrame({'arcCode': [34, 17, 50, 49], 'meanders': [7, 11, 13, 17]}, schema={'arcCode': polars.UInt8, 'meanders': polars.UInt8})
dataframeClosures128: polars.DataFrame = polars.DataFrame(
	{'arcCode': [0x5555555555555555000000000000000A, 0xAAAAAAAAAAAAAAAA0000000000000005], 'meanders': [19, 23]}, schema={'arcCode': polars.UInt128, 'meanders': polars.UInt8}
)
dataframeAligned128: polars.DataFrame = polars.DataFrame(
	{'arcCode': [0x1555555555555555000000000000000A, 0x2AAAAAAAAAAAAAAA0000000000000005], 'meanders': [19, 23]}, schema={'arcCode': polars.UInt128, 'meanders': polars.UInt8}
)
