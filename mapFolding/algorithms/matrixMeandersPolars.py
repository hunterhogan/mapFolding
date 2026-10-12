from __future__ import annotations

from mapFolding.dataStructures import getDatatypePolars
from mapFolding.kitFilesystem import storePolars
from mapFolding.synthesized.matrixMeanders.bigIntPolars import countBigInt
from mapFolding.synthesized.matrixMeanders.polarsWide import integersWidePolars吗
from typing import TYPE_CHECKING
import polars

if TYPE_CHECKING:
	from collections.abc import Callable
	from mapFolding.dataBaskets import StateMeanders
	from mapFolding.kitFilesystem import _PolarsMaterializer
	from mapFolding.theTypes import 形PolarsInteger

def addLoop(arcCode: polars.Expr, 形ArcCode: 形PolarsInteger) -> polars.Expr:
	return (arcCode.cast(形ArcCode) * 4) | 3

def dragUp(bitsAlfa: polars.Expr, bitsZulu: polars.Expr, 形ArcCode: 形PolarsInteger) -> polars.Expr:
	# This has a stack. It is probably impossible to eliminate it, so don't multiply it with more stacks.
	return (bitsZulu.cast(形ArcCode) * 8) | ((bitsAlfa // 4).cast(形ArcCode)) | ((((bitsAlfa & 1) ^ 1) * 2).cast(polars.UInt8))

def dragDown(bitsAlfa: polars.Expr, bitsZulu: polars.Expr, 形ArcCode: 形PolarsInteger) -> polars.Expr:
	# This has a stack. It is probably impossible to eliminate it, so don't multiply it with more stacks.
	return (bitsAlfa.cast(形ArcCode) * 4) | ((bitsZulu // 2).cast(形ArcCode)) | (((bitsZulu & 1) ^ 1).cast(polars.UInt8))

def connectArcs(arcCode: polars.Expr) -> polars.Expr:
	return arcCode // 4

def alignArcs(dataframe: polars.LazyFrame, bitWidthArcCode: int, 形ArcCode: polars.DataType) -> polars.LazyFrame:
	schemaMeanders: polars.Schema = dataframe.collect_schema()
	bitsWithExtra_0b1: polars.Expr = polars.col('arcCode') // ((polars.col('arcCode') & 1) + 1)
	bitsTarget: polars.Expr = polars.lit(4, dtype=形ArcCode)
	for bitPosition in range((2 + 4 * ((bitWidthArcCode - 3) // 4)), 0, -4):
		flipExtra_0b1_Here: int = 1 << bitPosition
		bitsTarget = polars.when(
				bitPosition // 4 < (bitsWithExtra_0b1 & polars.lit(((flipExtra_0b1_Here * 4 - 1) // 3), dtype=形ArcCode)).bitwise_count_ones()
			).then(polars.lit(flipExtra_0b1_Here, dtype=形ArcCode)).otherwise(bitsTarget)
		del bitPosition, flipExtra_0b1_Here
	dataframe = dataframe.with_columns(
		(polars.col('arcCode') ^ (bitsTarget * ((polars.col('arcCode') & 1) + 1))).cast(schemaMeanders['arcCode']).alias('arcCode'))
	del bitsTarget, bitsWithExtra_0b1, schemaMeanders
	return dataframe

def transition(dataframe: polars.LazyFrame, selectorBits: int, arcCodeMaximum: int, bitWidthArcCode: int,
		materializePolars: _PolarsMaterializer, aggregation: polars.Expr | tuple[polars.Expr, ...],
		*, analysis: Callable[[polars.LazyFrame], polars.LazyFrame] = polars.LazyFrame.clone) -> polars.LazyFrame:
	schemaMeanders: polars.Schema = dataframe.collect_schema()
	bitsAlfa: polars.Expr = polars.col('arcCode') & polars.lit(selectorBits, dtype=schemaMeanders['arcCode'])
	bitsZulu: polars.Expr = (polars.col('arcCode') // 2) & polars.lit(selectorBits, dtype=schemaMeanders['arcCode'])

	selectorAddLoop: polars.Expr = polars.lit(value=True)
	selectorDraggingAlfa: polars.Expr = 1 < bitsAlfa
	selectorDraggingZulu: polars.Expr = 1 < bitsZulu
	selectorJoinAcrossRoad: polars.Expr = (
		(0 < (polars.col('arcCode') & polars.lit((selectorBits - 1), dtype=schemaMeanders['arcCode'])))
		& (0 < (polars.col('arcCode') & polars.lit(((selectorBits - 1) * 2), dtype=schemaMeanders['arcCode']))))
	形ArcCode: 形PolarsInteger = getDatatypePolars(bitWidthArcCode + 2)

	if arcCodeMaximum < (1 << (bitWidthArcCode + 2)):
		形ArcCode = getDatatypePolars(arcCodeMaximum.bit_length() - 1)
		selectorAddLoop = polars.col('arcCode') < polars.lit((arcCodeMaximum // 4), dtype=schemaMeanders['arcCode'])
		selectorDraggingAlfa &= bitsZulu < polars.lit((arcCodeMaximum // 8), dtype=schemaMeanders['arcCode'])
		selectorDraggingZulu &= bitsAlfa < polars.lit((arcCodeMaximum // 4), dtype=schemaMeanders['arcCode'])

	dataframeAnalyzed: polars.LazyFrame = materializePolars(dataframe.filter(selectorAddLoop).select(
		addLoop(polars.col('arcCode'), 形ArcCode).alias('arcCode'), polars.exclude('arcCode')).pipe(analysis), aggregation)
	del selectorAddLoop

	dataframeAnalyzed = materializePolars(dataframe.filter(selectorDraggingAlfa).select(
		dragUp(bitsAlfa, bitsZulu, 形ArcCode).alias('arcCode'), polars.exclude('arcCode')).pipe(analysis), aggregation, append=True)
	del selectorDraggingAlfa

	dataframeAnalyzed = materializePolars(dataframe.filter(selectorDraggingZulu).select(
		dragDown(bitsAlfa, bitsZulu, 形ArcCode).alias('arcCode'), polars.exclude('arcCode')).pipe(analysis), aggregation, append=True)
	del bitsAlfa, bitsZulu, selectorDraggingZulu

	dataframeAnalyzed = materializePolars(dataframe.filter(selectorJoinAcrossRoad, (polars.col('arcCode') & 0b11) == 0b00).select(
		connectArcs(polars.col('arcCode')).cast(形ArcCode).alias('arcCode'), polars.exclude('arcCode')).pipe(analysis), aggregation, append=True)
	dataframeAnalyzed = materializePolars(dataframe.filter(selectorJoinAcrossRoad, (polars.col('arcCode') & 0b11).is_between(0b01, 0b10)).pipe(
		alignArcs, bitWidthArcCode, schemaMeanders['arcCode']).select(
		connectArcs(polars.col('arcCode')).cast(形ArcCode).alias('arcCode'), polars.exclude('arcCode')).pipe(analysis), aggregation, append=True)
	del schemaMeanders, selectorJoinAcrossRoad, 形ArcCode

	return dataframeAnalyzed

def count(state: StateMeanders) -> StateMeanders:
	meandersMaximum: int = max(state.lookupMeanders.values())
	schemaMeanders: polars.Schema = polars.Schema({
		'arcCode': getDatatypePolars(state.bitWidth), 'meanders': getDatatypePolars(meandersMaximum.bit_length())})
	dataframeMeanders: polars.LazyFrame = polars.LazyFrame({
		'arcCode': polars.Series('arcCode', state.lookupMeanders.keys(), dtype=schemaMeanders['arcCode'])
		, 'meanders': polars.Series('meanders', state.lookupMeanders.values(), dtype=schemaMeanders['meanders'])}, schema=schemaMeanders)
	del schemaMeanders

	state.lookupMeanders.clear()
	with storePolars(group_by='arcCode') as materializePolars:
		while 0 < state.boundary and not integersWidePolars吗(state, meandersMaximum):
			形Meanders: 形PolarsInteger = getDatatypePolars((meandersMaximum * (state.bitWidth + 2)).bit_length())
			dataframeMeanders = dataframeMeanders.cast({
				'arcCode': getDatatypePolars(state.bitWidth)
				, 'meanders': getDatatypePolars(meandersMaximum.bit_length())})
			state.boundary -= 1
			state.set_arcCodeMaximum()
			dataframeMeanders = transition(
					dataframeMeanders, state.bitsLocator, state.arcCodeMaximum, state.bitWidth, materializePolars
				, polars.col('meanders').cast(形Meanders).sum().cast(形Meanders))
			del 形Meanders

			arcCodeMaximum, meandersMaximum = dataframeMeanders.select(polars.all().max()).collect(engine='streaming').row(0)
			state.bitWidth = arcCodeMaximum.bit_length()
			state.setBitsLocator()
			del arcCodeMaximum

		state.lookupMeanders = dict(dataframeMeanders.collect(engine='streaming').iter_rows())

	del dataframeMeanders, materializePolars, meandersMaximum
	return state

def doTheNeedful(state: StateMeanders) -> StateMeanders:
	while 0 < state.boundary:
		if integersWidePolars吗(state):
			state = countBigInt(state)
		else:
			state = count(state)
	return state
