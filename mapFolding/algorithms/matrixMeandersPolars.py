from __future__ import annotations

from itertools import chain
from mapFolding.dataStructures import getDatatypePolars
from mapFolding.kitFilesystem import storePolars
from mapFolding.synthesized.matrixMeanders.bigIntPolars import countBigInt
from mapFolding.synthesized.matrixMeanders.polarsWide import integersWidePolars吗
from typing import TYPE_CHECKING
import polars

if TYPE_CHECKING:
	from mapFolding.dataBaskets import StateMeanders
	from mapFolding.theTypes import 形PolarsInteger

def addLoop(arcCode: polars.Expr, datatypeArcCode: 形PolarsInteger) -> polars.Expr:
	return (arcCode.cast(datatypeArcCode) * 4) | 3

def dragUp(bitsAlfa: polars.Expr, bitsZulu: polars.Expr, datatypeArcCode: 形PolarsInteger) -> polars.Expr:
	# This has a stack. It is probably impossible to eliminate it, so don't multiply it with more stacks.
	return (bitsZulu.cast(datatypeArcCode) * 8) | (bitsAlfa // 4).cast(datatypeArcCode) | (((bitsAlfa & 1) ^ 1) * 2).cast(polars.UInt8)

def dragDown(bitsAlfa: polars.Expr, bitsZulu: polars.Expr, datatypeArcCode: 形PolarsInteger) -> polars.Expr:
	# This has a stack. It is probably impossible to eliminate it, so don't multiply it with more stacks.
	return (bitsAlfa.cast(datatypeArcCode) * 4) | (bitsZulu // 2).cast(datatypeArcCode) | ((bitsZulu & 1) ^ 1).cast(polars.UInt8)

def connectArcs(arcCode: polars.Expr) -> polars.Expr:
	return arcCode // 4

def alignArcs(dataframeMeanders: polars.LazyFrame, bitWidth: int, datatypeArcCode: polars.DataType) -> polars.LazyFrame:
	bitsWithExtra_0b1: polars.Expr = polars.col('arcCode') // ((polars.col('arcCode') & 1) + 1)
	bitPosition: int = 2 + 4 * ((bitWidth - 3) // 4)
	bitsTarget: polars.Expr = polars.lit(4, dtype=datatypeArcCode)
	while 0 < bitPosition:
		flipExtra_0b1_Here: int = 1 << bitPosition
		bitsTarget = polars.when(
			bitPosition // 4 < (bitsWithExtra_0b1 & polars.lit((flipExtra_0b1_Here * 4 - 1) // 3, dtype=datatypeArcCode)).bitwise_count_ones()
		).then(polars.lit(flipExtra_0b1_Here, dtype=datatypeArcCode)).otherwise(bitsTarget)
		bitPosition -= 4
	return dataframeMeanders.with_columns(
		(polars.col('arcCode') ^ (bitsTarget * ((polars.col('arcCode') & 1) + 1))).alias('arcCode'))

def transition(dataframeMeanders: polars.LazyFrame, bitsLocator: int, arcCodeMAXIMUM: int, bitWidth: int) -> polars.LazyFrame:
	schemaMeanders: polars.Schema = dataframeMeanders.collect_schema()
	bitsAlfa: polars.Expr = polars.col('arcCode') & polars.lit(bitsLocator, dtype=schemaMeanders['arcCode'])
	bitsZulu: polars.Expr = (polars.col('arcCode') // 2) & polars.lit(bitsLocator, dtype=schemaMeanders['arcCode'])
	datatypeArcCode: 形PolarsInteger = getDatatypePolars(bitWidth + 2)
	dataframeLoop: polars.LazyFrame = dataframeMeanders
	dataframeDraggingAlfa: polars.LazyFrame = dataframeMeanders.filter(1 < bitsAlfa)
	dataframeDraggingZulu: polars.LazyFrame = dataframeMeanders.filter(1 < bitsZulu)
	dataframeConnecting: polars.LazyFrame = dataframeMeanders.filter(1 < bitsAlfa, 1 < bitsZulu)
	if arcCodeMAXIMUM < 1 << (bitWidth + 2):
		datatypeArcCode = getDatatypePolars(arcCodeMAXIMUM.bit_length() - 1)
		dataframeLoop = dataframeLoop.filter(polars.col('arcCode') < polars.lit(arcCodeMAXIMUM // 4, dtype=schemaMeanders['arcCode']))
		dataframeDraggingAlfa = dataframeDraggingAlfa.filter(bitsZulu < polars.lit(arcCodeMAXIMUM // 8, dtype=schemaMeanders['arcCode']))
		dataframeDraggingZulu = dataframeDraggingZulu.filter(bitsAlfa < polars.lit(arcCodeMAXIMUM // 4, dtype=schemaMeanders['arcCode']))
	return polars.concat((
		dataframeLoop.select(addLoop(polars.col('arcCode'), datatypeArcCode).alias('arcCode'), 'meanders')
		, dataframeDraggingAlfa.select(dragUp(bitsAlfa, bitsZulu, datatypeArcCode).alias('arcCode'), 'meanders')
		, dataframeDraggingZulu.select(dragDown(bitsAlfa, bitsZulu, datatypeArcCode).alias('arcCode'), 'meanders')
		, dataframeConnecting.filter((polars.col('arcCode') & 3) == 0)
			.select(connectArcs(polars.col('arcCode')).cast(datatypeArcCode).alias('arcCode'), 'meanders')
		, dataframeConnecting.filter((polars.col('arcCode') & 3).is_between(1, 2))
			.pipe(alignArcs, bitWidth, schemaMeanders['arcCode'])
			.select(connectArcs(polars.col('arcCode')).cast(datatypeArcCode).alias('arcCode'), 'meanders')
		), parallel=False, rechunk=False)

def count(state: StateMeanders) -> StateMeanders:
	meandersMaximum: int = max(state.lookupMeanders.values())
	dataframeMeanders: polars.LazyFrame = polars.LazyFrame({
		'arcCode': polars.Series(state.lookupMeanders.keys(), dtype=getDatatypePolars(state.bitWidth))
		, 'meanders': polars.Series(state.lookupMeanders.values(), dtype=getDatatypePolars(meandersMaximum.bit_length()))})
	state.lookupMeanders.clear()
	with storePolars(group_by='arcCode') as materializePolars:
		while 0 < state.boundary and not integersWidePolars吗(state, meandersMaximum):
			datatypeMeanders: 形PolarsInteger = getDatatypePolars((meandersMaximum * (state.bitWidth + 2)).bit_length())
			dataframeMeanders = dataframeMeanders.cast({
				'arcCode': getDatatypePolars(state.bitWidth)
				, 'meanders': getDatatypePolars(meandersMaximum.bit_length())})
			state.boundary -= 1
			state.set_arcCodeMAXIMUM()
			dataframeMeanders = materializePolars(
				transition(dataframeMeanders, state.bitsLocator, state.arcCodeMAXIMUM, state.bitWidth)
				, polars.col('meanders').cast(datatypeMeanders).sum().cast(datatypeMeanders))
			arcCodeMaximum, meandersMaximum = dataframeMeanders.select(polars.all().max()).collect(engine='streaming').row(0)
			state.bitWidth = arcCodeMaximum.bit_length()
			state.setBitsLocator()
		state.lookupMeanders = dict(chain.from_iterable(map(polars.DataFrame.iter_rows,
			dataframeMeanders.collect_batches(maintain_order=False, engine='streaming'))))
	return state

def doTheNeedful(state: StateMeanders) -> StateMeanders:
	while 0 < state.boundary:
		if integersWidePolars吗(state):
			state = countBigInt(state)
		else:
			state = count(state)
	return state
