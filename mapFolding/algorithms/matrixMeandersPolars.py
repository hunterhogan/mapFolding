from __future__ import annotations

from mapFolding.algorithms.matrixMeanders import count as countUnbounded
from mapFolding.dataStructures import makeDataContainerPolars
from mapFolding.synthesized.matrixMeanders.polarsTransitions import addLoop, connectArcs, dragDown, dragUp
from mapFolding.theTypes import getDatatypePolars
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING
import polars

if TYPE_CHECKING:
	from mapFolding.dataBaskets import StateMeanders

def alignArcs(dataframeMeanders: polars.LazyFrame, bitWidth: int, datatypeArcCode: polars.DataType) -> polars.LazyFrame:
	dataframeMeanders = dataframeMeanders.with_columns(
		polars.when((polars.col('arcCode') & 3) == 0).then(-2).otherwise(0).cast(polars.Int8).alias('findTheExtra_0b1'))
	flipExtra_0b1_Here: int = 4
	while flipExtra_0b1_Here < 1 << bitWidth:
		bitsTarget: polars.Expr = polars.lit(flipExtra_0b1_Here, dtype=datatypeArcCode) * ((polars.col('arcCode') & 1) + 1)
		dataframeMeanders = dataframeMeanders.with_columns(
			polars.when(0 <= polars.col('findTheExtra_0b1')).then(
				polars.col('findTheExtra_0b1')
				+ polars.when((polars.col('arcCode') & bitsTarget) == 0).then(polars.lit(1, dtype=polars.Int8))
				.otherwise(polars.lit(-1, dtype=polars.Int8)))
			.otherwise(polars.lit(-2, dtype=polars.Int8)).alias('findTheExtra_0b1')
		).with_columns(
			polars.when(polars.col('findTheExtra_0b1') == -1).then(polars.col('arcCode') ^ bitsTarget)
			.otherwise(polars.col('arcCode')).alias('arcCode'))
		flipExtra_0b1_Here <<= 2
	return dataframeMeanders.drop('findTheExtra_0b1')

def count(dataframeMeanders: polars.LazyFrame, bitsLocator: int, arcCodeMAXIMUM: int, bitWidth: int) -> polars.LazyFrame:
	schemaMeanders: polars.Schema = dataframeMeanders.collect_schema()
	bitsAlfa: polars.Expr = polars.col('arcCode') & polars.lit(bitsLocator, dtype=schemaMeanders['arcCode'])
	bitsZulu: polars.Expr = polars.col('arcCode') // 2 & polars.lit(bitsLocator, dtype=schemaMeanders['arcCode'])
	return polars.concat((
		dataframeMeanders.select(addLoop(bitsAlfa, bitsZulu).alias('arcCode'), 'meanders')
		, dataframeMeanders.filter(1 < bitsAlfa).select(dragUp(bitsAlfa, bitsZulu).alias('arcCode'), 'meanders')
		, dataframeMeanders.filter(1 < bitsZulu).select(dragDown(bitsAlfa, bitsZulu).alias('arcCode'), 'meanders')
		, dataframeMeanders.filter(1 < bitsAlfa, 1 < bitsZulu, (polars.col('arcCode') & 3) != 3)
			.pipe(alignArcs, bitWidth, schemaMeanders['arcCode'])
			.select(connectArcs(bitsAlfa, bitsZulu).alias('arcCode'), 'meanders')
		), parallel=False, rechunk=False
	).filter(polars.col('arcCode') <= polars.lit(arcCodeMAXIMUM - 1, dtype=polars.UInt128)
	).group_by('arcCode').agg(polars.col('meanders').sum().cast(schemaMeanders['meanders']))

def doTheNeedful(state: StateMeanders) -> StateMeanders:
	meandersMaximum: int = max(state.lookupMeanders.values())
	if max(state.bitWidth + 3, (meandersMaximum * (state.bitWidth + 4)).bit_length()) <= 128:
		dataframeMeanders: polars.LazyFrame = polars.LazyFrame({
			'arcCode': polars.Series(state.lookupMeanders.keys(), dtype=getDatatypePolars(state.bitWidth))
			, 'meanders': polars.Series(state.lookupMeanders.values(), dtype=getDatatypePolars(meandersMaximum.bit_length()))})
		state.lookupMeanders.clear()
		with TemporaryDirectory(prefix='matrixMeandersPolars') as pathScratch:
			pathFilenamePrevious: Path | None = None
			while 0 < state.boundary and max(state.bitWidth + 3, (meandersMaximum * (state.bitWidth + 4)).bit_length()) <= 128:
				dataframeMeanders = dataframeMeanders.cast({
					'arcCode': getDatatypePolars(state.bitWidth + 3)
					, 'meanders': getDatatypePolars((meandersMaximum * (state.bitWidth + 4)).bit_length())})
				state.boundary -= 1
				state.set_arcCodeMAXIMUM()
				pathFilename: Path = Path(pathScratch, f'{state.boundary}.arrow')
				dataframeMeanders = makeDataContainerPolars(
					count(dataframeMeanders, state.bitsLocator, min(state.arcCodeMAXIMUM, 1 << 128), state.bitWidth), pathFilename)
				if pathFilenamePrevious is not None:
					pathFilenamePrevious.unlink()
				pathFilenamePrevious = pathFilename
				arcCodeMaximum, meandersMaximum = dataframeMeanders.select(polars.all().max()).collect(engine='streaming').row(0)
				state.bitWidth = arcCodeMaximum.bit_length()
				state.setBitsLocator()
			state.lookupMeanders = dict(dataframeMeanders.collect(engine='streaming').iter_rows())
	if 0 < state.boundary:
		state = countUnbounded(state)
	return state
