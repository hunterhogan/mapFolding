from mapFolding.dataBaskets import MapFoldingState
from mapFolding.oeis import dictionaryOEIS
import sys
import time

def activeLeafGreaterThan0(state: MapFoldingState) -> bool:
	return state.leaf1ndex > 0

def activeLeafGreaterThanLeavesTotal(state: MapFoldingState) -> bool:
	return state.leaf1ndex > state.leavesTotal

def activeLeafIsTheFirstLeaf(state: MapFoldingState) -> bool:
	return state.leaf1ndex <= 1

def activeLeafIsUnconstrainedInAllDimensions(state: MapFoldingState) -> bool:
	return not state.dimensionsUnconstrained

def activeLeafUnconstrainedInThisDimension(state: MapFoldingState) -> MapFoldingState:
	state.dimensionsUnconstrained -= 1
	return state

def filterCommonGaps(state: MapFoldingState) -> MapFoldingState:
	state.gapsWhere[state.gap1ndex] = state.gapsWhere[state.indexMiniGap]
	if state.countDimensionsGapped[state.gapsWhere[state.indexMiniGap]] == state.dimensionsUnconstrained:
		state = incrementActiveGap(state)
	state.countDimensionsGapped[state.gapsWhere[state.indexMiniGap]] = 0
	return state

def gapAvailable(state: MapFoldingState) -> bool:
	return state.leaf1ndex > 0

def incrementActiveGap(state: MapFoldingState) -> MapFoldingState:
	state.gap1ndex += 1
	return state

def incrementGap1ndexCeiling(state: MapFoldingState) -> MapFoldingState:
	state.gap1ndexCeiling += 1
	return state

def incrementIndexMiniGap(state: MapFoldingState) -> MapFoldingState:
	state.indexMiniGap += 1
	return state

def initializeIndexMiniGap(state: MapFoldingState) -> MapFoldingState:
	state.indexMiniGap = state.gap1ndex
	return state

def initializeVariablesToFindGaps(state: MapFoldingState) -> MapFoldingState:
	state.dimensionsUnconstrained = state.dimensionsTotal
	state.gap1ndexCeiling = state.gapRangeStart[state.leaf1ndex - 1]
	state.indexDimension = 0
	return state

def insertActiveLeaf(state: MapFoldingState) -> MapFoldingState:
	state.indexLeaf = 0
	while state.indexLeaf < state.leaf1ndex:
		state.gapsWhere[state.gap1ndexCeiling] = state.indexLeaf
		state.gap1ndexCeiling += 1
		state.indexLeaf += 1
	return state

def insertActiveLeafAtGap(state: MapFoldingState) -> MapFoldingState:
	state.gap1ndex -= 1
	state.leafAbove[state.leaf1ndex] = state.gapsWhere[state.gap1ndex]
	state.leafBelow[state.leaf1ndex] = state.leafBelow[state.leafAbove[state.leaf1ndex]]
	state.leafBelow[state.leafAbove[state.leaf1ndex]] = state.leaf1ndex
	state.leafAbove[state.leafBelow[state.leaf1ndex]] = state.leaf1ndex
	state.gapRangeStart[state.leaf1ndex] = state.gap1ndex
	state.leaf1ndex += 1
	return state

def leafBelowSentinelIs1(state: MapFoldingState) -> bool:
	return state.leafBelow[0] == 1

def leafConnecteeIsActiveLeaf(state: MapFoldingState) -> bool:
	return state.leafConnectee == state.leaf1ndex

def lookForGaps(state: MapFoldingState) -> MapFoldingState:
	state.gapsWhere[state.gap1ndexCeiling] = state.leafConnectee
	if state.countDimensionsGapped[state.leafConnectee] == 0:
		state = incrementGap1ndexCeiling(state)
	state.countDimensionsGapped[state.leafConnectee] += 1
	return state

def lookupLeafConnecteeInConnectionGraph(state: MapFoldingState) -> MapFoldingState:
	state.leafConnectee = state.connectionGraph[state.indexDimension, state.leaf1ndex, state.leaf1ndex]
	return state

def loopingLeavesConnectedToActiveLeaf(state: MapFoldingState) -> bool:
	return state.leafConnectee != state.leaf1ndex

def loopingThroughTheDimensions(state: MapFoldingState) -> bool:
	return state.indexDimension < state.dimensionsTotal

def loopingToActiveGapCeiling(state: MapFoldingState) -> bool:
	return state.indexMiniGap < state.gap1ndexCeiling

def noGapsHere(state: MapFoldingState) -> bool:
	return (state.leaf1ndex > 0) and (state.gap1ndex == state.gapRangeStart[state.leaf1ndex - 1])

def tryAnotherLeafConnectee(state: MapFoldingState) -> MapFoldingState:
	state.leafConnectee = state.connectionGraph[state.indexDimension, state.leaf1ndex, state.leafBelow[state.leafConnectee]]
	return state

def tryNextDimension(state: MapFoldingState) -> MapFoldingState:
	state.indexDimension += 1
	return state

def undoLastLeafPlacement(state: MapFoldingState) -> MapFoldingState:
	state.leaf1ndex -= 1
	state.leafBelow[state.leafAbove[state.leaf1ndex]] = state.leafBelow[state.leaf1ndex]
	state.leafAbove[state.leafBelow[state.leaf1ndex]] = state.leafAbove[state.leaf1ndex]
	return state

"""src/irvine/oeis/a007/A007822.java
package irvine.oeis.a007;

import irvine.math.z.Z;
import irvine.oeis.a001.A001415;

/**
 * A007822 Number of symmetric foldings of 2n+1 stamps.
 * @author Sean A. Irvine (Java port)
 */
public class A007822 extends A001415 {

  /** Construct the sequence. */
  public A007822() {
	super(1);
  }

  private int mN = 1;
  private long mCount = 0;

  private boolean isSymmetric(final int[] c, final int delta) {
	for (int k = 0; k < (c.length - 1) / 2; ++k) {
	  if (c[(delta + k) % c.length] != c[(delta + c.length - 2 - k) % c.length]) {
		return false;
	  }
	}
	return true;
  }

  @Override
  protected void process(final int[] a, final int[] b, final int n) {
	final int[] c = new int[a.length];
	int j = 0;
	for (int k = 0; k < b.length; k++) {
	  c[k] = b[j] - j;
	  j = b[j];
	}
	for (int k = 0; k < a.length; ++k) {
	  if (isSymmetric(c, k)) {
		++mCount;
	  }
	}
  }
  protected void process(final int[] a, final int[] b, final int n) {
	final int[] c = new int[a.length];
	int j = 0;
	for (int k = 0; k < b.length; k++) {
	  c[k] = b[j] - j;
	  j = b[j];
	}
	for (int k = 0; k < a.length; ++k) {
	  if (isSymmetric(c, k)) {
		++mCount;
	  }
	}
  }

  @Override
  public Z next() {
	mN += 2;
	mCount = 0;
	foldings(new int[] {mN - 1}, true, 0, 0);
	return Z.valueOf((mCount + 1) / 2);
  }
}
"""  # noqa: E101

"""NOTE Temporary identifiers during porting:

a
b
c
delta
foldings
isSymmetric
j
k
mCount
n
process

"""

def isSymmetric(c: list[int], delta: int) -> bool:
	k = 0
	while k < (len(c) - 1) // 2: # NOTE in a `while` loop, the comparator can be a float. Important?
		if (c[(delta + k) % len(c)] != c[(delta + len(c) - 2 - k) % len(c)]):
			return False
		k += 1
	return True

def process(a: list[int], b: list[int], n: int) -> int:  # noqa: ARG001
	mCount = 0
	c: list[int] = [0] * len(a)
	j = 0

	k = 0
	while (k < len(b)):
		c[k] = b[j] - j
		j = b[j]
		k += 1

	k = 0
	while (k < len(a)):
		if (isSymmetric(c, k)):
			mCount += 1
		k += 1

	return mCount

def foldings(state: MapFoldingState) -> int:
	mCount: int = 0
	while activeLeafGreaterThan0(state):
		if activeLeafIsTheFirstLeaf(state) or leafBelowSentinelIs1(state):
			if activeLeafGreaterThanLeavesTotal(state):
				a: list[int] = state.leafAbove.tolist()
				b: list[int] = state.leafBelow.tolist()
				n = int(state.leavesTotal)
				mCount += process(a, b, n)
			else:
				state = initializeVariablesToFindGaps(state)
				while loopingThroughTheDimensions(state):
					state = lookupLeafConnecteeInConnectionGraph(state)
					if leafConnecteeIsActiveLeaf(state):
						state = activeLeafUnconstrainedInThisDimension(state)
					else:
						while loopingLeavesConnectedToActiveLeaf(state):
							state = lookForGaps(state)
							state = tryAnotherLeafConnectee(state)
					state = tryNextDimension(state)
				if activeLeafIsUnconstrainedInAllDimensions(state):
					state = insertActiveLeaf(state)
				state = initializeIndexMiniGap(state)
				while loopingToActiveGapCeiling(state):
					state = filterCommonGaps(state)
					state = incrementIndexMiniGap(state)
		while noGapsHere(state):
			state = undoLastLeafPlacement(state)
		if gapAvailable(state):
			state = insertActiveLeafAtGap(state)
	return mCount

def doTheNeedful(state: MapFoldingState) -> int:
	mCount: int = foldings(state)
	"""NOTE

	```java
	mN += 2;
	mCount = 0;
	foldings(new int[] {mN - 1}, true, 0, 0);
	return Z.valueOf((mCount + 1) / 2);
	```

	I am deeply suspicious of `(mCount + 1) / 2`.
	`mN - 1` and `mCount + 1` might cancel each other out.
	But `mN += 2` is not obviously an inversion of `... / 2`.

	"""
	return (mCount + 1) // 2

if __name__ == '__main__':
	for n in range(2, 7):
		mapShape = dictionaryOEIS['A007822']['getMapShape'](n)
		state = MapFoldingState(mapShape)
		timeStart = time.perf_counter()
		foldsTotal = doTheNeedful(state)
		sys.stdout.write(f"{mapShape = } {foldsTotal == dictionaryOEIS['A007822']['valuesKnown'][n]} {n = } {foldsTotal = } {dictionaryOEIS['A007822']['valuesKnown'][n]} {time.perf_counter() - timeStart:.2f}\n")
