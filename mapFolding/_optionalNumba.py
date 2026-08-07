# DOCUMENT
from __future__ import annotations

from hunterMakesPy.parseParameters import defineConcurrencyLimit
from numba import get_num_threads, set_num_threads
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from hunterMakesPy.theTypes import Limitation

def defineProcessorLimitNumba(CPUlimit: Limitation) -> int:
	# DOCUMENT
	concurrencyLimit: int = defineConcurrencyLimit(limit=CPUlimit, cpuTotal=get_num_threads())
	set_num_threads(concurrencyLimit)
	return get_num_threads()
