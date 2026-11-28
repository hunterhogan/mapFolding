
from __future__ import annotations

from mapFolding.oeis._termsA400429 import writeTriangleRows

if __name__ == '__main__':
	flow = 'rows'
	if flow == 'rows':
		nStop: int = 47
		writeTriangleRows(nStop)

"""
source /home/hunte/mapFolding/.venv/bin/activate && cd mapFolding
sudo nice -n -10 /home/hunte/mapFolding/.venv/bin/python -X faulthandler=0 -X tracemalloc=0 -X frozen_modules=on /home/hunte/mapFolding/easyRun/triangleSemiMake.py
"""
