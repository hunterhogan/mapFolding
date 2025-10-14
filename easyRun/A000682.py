# ruff: noqa
# pyright: basic
from mapFolding.basecamp import NOTcountingFolds
import sys
import warnings

if sys.version_info >= (3, 14):
	warnings.filterwarnings("ignore", category=FutureWarning)

def main():
	oeisID = 'A000682'
	n=45
	print(NOTcountingFolds(oeisID, n, 'matrixMeanders'))

	from mapFolding import dictionaryOEIS
	if n < dictionaryOEIS[oeisID]['valueUnknown']:
		print(dictionaryOEIS[oeisID]['valuesKnown'][n])

if __name__ == "__main__":
	main()

r"""
deactivate && C:\apps\mapFolding\.vtail\Scripts\activate.bat && title good && cls
title running && start "running" /B /HIGH /wait py -X faulthandler=0 -X tracemalloc=0 -X frozen_modules=on easyRun\A000682.py & title I'm done
"""
