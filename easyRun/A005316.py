# ruff: noqa
# pyright: basic
from mapFolding.basecamp import A005316
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

n=25
print(A005316(n))

from mapFolding import dictionaryOEIS

if n < dictionaryOEIS['A005316']['valueUnknown']:
	print(dictionaryOEIS['A005316']['valuesKnown'][n])


r"""
deactivate && C:\apps\mapFolding\.vtail\Scripts\activate.bat && title good && cls
title running && py Z0Z_A005316.py && title I'm done || title Error

"""
