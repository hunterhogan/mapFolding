# ruff: noqa
# pyright: basic
from mapFolding.basecamp import NOTcountingFolds
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

n=25
print(NOTcountingFolds('A005316', n))

from mapFolding import dictionaryOEIS

if n < dictionaryOEIS['A005316']['valueUnknown']:
	print(dictionaryOEIS['A005316']['valuesKnown'][n])


r"""
deactivate && C:\apps\mapFolding\.vtail\Scripts\activate.bat && title good && cls

"""
