from basecamp import *
from beDRY import *
from oeis import *
from theDao import *
from theSSOT import *
from theSSOTdatatypes import *

_dictionaryListsImportFrom: dict[str, list[str]]

def __getattr__(name: str): ...

_mapSymbolToModule: dict[str, str]
