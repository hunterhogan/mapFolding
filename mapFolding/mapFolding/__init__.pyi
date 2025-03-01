from theDao import *
from theSSOT import *
from beDRY import *
from basecamp import *
from oeis import *

_dictionaryListsImportFrom: dict[str, list[str]]

def __getattr__(name: str): ...

_mapSymbolToModule: dict[str, str]
