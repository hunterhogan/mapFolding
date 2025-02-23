from _typeshed import Incomplete
from pygments.lexer import RegexLexer

msg: str

class NumbaIRLexer(RegexLexer):
    """
    Pygments style lexer for Numba IR (for use with highlighting etc).
    """
    name: str
    aliases: Incomplete
    filenames: Incomplete
    identifier: str
    fun_or_var: str
    tokens: Incomplete

def by_colorscheme():
    """
    Get appropriate style for highlighting according to
    NUMBA_COLOR_SCHEME setting
    """
