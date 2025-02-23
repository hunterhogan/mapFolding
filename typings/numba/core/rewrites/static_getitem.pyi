from _typeshed import Incomplete
from numba.core import errors as errors, ir as ir, types as types
from numba.core.rewrites import Rewrite as Rewrite, register_rewrite as register_rewrite

class RewriteConstGetitems(Rewrite):
    """
    Rewrite IR expressions of the kind `getitem(value=arr, index=$constXX)`
    where `$constXX` is a known constant as
    `static_getitem(value=arr, index=<constant value>)`.
    """
    getitems: Incomplete
    block: Incomplete
    def match(self, func_ir, block, typemap, calltypes): ...
    def apply(self):
        """
        Rewrite all matching getitems as static_getitems.
        """

class RewriteStringLiteralGetitems(Rewrite):
    """
    Rewrite IR expressions of the kind `getitem(value=arr, index=$XX)`
    where `$XX` is a StringLiteral value as
    `static_getitem(value=arr, index=<literal value>)`.
    """
    getitems: Incomplete
    block: Incomplete
    calltypes: Incomplete
    def match(self, func_ir, block, typemap, calltypes):
        """
        Detect all getitem expressions and find which ones have
        string literal indexes
        """
    def apply(self):
        """
        Rewrite all matching getitems as static_getitems where the index
        is the literal value of the string.
        """

class RewriteStringLiteralSetitems(Rewrite):
    """
    Rewrite IR expressions of the kind `setitem(value=arr, index=$XX, value=)`
    where `$XX` is a StringLiteral value as
    `static_setitem(value=arr, index=<literal value>, value=)`.
    """
    setitems: Incomplete
    block: Incomplete
    calltypes: Incomplete
    def match(self, func_ir, block, typemap, calltypes):
        """
        Detect all setitem expressions and find which ones have
        string literal indexes
        """
    def apply(self):
        """
        Rewrite all matching setitems as static_setitems where the index
        is the literal value of the string.
        """

class RewriteConstSetitems(Rewrite):
    """
    Rewrite IR statements of the kind `setitem(target=arr, index=$constXX, ...)`
    where `$constXX` is a known constant as
    `static_setitem(target=arr, index=<constant value>, ...)`.
    """
    setitems: Incomplete
    block: Incomplete
    def match(self, func_ir, block, typemap, calltypes): ...
    def apply(self):
        """
        Rewrite all matching setitems as static_setitems.
        """
