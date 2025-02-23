def parse_signature(sig):
    """Parse generalized ufunc signature.

    NOTE: ',' (COMMA) is a delimiter; not separator.
          This means trailing comma is legal.
    """
