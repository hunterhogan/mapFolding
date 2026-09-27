import polars

def addLoop(bitsAlfa: polars.Expr, bitsZulu: polars.Expr) -> polars.Expr:
    return (bitsZulu * 2 ** 1 | bitsAlfa) * 2 ** 2 | 3

def dragUp(bitsAlfa: polars.Expr, bitsZulu: polars.Expr) -> polars.Expr:
    return (bitsAlfa & 1 ^ 1) * 2 ** 1 | bitsAlfa // 2 ** 2 | bitsZulu * 2 ** 3

def dragDown(bitsAlfa: polars.Expr, bitsZulu: polars.Expr) -> polars.Expr:
    return bitsZulu & 1 ^ 1 | bitsAlfa * 2 ** 2 | bitsZulu // 2 ** 1

def connectArcs(bitsAlfa: polars.Expr, bitsZulu: polars.Expr) -> polars.Expr:
    return (bitsZulu // 2 ** 2 * 2 ** 3 | bitsAlfa) // 2 ** 2
