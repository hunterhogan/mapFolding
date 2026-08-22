"""Make docstrings."""
from __future__ import annotations

from astToolkit import Be, Grab, IfThis, Make, NodeChanger, NodeTourist, Then
from astToolkit.changeDef import makeDictionaryFunctionDef
from astToolkit.filesystem import parsePathFilename2astModule
from astToolkit.transformationTools import unjoinBinOP
from hunterMakesPy import raiseIfNone
from hunterMakesPy.filesystemToolkit import writePython
from mapFolding.oeis import getMetadata
from typing import TYPE_CHECKING
import ast

if TYPE_CHECKING:
    from mapFolding.oeis._dataBaskets import MetadataOEISid
    from pathlib import Path

#------------------ General Settings ----------------------------------------------------------------------------------
sourcePrefix: str = 'zCuzDocStoopid'

moduleWarning = "This is a generated file; edit the source file.\n"

def transformOEISidByFormula(pathFilenameSource: Path) -> Path:
    """Transform the docstrings of functions corresponding to OEIS sequences."""
    pathFilenameWrite: Path = pathFilenameSource.with_stem(pathFilenameSource.stem.removeprefix(sourcePrefix))
    astModule: ast.Module = parsePathFilename2astModule(pathFilenameSource)
    dictionaryFunctionDef: dict[str, ast.FunctionDef] = makeDictionaryFunctionDef(astModule)

    for oeisID, FunctionDef in dictionaryFunctionDef.items():
        if oeisID.startswith('A') and len(oeisID) == 7 and oeisID[1:7].isdigit():
            boxOf_f_astConstant: list[ast.expr] = []
            NodeTourist(Be.MatchValue, Grab.valueAttribute(Then.appendTo(boxOf_f_astConstant))).visit(FunctionDef)
            if 1 == len(boxOf_f_astConstant):
                slice_ast_expr: ast.expr = boxOf_f_astConstant[0]
            else:
                slice_ast_expr = Make.Tuple(boxOf_f_astConstant)

            boxOf_arg: list[ast.expr] = unjoinBinOP(FunctionDef.args.args[1], ast.BitOr)
            boxOf_arg.reverse()
            boxOf_arg.insert(0, Make.Subscript(Make.Name('Literal'), slice=slice_ast_expr))
            NodeChanger(Be.arg.argIs('f'.__eq__), Grab.annotationAttribute(Then.replaceWith(Make.BitOr.join(boxOf_arg)))).visit(FunctionDef)

            boxOf_f: list[str] = list(map(ast.literal_eval, boxOf_f_astConstant))
            functionOf: str = ' or '.join(boxOf_f)
            metadata: MetadataOEISid = getMetadata(oeisID)

            ImaDocstring: str = f"""
    Compute {oeisID}(n) as a function of {functionOf}.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of {oeisID} is: "{metadata['description']}"

    The domain of {oeisID} starts at {metadata['offset']}, therefore for values of `n` < {metadata['offset']}, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is {metadata['valueUnknown']}.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.

    Returns
    -------
    a(n) : int
        {metadata['description']}

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/{oeisID}
    """

            astExprDocstring: ast.Expr = Make.Expr(Make.Constant(ImaDocstring))

            NodeChanger(IfThis.isFunctionDefIdentifier(oeisID)
                , Grab.bodyAttribute(Grab.index(0, Then.insertThisAbove([astExprDocstring])))
            ).visit(astModule)

    ast.fix_missing_locations(astModule)

    docstringModule: str = raiseIfNone(ast.get_docstring(astModule))
    moduleAsString: str = ast.unparse(astModule)
    moduleAsString = moduleAsString.replace(docstringModule, docstringModule + "\n\n" + moduleWarning)

    return writePython(moduleAsString, pathFilenameWrite)
