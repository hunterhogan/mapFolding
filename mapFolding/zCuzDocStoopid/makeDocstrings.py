"""Make docstrings."""
from astToolkit import IfThis, Make, NodeChanger, parsePathFilename2astModule
from astToolkit.transformationTools import makeDictionaryFunctionDef
from hunterMakesPy import raiseIfNone, writeStringToHere
from mapFolding import dictionaryOEISMapFolding, dictionaryOEISMeanders, packageSettings
from pathlib import Path
import ast

sourcePrefix: str = 'zCuzDocStoopid'

pathRoot: Path = packageSettings.pathPackage / "algorithms"

pathFilenameSource: Path = next(iter(pathRoot.glob(f"{sourcePrefix}*.py"))).absolute()
pathFilenameWrite: Path = pathFilenameSource.with_stem(pathFilenameSource.stem.removeprefix(sourcePrefix))

astModule: ast.Module = parsePathFilename2astModule(pathFilenameSource)
dictionaryFunctionDef: dict[str, ast.FunctionDef] = makeDictionaryFunctionDef(astModule)

moduleWarning = """
NOTE: This is a generated file; edit the source file.
"""

oeisID = 'Error during transformation' # `ast.FunctionDef.name` of function in `pathFilenameSource`.
functionOf: str = 'Error during transformation' # The value of `functionOf` is in the docstring of function `oeisID` in `pathFilenameSource`.

for oeisID, FunctionDef in dictionaryFunctionDef.items():
    dictionaryOEIS = dictionaryOEISMapFolding if oeisID in dictionaryOEISMapFolding else dictionaryOEISMeanders
    functionOf = raiseIfNone(ast.get_docstring(FunctionDef))

    ImaDocstring= 	f"""
    Compute {oeisID}(n) as a function of {functionOf}.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) says {oeisID} is: "{dictionaryOEIS[oeisID]['description']}"

    The domain of {oeisID} starts at {dictionaryOEIS[oeisID]['offset']}, therefore for values of `n` < {dictionaryOEIS[oeisID]['offset']}, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is {dictionaryOEIS[oeisID]['valueUnknown']}.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.

    Returns
    -------
    a(n) : int
        {dictionaryOEIS[oeisID]['description']}

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/{oeisID}
    """

    astExprDocstring = Make.Expr(Make.Constant(ImaDocstring))

    findThis = IfThis.isFunctionDefIdentifier(oeisID)
    def _doThat(node: ast.FunctionDef) -> ast.AST:
        node.body[0] = astExprDocstring
        return node

    # doThat = Grab.bodyAttribute(lambda body: Then.replaceWith(astExprDocstring)((body[0])))  # noqa: ERA001

    NodeChanger(findThis, _doThat).visit(astModule)

ast.fix_missing_locations(astModule)

writeStringToHere(ast.unparse(astModule), pathFilenameWrite)
