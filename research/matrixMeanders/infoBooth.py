# DEVELOPMENT
# ruff: file-ignore[undocumented-public-function]
# ruff: file-ignore[undocumented-public-module]
from __future__ import annotations

from mapFolding.oeis._theSSOT import pathCache  # ruff: ignore[import-private-name]
from mapFolding.theSSOT import settingsPackage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from pathlib import Path

pathResearchMatrixMeanders: Path = (settingsPackage.pathPackage / '..' / 'research' / 'matrixMeanders').resolve()
pathData: Path = settingsPackage.pathPackage / 'oeis' / 'A400429'
pathFormulasTriangle: Path = pathData
pathArchiveTriangle: Path = pathResearchMatrixMeanders / 'archiveTriangle'

filenameTriangleSemiText: str = 'triangleSemi.txt'
filenameTriangleSemiCommaSeparatedValues: str = 'A400429.csv'
filenameTriangleSemiSubmissionOEIS: str = 'A400429.txt'
filenameFormulaA005315: str = '_A005315.py'
filenameFormulaA005315PolynomialCyclic: str = 'A005315PolynomialCyclic.py'
filenameArchiveA005315Binomial: str = 'A005315Binomial.py'

pathFilenameTriangleSemiText: Path = pathData / filenameTriangleSemiText
pathFilenameTriangleSemiCommaSeparatedValues: Path = pathData / filenameTriangleSemiCommaSeparatedValues
pathFilenameTriangleSemiSubmissionOEIS: Path = pathCache / filenameTriangleSemiSubmissionOEIS
pathFilenameFormulaA005315: Path = pathFormulasTriangle / filenameFormulaA005315
pathFilenameFormulaA005315PolynomialCyclic: Path = pathArchiveTriangle / filenameFormulaA005315PolynomialCyclic
pathFilenameArchiveA005315Binomial: Path = pathArchiveTriangle / filenameArchiveA005315Binomial

filenameTriangleSemi: str = filenameTriangleSemiText
filenameCSV: str = filenameTriangleSemiCommaSeparatedValues

def makeFilenameDiagonal(次diagonal: int) -> str:
	return f'diagonal{次diagonal}.csv'

def makePathFilenameDiagonal(次diagonal: int) -> Path:
	return pathData / makeFilenameDiagonal(次diagonal)
