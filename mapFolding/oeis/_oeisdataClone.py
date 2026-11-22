from __future__ import annotations

from mapFolding.kitFilesystem import copyTexts, removeDirectoriesEmpty
from mapFolding.kitGit import checkoutPaths, updateRepo, 工repo
from mapFolding.oeis._kitFilesystem import makeBoxOfPathFilename
from mapFolding.oeis._theSSOT import settingsOEIS
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Sequence
	from mapFolding.oeis._theTypes import OEISid

def checkoutSnapshot(boxOfOEISid: Sequence[OEISid] | None = None, path_oeisdata: Path | None = None, pathSnapshot: Path | None = None) -> Path:
	if boxOfOEISid is None:
		boxOfOEISid = settingsOEIS.boxOfIDs
	path_oeisdata = path_oeisdata or settingsOEIS.pathClone
	pathSnapshot = pathSnapshot or settingsOEIS.pathSnapshot
	with 工repo(path_oeisdata, settingsOEIS.urlRepository) as repo:
		checkoutPaths(makeBoxOfPathFilename(boxOfOEISid, PurePosixPath()), repo)
	return copyTexts(makeBoxOfPathFilename(boxOfOEISid, PurePosixPath()), path_oeisdata, pathSnapshot)

# SEMIOTICS
def updateSnapshot(boxOfOEISid: Sequence[OEISid] | None = None, path_oeisdata: Path | None = None, pathSnapshot: Path | None = None) -> Path:
	with 工repo(path_oeisdata or settingsOEIS.pathClone, settingsOEIS.urlRepository) as repo:
		updateRepo(repo)
	pathSnapshot = checkoutSnapshot(boxOfOEISid, path_oeisdata, pathSnapshot)
	removeDirectoriesEmpty(settingsOEIS.pathTerms)
	return pathSnapshot
