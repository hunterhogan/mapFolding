# TODO Replace GitPython. GitPython runs the installed git executable through subprocess, which is fragile and dangerous, and
# `import git` raises when it cannot find a git executable. The likely replacement: PyGithub reads entries and Git LFS pointers
# from `url_oeisdata` (with a GitHub token), and the oeis.org Git LFS batch API supplies b-file contents, verified against
# each pointer's SHA-256. GitHub does not serve the Git LFS objects of oeisdata: oeisdata stores them on oeis.org.
from __future__ import annotations

from functools import partial
from git import Repo
from itertools import batched, starmap
from mapFolding.oeis._oeisdata import copy_oeisdata, makeBoxOfPathFilename
from mapFolding.oeis._theSSOT import url_oeisdata
from more_itertools import consume
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Iterable, Sequence
	from mapFolding.theTypes import OEISid
	from pathlib import Path

def 工repo(path_oeisdata: Path) -> Repo:
	if path_oeisdata.exists():
		repo: Repo = Repo(path_oeisdata)
	else:
		repo = Repo.clone_from(url_oeisdata, path_oeisdata, multi_options=['--filter=blob:none', '--depth=1', '--sparse', '--no-checkout'])
		repo.git.config('lfs.fetchexclude', '')
		repo.git.sparse_checkout('set', '--no-cone', '/time.txt')
		repo.git.checkout('main')
	return repo

def checkout_oeisdata(boxOfOEISid: Iterable[OEISid], repo: Repo) -> Repo:
	consume(starmap(partial(repo.git.sparse_checkout, 'add'), batched(map(str, makeBoxOfPathFilename(boxOfOEISid, PurePosixPath('/'))), 500, strict=False)))
	return repo

def update_oeisdata(repo: Repo) -> Repo:
	repo.remotes.origin.fetch('main', depth=1)
	repo.head.reset('origin/main', index=True, working_tree=True)
	return repo

def checkoutSnapshot(boxOfOEISid: Sequence[OEISid], path_oeisdata: Path, pathSnapshot: Path) -> Path:
	checkout_oeisdata(boxOfOEISid, 工repo(path_oeisdata))
	return copy_oeisdata(makeBoxOfPathFilename(boxOfOEISid, PurePosixPath()), path_oeisdata, pathSnapshot)

def updateSnapshot(boxOfOEISid: Sequence[OEISid], path_oeisdata: Path, pathSnapshot: Path) -> Path:
	update_oeisdata(工repo(path_oeisdata))
	checkoutSnapshot(boxOfOEISid, path_oeisdata, pathSnapshot)
	return copy_oeisdata((PurePosixPath('time.txt'),), path_oeisdata, pathSnapshot)
