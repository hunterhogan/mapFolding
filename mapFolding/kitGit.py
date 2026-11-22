"""Clone, update, and check out selected files of a remote Git repository, and resolve Git LFS pointers.

(AI generated docstring)

You can use this module to keep a sparse local clone of a remote repository without a Git
installation. A clone has one commit and no file contents, so a check out fetches only the files
that the caller selects.

Contents
--------
Functions
	checkoutPaths
		Fetch the selected files, resolve Git LFS pointers, and write the files to the working tree.
	getObjectIDsWanted
		Select the objects that a fetch requests, for use as `determine_wants`.
	readTextGit
		Read the text of an object, resolving a Git LFS pointer.
	updateRepo
		Fetch the latest commit of the `main` branch without file contents.
	工repo
		Open the clone of a repository, cloning the repository when the clone is absent.
"""
from __future__ import annotations

from contextlib import closing
from dulwich.client import get_transport_and_path
from dulwich.config import ConfigFile
from dulwich.lfs import LFSFilterDriver, LFSPointer, LFSStore
from dulwich.object_store import iter_tree_contents
from dulwich.objectspec import parse_tree
from dulwich.refs import Ref
from dulwich.repo import MemoryRepo, Repo
from functools import partial
from humpy_cytoolz import compose_left
from io import BytesIO
from itertools import filterfalse, repeat
from mapFolding.kitFilesystem import writeText
from more_itertools import consume
from operator import attrgetter, itemgetter
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Iterable, Mapping
	from dulwich.objects import ObjectID

def 工repo(pathClone: Path, urlRepository: str) -> Repo:
	"""Open the clone of `urlRepository` at `pathClone`, or make the clone when `pathClone` is absent.

	(AI generated docstring)

	A new clone has one commit of the `main` branch, no file contents, and no checked out files.

	Parameters
	----------
	pathClone : Path
		Directory of the clone.
	urlRepository : str
		Remote repository to clone when `pathClone` does not exist.

	Returns
	-------
	repo : Repo
		The open repository. Close the repository when finished.
	"""
	if pathClone.exists():
		repo: Repo = Repo(pathClone)
	else:
		client, path = get_transport_and_path(urlRepository)
		with closing(client):
			repo = client.clone(path, str(pathClone), mkdir=True, checkout=False, depth=1, branch='main'
				, filter_spec=b'blob:none', protocol_version=2, origin_url=urlRepository)
	return repo

def getObjectIDsWanted(boxOfObjectIDs: list[ObjectID]
	, *_args: Mapping[Ref, ObjectID] | int | None, **_kwargs: Mapping[Ref, ObjectID] | int | None) -> list[ObjectID]:
	"""Select `boxOfObjectIDs` as the objects that a fetch requests.

	(AI generated docstring)

	Pass this function to a fetch through `functools.partial` as `determine_wants`, with
	`boxOfObjectIDs` bound. The remaining arguments are the advertised references and the fetch
	depth, which this function ignores.

	Parameters
	----------
	boxOfObjectIDs : list[ObjectID]
		The objects to request.

	Returns
	-------
	boxOfObjectIDs : list[ObjectID]
		The supplied objects.
	"""
	return boxOfObjectIDs

def readTextGit(objectID: ObjectID, repo: Repo, filterLFS: LFSFilterDriver) -> str:
	"""Read the contents of `objectID` as UTF-8 text, resolving a Git LFS pointer.

	(AI generated docstring)

	Parameters
	----------
	objectID : ObjectID
		Object in the object store of `repo`.
	repo : Repo
		Repository that holds `objectID`.
	filterLFS : LFSFilterDriver
		Supplies the contents that a Git LFS pointer names.

	Returns
	-------
	contents : str
		The decoded contents of the object, or of the file that its Git LFS pointer names.
	"""
	contents: bytes = repo.object_store.get_raw(objectID)[1]
	pointer: LFSPointer | None = LFSPointer.from_bytes(contents)
	if pointer is not None:
		contents = filterLFS.get_content(pointer)
	return contents.decode('utf-8')

def checkoutPaths(boxOfRelativePath: Iterable[PurePosixPath], repo: Repo) -> Repo:
	"""Write the files at `boxOfRelativePath` in the head commit of `repo` to the working tree of `repo`.

	(AI generated docstring)

	You can use this function to check out part of a repository. The function fetches each selected
	file that the object store lacks, and the `.lfsconfig` file, from the remote repository, and it
	resolves Git LFS pointers. A path that the head commit lacks is skipped.

	Parameters
	----------
	boxOfRelativePath : Iterable[PurePosixPath]
		Paths relative to the root of the repository.
	repo : Repo
		An open repository whose head commit contains `.lfsconfig`.

	Returns
	-------
	repo : Repo
		The supplied repository.
	"""
	boxOfPaths: set[bytes] = set(map(compose_left(PurePosixPath.as_posix, str.encode), boxOfRelativePath))
	boxOfPaths.add(b'.lfsconfig')
	tree = parse_tree(repo, repo.head())
	boxOfPaths.intersection_update(map(attrgetter('path'), iter_tree_contents(repo.object_store, tree.id)))
	lookupObjects: dict[bytes, ObjectID] = dict(zip(boxOfPaths
		, map(itemgetter(1), map(partial(tree.lookup_path, repo.__getitem__), boxOfPaths)), strict=True))
	boxOfObjectIDs: list[ObjectID] = list(filterfalse(repo.object_store.__contains__, lookupObjects.values()))
	if boxOfObjectIDs:
		client, path = get_transport_and_path(repo.get_config().get((b'remote', b'origin'), b'url').decode(), config=repo.get_config_stack())
		with closing(client), closing(MemoryRepo()) as repoObjects:
			client.fetch(path, repoObjects, determine_wants=partial(getObjectIDsWanted, boxOfObjectIDs), protocol_version=2)
			repo.object_store.add_objects(tuple(zip(map(repoObjects.object_store.__getitem__, repoObjects.object_store), repeat(None), strict=False)))
	filterLFS = LFSFilterDriver(LFSStore.from_repo(repo, create=True)
		, ConfigFile.from_file(BytesIO(repo.object_store.get_raw(lookupObjects.pop(b'.lfsconfig'))[1])))
	consume(map(writeText, map(partial(readTextGit, repo=repo, filterLFS=filterLFS), lookupObjects.values())
		, map(compose_left(bytes.decode, Path(repo.path).joinpath), lookupObjects), strict=True))
	return repo

def updateRepo(repo: Repo) -> Repo:
	"""Fetch the latest commit of the `main` branch of the `origin` remote, without file contents, and make the commit the head.

	(AI generated docstring)

	Parameters
	----------
	repo : Repo
		An open repository.

	Returns
	-------
	repo : Repo
		The supplied repository.
	"""
	client, path = get_transport_and_path(repo.get_config().get((b'remote', b'origin'), b'url').decode(), config=repo.get_config_stack())
	with closing(client):
		result = client.fetch(path, repo, depth=1, ref_prefix=[b'refs/heads/main'], filter_spec=b'blob:none', protocol_version=2)
	repo.refs.import_refs(Ref(b'refs/remotes/origin'), {Ref(b'main'): result.refs[Ref(b'refs/heads/main')]})
	repo.refs[Ref(b'HEAD')] = repo.refs[Ref(b'refs/remotes/origin/main')]
	return repo
