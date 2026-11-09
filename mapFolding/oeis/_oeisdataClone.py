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
from mapFolding.oeis._oeisdata import copy_oeisdata, makeBoxOfPathFilename
from mapFolding.oeis._theSSOT import settingsOEIS
from more_itertools import consume
from operator import attrgetter, itemgetter
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Iterable, Mapping, Sequence
	from dulwich.objects import ObjectID
	from mapFolding.theTypes import OEISid

def 工repo(path_oeisdata: Path | None = None) -> Repo:
	path_oeisdata = path_oeisdata or settingsOEIS.pathClone
	if path_oeisdata.exists():
		repo: Repo = Repo(path_oeisdata)
	else:
		client, path = get_transport_and_path(settingsOEIS.urlRepository)
		with closing(client):
			repo = client.clone(path, str(path_oeisdata), mkdir=True, checkout=False, depth=1, branch='main'
				, filter_spec=b'blob:none', protocol_version=2, origin_url=settingsOEIS.urlRepository)
	return repo

def getObjectIDsWanted(boxOfObjectIDs: list[ObjectID]
	, *_args: Mapping[Ref, ObjectID] | int | None, **_kwargs: Mapping[Ref, ObjectID] | int | None) -> list[ObjectID]:
	return boxOfObjectIDs

def readTextGit(objectID: ObjectID, repo: Repo, filterLFS: LFSFilterDriver) -> str:
	contents: bytes = repo.object_store.get_raw(objectID)[1]
	pointer: LFSPointer | None = LFSPointer.from_bytes(contents)
	if pointer is not None:
		contents = filterLFS.get_content(pointer)
	return contents.decode('utf-8')

def checkout_oeisdata(boxOfOEISid: Iterable[OEISid], repo: Repo) -> Repo:
	boxOfPaths: set[bytes] = set(map(compose_left(PurePosixPath.as_posix, str.encode), makeBoxOfPathFilename(boxOfOEISid, PurePosixPath())))
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

def update_oeisdata(repo: Repo) -> Repo:
	client, path = get_transport_and_path(repo.get_config().get((b'remote', b'origin'), b'url').decode(), config=repo.get_config_stack())
	with closing(client):
		result = client.fetch(path, repo, depth=1, ref_prefix=[b'refs/heads/main'], filter_spec=b'blob:none', protocol_version=2)
	repo.refs.import_refs(Ref(b'refs/remotes/origin'), {Ref(b'main'): result.refs[Ref(b'refs/heads/main')]})
	repo.refs[Ref(b'HEAD')] = repo.refs[Ref(b'refs/remotes/origin/main')]
	return repo

def checkoutSnapshot(boxOfOEISid: Sequence[OEISid] | None = None, path_oeisdata: Path | None = None, pathSnapshot: Path | None = None) -> Path:
	if boxOfOEISid is None:
		boxOfOEISid = settingsOEIS.boxOfIDs
	path_oeisdata = path_oeisdata or settingsOEIS.pathClone
	pathSnapshot = pathSnapshot or settingsOEIS.pathSnapshot
	with 工repo(path_oeisdata) as repo:
		checkout_oeisdata(boxOfOEISid, repo)
	return copy_oeisdata(makeBoxOfPathFilename(boxOfOEISid, PurePosixPath()), path_oeisdata, pathSnapshot)

def updateSnapshot(boxOfOEISid: Sequence[OEISid] | None = None, path_oeisdata: Path | None = None, pathSnapshot: Path | None = None) -> Path:
	with 工repo(path_oeisdata) as repo:
		update_oeisdata(repo)
	return checkoutSnapshot(boxOfOEISid, path_oeisdata, pathSnapshot)
