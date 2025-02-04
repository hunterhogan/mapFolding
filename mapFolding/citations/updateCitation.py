from cffconvert.cli.create_citation import create_citation
from mapFolding.citations.constants import GITHUB_API_VERSION_HEADER
from packaging.metadata import Metadata as PyPAMetadata
from typing import Any, Dict, List
import attrs
import cffconvert
import os
import packaging
import packaging.metadata
import packaging.utils
import packaging.version
import pathlib
import requests
import ruamel.yaml
import tomli

listProjectURLsTarget: List[str] = ["homepage", "license", "repository"]

"""
Tentative plan:
- Commit and push to GitHub
- GitHub Action gathers information from the sources of truth
- If the citation needs to be updated, write to both
    - pathFilenameCitationSSOT
    - pathFilenameCitationDOTcffRepo
- Commit and push to GitHub
    - this complicates things
    - I want the updated citation to be in the `commit` field of itself: but the commit field isn't even working right now
"""

@attrs.define
class CitationNexus:
    """
    - one-to-one correlation with `cffconvert.lib.cff_1_2_x.citation` class Citation_1_2_x.cffobj
    """
    cffDASHversion: str
    message: str

    abstract: str | None = None
    authors: list[dict[str,str]] = attrs.field(factory=list)
    # GitHub TODO
    commit: str | None = None
    contact: list[dict[str,str]] = attrs.field(factory=list)
    dateDASHreleased: str | None = None
    doi: str | None = None
    identifiers: list[str] = attrs.field(factory=list)
    keywords: list[str] = attrs.field(factory=list)
    license: str | None = None
    licenseDASHurl: str | None = None
    preferredDASHcitation: str | None = None
    # TODO bibtex files in pathCitationSSOT. Conversion method and timing TBD.
    references: list[str] = attrs.field(factory=list)
    repository: str | None = None
    repositoryDASHartifact: str | None = None
    repositoryDASHcode: str | None = None
    title: str | None = None
    type: str | None = None
    url: str | None = None
    version: str | None = None

    def setInStone(self, prophet: str) -> "CitationNexus":
        match prophet:
            case "Citation":
                pass
                # "freeze" these items
                # setattr(self.cffDASHversion, 'type', Final[str])
                # setattr(self.doi, 'type', Final[str])
                # cffDASHversion: str
                # message: str
                # abstract: str | None = None
                # doi: str | None = None
                # preferredDASHcitation: str | None = None
                # type: str | None = None
            case "GitHub":
                pass
                # "freeze" these items
                # setattr(self.commit, 'type', Final[str])
                # setattr(self.dateDASHreleased, 'type', Final[str])
                # setattr(self.identifiers, 'type', Final[list[str]])
                # setattr(self.repositoryDASHcode, 'type', Final[str])
            case "PyPA":
                pass
                # "freeze" these items
                # setattr(self.keywords, 'type', Final[list[str]])
                # setattr(self.license, 'type', Final[str])
                # setattr(self.licenseDASHurl, 'type', Final[str])
                # setattr(self.repository, 'type', Final[str])
                # setattr(self.url, 'type', Final[str])
                # setattr(self.version, 'type', Final[str])
            case "PyPI":
                pass
                # "freeze" these items
                # setattr(self.repositoryDASHartifact, 'type', Final[str])
            case "pyprojectDOTtoml":
                pass
                # "freeze" these items
                # setattr(self.authors, 'type', Final[list[dict[str,str]]])
                # setattr(self.contact, 'type', Final[list[dict[str,str]]])
                # setattr(self.title, 'type', Final[str])
        return self

def addPypaMetadata(nexusCitation: CitationNexus, metadata: PyPAMetadata) -> CitationNexus:
    if not metadata.name:
        raise ValueError("Metadata name is required.")

    nexusCitation.title = metadata.name
    if metadata.version: nexusCitation.version = str(metadata.version)
    if metadata.keywords: nexusCitation.keywords = metadata.keywords
    if metadata.license_expression: nexusCitation.license = metadata.license_expression

    Z0Z_lookup: Dict[str, str] = {
        "homepage": "url",
        "license": "licenseDASHurl",
        "repository": "repository",
    }
    if metadata.project_urls:
        for urlTarget in listProjectURLsTarget:
            url = metadata.project_urls.get(urlTarget, None)
            if url:
                setattr(nexusCitation, Z0Z_lookup[urlTarget], url)

    nexusCitation = nexusCitation.setInStone("PyPA")
    return nexusCitation

def add_pyprojectDOTtoml(nexusCitation: CitationNexus, packageData: Dict[str, Any]) -> CitationNexus:
    def Z0Z_ImaNotValidatingNoNames(person: Dict[str, str]) -> Dict[str, str]:
        cffPerson: Dict[str, str] = {}
        if person.get('name', None):
            cffPerson['given-names'], cffPerson['family-names'] = person['name'].split(' ', 1)
        if person.get('email', None):
            cffPerson['email'] = person['email']
        return cffPerson
    listAuthors = packageData.get("authors", None)
    if not listAuthors:
        raise ValueError("Authors are required.")
    else:
        listPersons = []
        for person in listAuthors:
            listPersons.append(Z0Z_ImaNotValidatingNoNames(person))
            nexusCitation.authors = listPersons
    if packageData.get("maintainers", None):
        listPersons = []
        for person in packageData["maintainers"]:
            listPersons.append(Z0Z_ImaNotValidatingNoNames(person))
            nexusCitation.contact = listPersons
    nexusCitation.title = packageData["name"]
    nexusCitation = nexusCitation.setInStone("pyprojectDOTtoml")
    return nexusCitation

def getGitHubRelease(nexusCitation: CitationNexus) -> Dict[str, Any]:
    """Return a dictionary with GitHub release data.

    The dictionary contains the following keys:
        commit: The commit hash (using the API field 'target_commitish').
        date-released: The published date (in YYYY-MM-DD format).
        identifiers: A list with one identifier object, whose description is
            'The URL for {nexusCitation.title} {nexusCitation.version}.'
        repository-code: A URL for the commit in the repository.

    Raises:
        ValueError: If the nexusCitation.repository is not set or cannot be parsed.
        RuntimeError: If the HTTP request to GitHub fails.
    """
    if not nexusCitation.repository:
        raise ValueError("Repository URL is required to get GitHub release info.")

    urlparts = nexusCitation.repository.replace("https://github.com", "", 1).strip("/").split("/") + [None] * 5
    ownername, reponame, _2, refvalue, *_filename_parts = urlparts
    reponame = reponame.replace(".git", "") # type: ignore # Remove .git from the repository name, if present.
    assert ownername is not None, "URL should include the name of the owner/organization."
    assert reponame is not None, "URL should include the name of the repository."
    if refvalue is None:
        repos_api = f"https://api.github.com/repos/{ownername}/{reponame}/releases/latest"
        headers = GITHUB_API_VERSION_HEADER
        headers.update({"Accept": "application/vnd.github+json"})
        token = os.environ.get("GITHUB_TOKEN")
        headers.update({"Authorization": f"Bearer { token }"})
        response = requests.get(repos_api, headers=headers)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to get GitHub release info: {response.status_code}")

    releaseData = response.json()
    # commitHash = releaseData.get("target_commitish")
    publishedAt = releaseData.get("published_at")
    if publishedAt:
        # Convert ISO timestamp (e.g., "2020-12-31T12:34:56Z") to "YYYY-MM-DD".
        publishedAt = publishedAt.split("T")[0]

    releaseHtmlUrl = releaseData.get("html_url")
    identifierDescription = f"The URL for {nexusCitation.title} {nexusCitation.version}."
    return {
        # "commit": commitHash,
        "dateDASHreleased": publishedAt,
        "identifiers": [{
            "type": "url",
            "value": releaseHtmlUrl,
            "description": identifierDescription,
        }],
        "repositoryDASHcode": releaseHtmlUrl,
    }

def addGitHubRelease(nexusCitation: CitationNexus) -> CitationNexus:
    """
    Update the nexusCitation with GitHub release information.

    This function populates the following fields on the nexusCitation:
        - commit: using the commit hash from GitHub.
        - dateDASHreleased: the release date.
        - identifiers: appends a GitHub-specific identifier.
        - repositoryDASHcode: the URL to view the commit in the repository.

    Returns:
        The updated CitationNexus instance.

    Raises:
        Any exception raised by getGitHubRelease.
    """
    gitHubReleaseData = getGitHubRelease(nexusCitation)
    nexusCitation.commit = gitHubReleaseData.get("commit")
    nexusCitation.dateDASHreleased = gitHubReleaseData.get("dateDASHreleased")
    # Overwrite the existing list of identifiers. This could be better
    nexusCitation.identifiers = gitHubReleaseData.get("identifiers", [])
    nexusCitation.repositoryDASHcode = gitHubReleaseData.get("repositoryDASHcode")
    return nexusCitation

def getPyPIrelease(nexusCitation: CitationNexus) -> Dict[str, Any]:
    if not nexusCitation.title:
        raise ValueError("Package name (title) is required to get PyPI release info.")
    if not nexusCitation.version:
        raise ValueError("Package version is required to get PyPI release info.")

    packageName = packaging.utils.canonicalize_name(nexusCitation.title)
    version = str(nexusCitation.version)
    return {
        "repositoryDASHartifact": f"https://pypi.org/project/{packageName}/{version}/"
    }

def addPyPIrelease(nexusCitation: CitationNexus) -> CitationNexus:
    pypiReleaseData = getPyPIrelease(nexusCitation)
    nexusCitation.repositoryDASHartifact = pypiReleaseData.get("repositoryDASHartifact")
    return nexusCitation

def getNexusCitation(pathFilenameCitationSSOT: pathlib.Path) -> CitationNexus:

    # `cffconvert.cli.create_citation.create_citation()` is PAINFULLY mundane, but a major problem
    # in the CFF ecosystem is divergence. Therefore, I will use this function so that my code
    # converges with the CFF ecosystem.
    citationObject: cffconvert.Citation = create_citation(infile=pathFilenameCitationSSOT, url=None)
    # `._parse()` is a yaml loader: use it for convergence
    cffobj: Dict[Any, Any] = citationObject._parse()

    nexusCitation = CitationNexus(
        cffDASHversion=cffobj["cff-version"],
        message=cffobj["message"],
    )

    Z0Z_list: List[attrs.Attribute] = list(attrs.fields(type(nexusCitation)))
    for Z0Z_field in Z0Z_list:
        cffobjKeyName: str = Z0Z_field.name.replace("DASH", "-")
        cffobjValue = cffobj.get(cffobjKeyName)
        if cffobjValue: # An empty list will be False
            setattr(nexusCitation, Z0Z_field.name, cffobjValue)

    nexusCitation = nexusCitation.setInStone("Citation")
    return nexusCitation

def getPypaMetadata(packageData: Dict[str, Any]) -> PyPAMetadata:
    """
    Create a PyPA metadata object (version 2.4) from packageData.
    https://packaging.python.org/en/latest/specifications/core-metadata/
    """
    dictionaryProjectURLs: Dict[str, str] = {}
    for urlName, url in packageData.get("urls", {}).items():
        urlName = urlName.lower()
        if urlName in listProjectURLsTarget:
            dictionaryProjectURLs[urlName] = url

    metadataRaw = packaging.metadata.RawMetadata(
        keywords=packageData.get("keywords", []),
        license_expression=packageData.get("license", {}).get("text", ""),
        metadata_version="2.4",
        name=packaging.utils.canonicalize_name(packageData.get("name", None), validate=True), # packaging.metadata.InvalidMetadata: 'name' is a required field
        project_urls=dictionaryProjectURLs,
        version=packageData.get("version", None),
    )

    metadata = PyPAMetadata().from_raw(metadataRaw)
    return metadata

def writeCitation(nexusCitation: CitationNexus, pathFilenameCitationSSOT: pathlib.Path, pathFilenameCitationDOTcffRepo: pathlib.Path):
    # NOTE embarrassingly hacky process to follow
    parameterIndent= 2
    parameterLineWidth = 60
    yamlWorkhorse = ruamel.yaml.YAML()

    def srsly(Z0Z_filed, Z0Z_value):
        if Z0Z_value: # empty lists
            return True
        else:
            return False

    dictionaryCitation = attrs.asdict(nexusCitation, filter=srsly)
    for keyName in list(dictionaryCitation.keys()):
        dictionaryCitation[keyName.replace("DASH", "-")] = dictionaryCitation.pop(keyName)

    pathFilenameForValidation = pathFilenameCitationSSOT.with_stem('validation')

    def writeStream(pathFilename):
        with open(pathFilename, 'w') as pathlibIsAStealthContextManagerThatRuamelCannotDetectAndRefusesToWorkWith:
            yamlWorkhorse.dump(dictionaryCitation, pathlibIsAStealthContextManagerThatRuamelCannotDetectAndRefusesToWorkWith)

    writeStream(pathFilenameForValidation)

    citationObject: cffconvert.Citation = create_citation(infile=pathFilenameForValidation, url=None)
    if citationObject.validate() is None:
        writeStream(pathFilenameCitationSSOT)
        writeStream(pathFilenameCitationDOTcffRepo)

    pathFilenameForValidation.unlink()

def logistics():
    # Prefer reliable, dynamic values over hardcoded ones
    pathRepoRoot = pathlib.Path(__file__).parent.parent.parent
    pathFilenamePackageSSOT = pathRepoRoot / 'pyproject.toml'

    tomlPackageData: Dict[str, Any] = tomli.loads(pathFilenamePackageSSOT.read_text())['project']
    # https://packaging.python.org/en/latest/specifications/pyproject-toml/

    packageName: str = tomlPackageData.get("name", None)
    if not packageName:
        raise ValueError("Package name is required.")

    filenameCitationDOTcff = 'CITATION.cff'
    pathCitations = pathRepoRoot / packageName / 'citations'
    pathFilenameCitationSSOT = pathCitations / filenameCitationDOTcff
    pathFilenameCitationDOTcffRepo = pathRepoRoot / filenameCitationDOTcff

    nexusCitation = getNexusCitation(pathFilenameCitationSSOT)

    pypaMetadata: PyPAMetadata = getPypaMetadata(tomlPackageData)

    nexusCitation = addPypaMetadata(nexusCitation, pypaMetadata)
    nexusCitation = add_pyprojectDOTtoml(nexusCitation, tomlPackageData)

    nexusCitation = addGitHubRelease(nexusCitation)
    nexusCitation = addPyPIrelease(nexusCitation)

    filenameGitHubAction = 'updateCitation.yml'
    pathFilenameGitHubAction = pathRepoRoot / '.github' / 'workflows' / filenameGitHubAction

    writeCitation(nexusCitation, pathFilenameCitationSSOT, pathFilenameCitationDOTcffRepo)

if __name__ == '__main__':
    logistics()
