from cffconvert.cli.create_citation import create_citation
from packaging.metadata import Metadata as PyPAMetadata
from typing import Any, Dict, List
import attrs
import cffconvert
import tempfile
import packaging
import packaging.metadata
import packaging.utils
import packaging.version
import pathlib
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
    - I want the updated citation to be in the `commit` field of itself
"""

@attrs.define
class CitationNexus:
    """
    - one-to-one correlation with `cffconvert.lib.cff_1_2_x.citation` class Citation_1_2_x.cffobj
    """
    cffDASHversion: str # pathFilenameCitationSSOT
    message: str # pathFilenameCitationSSOT

    abstract: str | None = None # pathFilenameCitationSSOT
    authors: list[dict[str,str]] = attrs.field(factory=list) # pathFilenamePackageSSOT; pyproject.toml authors
    commit: str | None = None # workflows['Make GitHub Release']
    contact: list[dict[str,str]] = attrs.field(factory=list) # pathFilenamePackageSSOT; pyproject.toml maintainers
    dateDASHreleased: str | None = None # workflows['Make GitHub Release']
    doi: str | None = None # pathFilenameCitationSSOT
    identifiers: list[str] = attrs.field(factory=list) # workflows['Make GitHub Release']
    keywords: list[str] = attrs.field(factory=list) # pathFilenamePackageSSOT; packaging.metadata.Metadata.keywords
    license: str | None = None # pathFilenamePackageSSOT; packaging.metadata.Metadata.license_expression
    licenseDASHurl: str | None = None # pathFilenamePackageSSOT; packaging.metadata.Metadata.project_urls: license or pyproject.toml urls license
    preferredDASHcitation: str | None = None # pathFilenameCitationSSOT
    references: list[str] = attrs.field(factory=list) # bibtex files in pathCitationSSOT. Conversion method and timing TBD.
    repositoryDASHartifact: str | None = None # (https://pypi.org/pypi/{package_name}/json').json()['releases']
    repositoryDASHcode: str | None = None # workflows['Make GitHub Release']
    repository: str | None = None # pathFilenamePackageSSOT; packaging.metadata.Metadata.project_urls: repository
    title: str | None = None # pathFilenamePackageSSOT; pyproject.toml name (packaging normalizes the names)
    type: str | None = None # pathFilenameCitationSSOT
    url: str | None = None # pathFilenamePackageSSOT; packaging.metadata.Metadata.project_urls: homepage
    version: str | None = None # pathFilenamePackageSSOT; packaging.metadata.Metadata.version

    def setInStone(self, prophet: str) -> "CitationNexus":
        match prophet:
            case "Citation":
                pass
                # "freeze" these items
                # setattr(self.cffDASHversion, 'type', Final[str])
                # setattr(self.doi, 'type', Final[str])
                # cffDASHversion: str # pathFilenameCitationSSOT
                # message: str # pathFilenameCitationSSOT
                # abstract: str | None = None # pathFilenameCitationSSOT
                # doi: str | None = None # pathFilenameCitationSSOT
                # preferredDASHcitation: str | None = None # pathFilenameCitationSSOT
                # type: str | None = None # pathFilenameCitationSSOT
            case "PyPA":
                pass
                # "freeze" these items
                # setattr(self.keywords, 'type', Final[list[str]])
                # setattr(self.license, 'type', Final[str])
                # setattr(self.licenseDASHurl, 'type', Final[str])
                # setattr(self.repository, 'type', Final[str])
                # setattr(self.url, 'type', Final[str])
                # setattr(self.version, 'type', Final[str])
            case "pyprojectDOTtoml":
                pass
                # "freeze" these items
                # setattr(self.authors, 'type', Final[list[dict[str,str]]])
                # setattr(self.contact, 'type', Final[list[dict[str,str]]])
                # setattr(self.title, 'type', Final[str])
        return self

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

    pathFilenameForValidation = pathlib.Path(tempfile.mktemp())

    def writeStream(pathFilename):
        with open(pathFilename, 'w') as pathlibIsAStealthContextManagerThatRuamelCannotDetectAndRefusesToWorkWith:
            yamlWorkhorse.dump(dictionaryCitation, pathlibIsAStealthContextManagerThatRuamelCannotDetectAndRefusesToWorkWith)

    writeStream(pathFilenameForValidation)

    citationObject: cffconvert.Citation = create_citation(infile=pathFilenameForValidation, url=None)
    if citationObject.validate(verbose=True) is None:
        writeStream(pathFilenameCitationSSOT)
        writeStream(pathFilenameCitationDOTcffRepo)

def logistics():
    # Prefer reliable, dynamic values over hardcoded ones
    packageNameHARDCODED: str = 'mapFolding'

    packageName: str = packageNameHARDCODED
    pathRepoRoot = pathlib.Path(__file__).parent.parent.parent
    pathFilenamePackageSSOT = pathRepoRoot / 'pyproject.toml'
    filenameGitHubAction = 'updateCitation.yml'
    pathFilenameGitHubAction = pathRepoRoot / '.github' / 'workflows' / filenameGitHubAction

    filenameCitationDOTcff = 'CITATION.cff'
    pathCitations = pathRepoRoot / packageName / 'citations'
    pathFilenameCitationSSOT = pathCitations / filenameCitationDOTcff
    pathFilenameCitationDOTcffRepo = pathRepoRoot / filenameCitationDOTcff

    nexusCitation = getNexusCitation(pathFilenameCitationSSOT)

    tomlPackageData: Dict[str, Any] = tomli.loads(pathFilenamePackageSSOT.read_text())['project']
    # https://packaging.python.org/en/latest/specifications/pyproject-toml/
    pypaMetadata: PyPAMetadata = getPypaMetadata(tomlPackageData)

    nexusCitation = addPypaMetadata(nexusCitation, pypaMetadata)
    nexusCitation = add_pyprojectDOTtoml(nexusCitation, tomlPackageData)

    writeCitation(nexusCitation, pathFilenameCitationSSOT, pathFilenameCitationDOTcffRepo)

if __name__ == '__main__':
    logistics()
