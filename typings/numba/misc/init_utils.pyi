from _typeshed import Incomplete
from typing import NamedTuple

class version_info(NamedTuple):
    major: Incomplete
    minor: Incomplete
    patch: Incomplete
    short: Incomplete
    full: Incomplete
    string: Incomplete
    tuple: Incomplete
    git_revision: Incomplete

def generate_version_info(version):
    """Process a version string into a structured version_info object.

    Parameters
    ----------
    version: str
        a string describing the current version

    Returns
    -------
    version_info: tuple
        structured version information

    See also
    --------
    Look at the definition of 'version_info' in this module for details.

    """
