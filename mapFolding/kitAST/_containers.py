from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING

if TYPE_CHECKING:
	from astToolkit import identifierDotAttribute

class Settings形(NamedTuple):
	"""Configuration for mapping framework datatypes to compiled datatypes.

	This configuration class defines how abstract datatypes used in the map folding framework should
	be replaced with compiled datatypes during code generation. Each configuration specifies the
	source module, target type name, and optional import alias for the transformation.

	Attributes
	----------
	datatypeIdentifier : str
		Framework datatype identifier to be replaced.
	typeModule : identifierDotAttribute
		Module containing the target datatype (e.g., 'codon', 'numpy').
	typeIdentifier : str
		Concrete type name in the target module.
	type_asname : str | None = None
		Optional import alias for the type.
	"""

	datatypeIdentifier: str
	typeModule: identifierDotAttribute
	typeIdentifier: str
	type_asname: str | None = None
