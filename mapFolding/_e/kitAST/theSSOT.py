"""Centralized configuration and default values."""

from __future__ import annotations

from copy import deepcopy
from mapFolding.kitAST.theSSOT import default
from mapFolding.theSSOT import settingsPackage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
		from mapFolding.theTypes import Default

defaultElimination: Default = deepcopy(default)
defaultElimination['logicalPath'].update({
		'algorithm': f'{settingsPackage.identifierPackage}._e.algorithms'
		, 'synthetic': f'{settingsPackage.identifierPackage}._e.synthesized'
	})
defaultElimination['module'].update({
		'algorithm': 'eliminationCrease'
		, 'dataBasket': 'dataBaskets'
	})
defaultElimination['variable'].update({
		'counting': 'groupsOfFolds'
		, 'stateDataclass': 'StateElimination'
	})
