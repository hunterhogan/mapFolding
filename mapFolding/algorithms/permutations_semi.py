#=SIN=
# ruff: file-ignore[undocumented-public-class, undocumented-magic-method]
from __future__ import annotations

import dataclasses

@dataclasses.dataclass(slots=True)
class Gap:
	arm: Arm
	proximal: Gap = dataclasses.field(init=False)
	distal: Gap = dataclasses.field(init=False)

	def __post_init__(self) -> None:
		self.proximal = self
		self.distal = self

class Arm:
	__slots__ = ('complement', 'gapDistal', 'gapProximal')

	def __init__(self) -> None:
		self.complement: Arm = self

		self.gapProximal: Gap = Gap(self)
		self.gapProximal.proximal = Gap(self)
		self.gapProximal.proximal.proximal = self.gapProximal

		self.gapDistal: Gap = Gap(self)
		self.gapDistal.proximal = Gap(self)
		self.gapDistal.proximal.proximal = self.gapDistal

		self.gapProximal.distal = self.gapDistal
		self.gapDistal.proximal.distal = self.gapProximal.proximal

def count(crossingNext: int, arm: Arm, n: int) -> int:
	total: int = 1
	if crossingNext <= n:
		total = countArm(crossingNext, arm, n)
		total += countArm(crossingNext, arm.complement, n)
	return total

def countArm(crossingNext: int, arm: Arm, n: int) -> int:
	total: int = 0
	gap: Gap = arm.gapProximal.distal

	while gap is not arm.gapDistal:
		armTarget: Arm = gap.arm
		armFragmentProximal: Arm = makeComplements()
		gapProximal: Gap = arm.gapProximal
		arm.gapProximal = armFragmentProximal.gapDistal
		armFragmentProximal.gapDistal = gapProximal.proximal
		armFragmentProximal.gapProximal.distal = gap.proximal.distal
		gap.proximal.distal.proximal.distal = armFragmentProximal.gapProximal.proximal
		arm.gapProximal.distal = gap.distal
		gap.distal.proximal.distal = arm.gapProximal.proximal

		gapFragmentDistal: Gap = makeGap(arm)
		gapFragmentProximal: Gap = makeGap(armFragmentProximal)
		insertGap(armTarget, gapFragmentDistal)
		insertGap(armTarget.complement, gapFragmentProximal)
		total += count(crossingNext + 1, armTarget, n)
		removeGap(gapFragmentDistal)
		removeGap(gapFragmentProximal)

		gap.proximal.distal.proximal.distal = gap
		gap.distal.proximal.distal = gap.proximal
		arm.gapProximal = gapProximal
		gap = gap.distal

	return total

def makeComplements() -> Arm:
	arm: Arm = Arm()
	arm.complement = Arm()
	arm.complement.complement = arm
	return arm

def makeGap(arm: Arm) -> Gap:
	gap: Gap = Gap(arm)
	gap.proximal = Gap(arm.complement)
	gap.proximal.proximal = gap
	return gap

def insertGap(arm: Arm, gap: Gap) -> None:
	gapFollowing: Gap = arm.gapProximal.distal
	gap.distal = gapFollowing
	gap.proximal.distal = arm.gapProximal.proximal
	gapFollowing.proximal.distal = gap.proximal
	arm.gapProximal.distal = gap

def removeGap(gap: Gap) -> None:
	gap.proximal.distal.proximal.distal = gap.distal
	gap.distal.proximal.distal = gap.proximal.distal

def doTheNeedful(n: int) -> int:
	arm: Arm = makeComplements()
	insertGap(arm, makeGap(makeComplements()))
	insertGap(arm.complement, makeGap(makeComplements().complement))
	return count(2, arm, n)
