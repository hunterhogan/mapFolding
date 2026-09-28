from __future__ import annotations

from mapFolding.dataBaskets import StateMeanders

def integersWidePolars吗(state: StateMeanders, meandersMaximum: int | None=None) -> bool:
    if meandersMaximum is None:
        meandersMaximum = max(state.lookupMeanders.values())
    return 128 < max(state.bitWidth + 3, (meandersMaximum * (state.bitWidth + 4)).bit_length())
