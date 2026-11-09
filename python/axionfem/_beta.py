"""One-time notices for the beta parts of axionFEM.

A beta module works and is tested, but its interface may still change and its validation is less complete than that
of the validated electromagnetics.  The first use of each beta feature in a process emits one :class:`BetaWarning`;
silence it with ``warnings.simplefilter("ignore", axionfem.BetaWarning)``.
"""
from __future__ import annotations

import warnings

__all__ = ["BetaWarning", "warn_beta"]


class BetaWarning(UserWarning):
    """First use of a beta feature (emitted once per feature and process)."""


_WARNED: set[str] = set()


def warn_beta(feature: str, *, stacklevel: int = 3) -> None:
    """Emit the beta notice for ``feature`` (e.g. ``"axionfem.cavity_coupling"``) once per process."""
    if feature in _WARNED:
        return
    _WARNED.add(feature)
    warnings.warn(f"{feature} is a beta feature: its interface may change in a later release and its validation is "
                  "less complete than that of the validated electromagnetics; check results against the documented "
                  "limits.", BetaWarning, stacklevel=stacklevel)
