"""
hybrid/sovereignty_metrics.py
------------------------------
Time-windowed sovereignty rate tracker for the Nawal AI project.

"Sovereignty" = the fraction of queries answered by Nawal itself
(rather than falling back to the DeepSeek teacher model).

Usage example::

    from hybrid.sovereignty_metrics import SovereigntyMetrics

    metrics = SovereigntyMetrics()
    metrics.record("nawal")
    metrics.record("deepseek")
    print(metrics.sovereignty_rate())            # last-30-min window
    print(metrics.sovereignty_rate(window_minutes=60))  # last hour
    print(metrics.total_sovereignty_rate())      # all time
    print(metrics.is_on_track(month=6))          # True if ≥ 95% at month 6

Roadmap targets (from NAWAL_BRAIN_IMPLEMENTATION_PLAN.md):
    Month 1  → 50% sovereignty
    Month 3  → 70%
    Month 6  → 95%
    Month 12 → 90% (sustained; lower because DeepSeek is retained for PhD-level tasks)
"""

from __future__ import annotations

import time
from collections import deque

__all__ = ["SovereigntyMetrics"]

# Default sliding-window size in minutes (30 minutes)
_DEFAULT_WINDOW_MINUTES = 30

# Roadmap: month → minimum sovereignty fraction to be "on track".
# Values are inclusive lower bounds.
_ROADMAP_TARGETS: dict[int, float] = {
    1: 0.50,
    3: 0.70,
    6: 0.95,
    12: 0.90,
}


class SovereigntyMetrics:
    """
    Tracks query routing decisions (Nawal vs DeepSeek) and computes
    sovereignty rates over sliding time windows.

    Parameters
    ----------
    window_minutes : int
        Default look-back window for :meth:`sovereignty_rate`.
    """

    def __init__(self, window_minutes: int = _DEFAULT_WINDOW_MINUTES) -> None:
        self._default_window_minutes = window_minutes
        # Each item is (unix_timestamp_seconds, model_name)
        self._events: deque[tuple[float, str]] = deque()
        self._total_nawal = 0
        self._total_deepseek = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, model_used: str) -> None:
        """
        Record a single routing decision.

        Parameters
        ----------
        model_used : str
            Either ``"nawal"`` or ``"deepseek"``.

        Raises
        ------
        ValueError
            If *model_used* is not one of the accepted values.
        """
        model = model_used.lower().strip()
        if model not in ("nawal", "deepseek"):
            raise ValueError(f"model_used must be 'nawal' or 'deepseek', got {model_used!r}")
        now = time.monotonic()
        self._events.append((now, model))
        if model == "nawal":
            self._total_nawal += 1
        else:
            self._total_deepseek += 1

    def sovereignty_rate(self, window_minutes: int | None = None) -> float:
        """
        Fraction of queries handled by Nawal within the last *window_minutes*.

        Parameters
        ----------
        window_minutes : int or None
            Look-back window in minutes.  Defaults to the value passed at
            construction time (default: 30 minutes).

        Returns
        -------
        float
            A value in ``[0.0, 1.0]``.  Returns ``0.0`` if no events fall
            within the window.
        """
        window = window_minutes if window_minutes is not None else self._default_window_minutes
        cutoff = time.monotonic() - window * 60.0
        events_in_window = [e for e in self._events if e[0] >= cutoff]
        if not events_in_window:
            return 0.0
        nawal_count = sum(1 for _, m in events_in_window if m == "nawal")
        return nawal_count / len(events_in_window)

    def total_sovereignty_rate(self) -> float:
        """
        Sovereignty rate over the entire lifetime of this instance.

        Returns
        -------
        float
            ``0.0`` if no events have been recorded.
        """
        total = self._total_nawal + self._total_deepseek
        if total == 0:
            return 0.0
        return self._total_nawal / total

    def is_on_track(self, month: int) -> bool:
        """
        Check whether the current *total* sovereignty rate meets the roadmap
        target for the given month.

        Parameters
        ----------
        month : int
            Project month number.  If *month* is not in the roadmap table
            the target is interpolated from the nearest defined milestones.
            Months beyond 12 use the month-12 target.

        Returns
        -------
        bool
            ``True`` if ``total_sovereignty_rate() >= target``.
        """
        target = self._target_for_month(month)
        return self.total_sovereignty_rate() >= target

    def target_for_month(self, month: int) -> float:
        """Public accessor for the roadmap target at a given month."""
        return self._target_for_month(month)

    def snapshot(self, window_minutes: int | None = None) -> dict[str, object]:
        """
        Return a JSON-serialisable snapshot of current metrics.

        Returns
        -------
        dict
            Keys: ``window_minutes``, ``window_sovereignty_rate``,
            ``total_sovereignty_rate``, ``total_nawal``, ``total_deepseek``,
            ``total_queries``.
        """
        w = window_minutes if window_minutes is not None else self._default_window_minutes
        return {
            "window_minutes": w,
            "window_sovereignty_rate": self.sovereignty_rate(w),
            "total_sovereignty_rate": self.total_sovereignty_rate(),
            "total_nawal": self._total_nawal,
            "total_deepseek": self._total_deepseek,
            "total_queries": self._total_nawal + self._total_deepseek,
        }

    def reset(self) -> None:
        """Clear all recorded events and reset counters."""
        self._events.clear()
        self._total_nawal = 0
        self._total_deepseek = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _target_for_month(self, month: int) -> float:
        """Interpolate or look up roadmap target for *month*."""
        if month in _ROADMAP_TARGETS:
            return _ROADMAP_TARGETS[month]
        # Find the nearest milestone ≤ month
        milestones = sorted(_ROADMAP_TARGETS.keys())
        lower = max((m for m in milestones if m <= month), default=None)
        upper = min((m for m in milestones if m >= month), default=None)

        if lower is None:
            # Before first milestone — use first target
            return _ROADMAP_TARGETS[milestones[0]]
        if upper is None:
            # After last milestone — use last target
            return _ROADMAP_TARGETS[milestones[-1]]

        # Linear interpolation between lower and upper milestones
        t_low = _ROADMAP_TARGETS[lower]
        t_high = _ROADMAP_TARGETS[upper]
        frac = (month - lower) / (upper - lower)
        return t_low + frac * (t_high - t_low)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        total = self._total_nawal + self._total_deepseek
        rate = self.total_sovereignty_rate()
        return (
            f"SovereigntyMetrics(" f"total={total}, nawal={self._total_nawal}, " f"rate={rate:.1%})"
        )
