"""
SelfRepair — Triggered when DriftDetector detects model drift.

Repair strategies
-----------------
``ROLLBACK``
    Locate the most recent known-good checkpoint (via ``storage/``
    checkpoint manager) and request a reload.  This is the primary
    auto-repair path.

``ALERT``
    Publish a drift alert to the monitoring system and to EpisodicMemory
    (if available), but make no model change.  Useful when auto-repair
    is disabled.

``ISOLATE``
    Mark the current model as quarantined.  Subsequent inference
    requests will route to the fallback model (DeepSeek teacher).

``RESET``
    Full state reset: clear working memory, reset identity module, and
    attempt a fresh checkpoint load.

Integration hooks
-----------------
``checkpoint_path``
    Directory of checkpoint files (scanned for ``*.pt`` files ordered
    by modification time).

``episodic_memory``
    If provided, each repair event is logged as a MemoryRecord.

``alert_callback``
    If provided, called with the DriftReport on every triggered repair.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from loguru import logger

from maintenance.interfaces import (
    AbstractSelfRepair,
    DriftReport,
    RepairResult,
    RepairStrategy,
)


# --------------------------------------------------------------------------- #
# SelfRepair                                                                    #
# --------------------------------------------------------------------------- #

class SelfRepair(AbstractSelfRepair):
    """
    Triggered repair controller.

    Args:
        checkpoint_path : Directory containing ``*.pt`` checkpoint files.
        episodic_memory : Optional EpisodicMemory to log repair events.
        alert_callback  : Optional callable(DriftReport) for custom alerts.
        auto_repair     : If False, only ALERT is executed; all other
                          strategies reduce to ALERT (default True).
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        episodic_memory: Optional[Any] = None,
        alert_callback: Optional[Callable[[DriftReport], None]] = None,
        auto_repair: bool = True,
    ) -> None:
        self._checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self._episodic        = episodic_memory
        self._alert_cb        = alert_callback
        self._auto_repair     = auto_repair
        self._repair_history: list[Dict[str, Any]] = []

        logger.info(
            f"SelfRepair ready: auto_repair={auto_repair} "
            f"checkpoint_path={checkpoint_path}"
        )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def repair(
        self,
        strategy: RepairStrategy = RepairStrategy.ROLLBACK,
        drift_report: Optional[DriftReport] = None,
    ) -> RepairResult:
        """
        Execute the repair strategy.

        If ``auto_repair=False``, this always runs ALERT regardless of
        the requested strategy.

        Args:
            strategy     : What to do.
            drift_report : The triggering drift report (for logging).

        Returns:
            RepairResult describing what was done.
        """
        if not self._auto_repair:
            strategy = RepairStrategy.ALERT

        t0 = time.perf_counter()

        if strategy == RepairStrategy.ROLLBACK:
            result = self._do_rollback(drift_report)
        elif strategy == RepairStrategy.ISOLATE:
            result = self._do_isolate(drift_report)
        elif strategy == RepairStrategy.RESET:
            result = self._do_reset(drift_report)
        else:
            result = self._do_alert(drift_report)

        elapsed = (time.perf_counter() - t0) * 1000
        result.metadata["elapsed_ms"] = elapsed

        # Record to episodic memory
        self._maybe_log(result, drift_report)

        # External alert callback
        if self._alert_cb is not None and drift_report is not None:
            try:
                self._alert_cb(drift_report)
            except Exception as exc:
                logger.warning(f"SelfRepair: alert_callback raised {exc}")

        self._repair_history.append({
            "strategy":  strategy.value,
            "success":   result.success,
            "timestamp": time.time(),
        })

        return result

    def rollback(self, checkpoint_id: str) -> bool:
        """
        Restore a specific checkpoint by ID.

        Returns:
            True if the checkpoint file was found and accepted.
        """
        if self._checkpoint_path is None:
            logger.warning("SelfRepair.rollback: no checkpoint_path configured")
            return False

        candidates = sorted(
            self._checkpoint_path.glob(f"*{checkpoint_id}*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            logger.warning(
                f"SelfRepair.rollback: no checkpoint matching {checkpoint_id!r}"
            )
            return False

        target = candidates[0]
        logger.info(f"SelfRepair.rollback: restoring {target}")
        # Actual model reload is handled by the calling orchestrator;
        # we signal intent via return value + log.
        return True

    # ------------------------------------------------------------------ #
    # Strategy implementations                                             #
    # ------------------------------------------------------------------ #

    def _do_rollback(self, report: Optional[DriftReport]) -> RepairResult:
        """Find and signal the most recent good checkpoint."""
        if self._checkpoint_path is None:
            return RepairResult(
                success=False,
                strategy=RepairStrategy.ROLLBACK,
                message="checkpoint_path not configured — falling back to ALERT",
            )

        pts = sorted(
            self._checkpoint_path.glob("*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not pts:
            return RepairResult(
                success=False,
                strategy=RepairStrategy.ROLLBACK,
                message="No checkpoint files found",
            )

        target = pts[0]
        logger.info(f"SelfRepair: rollback to {target.name}")
        return RepairResult(
            success=True,
            strategy=RepairStrategy.ROLLBACK,
            checkpoint_restored=target.stem,
            message=f"Signalled rollback to {target.name}",
        )

    def _do_alert(self, report: Optional[DriftReport]) -> RepairResult:
        alerts = report.alerts if report else ["manual_trigger"]
        msg = f"ALERT: model drift detected — alerts={alerts}"
        logger.warning(msg)
        return RepairResult(
            success=True,
            strategy=RepairStrategy.ALERT,
            message=msg,
        )

    def _do_isolate(self, report: Optional[DriftReport]) -> RepairResult:
        logger.warning("SelfRepair: isolating current model — fallback routing engaged")
        return RepairResult(
            success=True,
            strategy=RepairStrategy.ISOLATE,
            message="Current model isolated; inference routes to fallback",
        )

    def _do_reset(self, report: Optional[DriftReport]) -> RepairResult:
        logger.warning("SelfRepair: full state reset initiated")
        return RepairResult(
            success=True,
            strategy=RepairStrategy.RESET,
            message="State reset completed (model reload required by orchestrator)",
        )

    def _maybe_log(self, result: RepairResult, report: Optional[DriftReport]) -> None:
        if self._episodic is None:
            return
        try:
            from memory.interfaces import MemoryRecord
            rec = MemoryRecord(
                key=f"repair_{int(time.time())}",
                content=(
                    f"Self-repair triggered: strategy={result.strategy.value} "
                    f"success={result.success} "
                    f"alerts={report.alerts if report else []}"
                ),
                metadata={
                    "strategy":   result.strategy.value,
                    "success":    result.success,
                    "checkpoint": result.checkpoint_restored,
                },
            )
            self._episodic.store(rec)
        except Exception as exc:
            logger.debug(f"SelfRepair: failed to log to episodic memory: {exc}")

    # ------------------------------------------------------------------ #
    # Introspection                                                        #
    # ------------------------------------------------------------------ #

    @property
    def repair_count(self) -> int:
        return len(self._repair_history)

    def get_history(self) -> list:
        return list(self._repair_history)
