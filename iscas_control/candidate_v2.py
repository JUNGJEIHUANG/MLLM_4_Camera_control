"""Metric-aligned / quality-constrained candidate prototype.

Not used in the historical real trials. This version is unvalidated on fresh
hardware data: a code/test pass is NOT an advantage claim. Inherited empirical
per-point factors do not become confidence bounds merely by aggregating them.
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np

from .models import TaskModel
from .types import Decision, Prediction, TaskKind


def aggregate_task_error(errors, kind):
    if not isinstance(kind, TaskKind):
        raise ValueError("Explicit TaskKind required")
    e = np.asarray(errors, dtype=float)
    if e.ndim != 1 or not len(e) or not np.isfinite(e).all() or (e < 0).any():
        raise ValueError("Finite nonnegative per-target errors required")
    if kind == TaskKind.LOCALIZATION:
        return float(np.sqrt(np.mean(e**2)))
    if kind in (TaskKind.RELATIVE_INTENSITY, TaskKind.RELATIVE_CONTRAST):
        return float(e.max())
    raise ValueError("Explicit task kind required")


class MetricAlignedTaskModel(TaskModel):
    """Same physical template/noise model, different explicitly versioned metric."""
    def observed(self, patches, action, task):
        base = super().observed(patches, action, task)
        risk = aggregate_task_error(self.errors(base["points"], action, task.kind), task.kind)
        return {**base, "error_upper": risk,
                "qualified": all(p.valid for p in base["points"]) and risk <= task.tolerance}

    def predict(self, action, task, current_points=None, current_action=None):
        base = super().predict(action, task, current_points, current_action)
        m, e = self.actions[action.key], self.estimator
        scale = 1.
        if current_points is not None:
            current = self.actions[current_action.key]
            ratios = [p.flux / f for p, f in zip(current_points, current.flux) if p.snr >= 3 and f > 0]
            if len(ratios) >= max(1, len(current_points) // 2):
                scale = float(np.median(ratios))
        patches = []
        for p in m.mean:
            beta = np.linalg.lstsq(e.design[e.ring], p[e.ring], rcond=None)[0]
            bg = e.design @ beta
            patches.append(bg + scale * (p-bg))
        points = e.measure(np.stack(patches), m.variance * max(1., scale))
        risk = aggregate_task_error(self.errors(points, action, task.kind), task.kind)
        factor = m.location_factor if task.kind == TaskKind.LOCALIZATION else m.contrast_factor
        return replace(base, error=risk/max(factor, 1.), upper_error=risk,
                       loss=(risk/task.tolerance)**2 + 4*float(base.min_snr < 5 or base.saturation_risk),
                       reason=("V2_METRIC_ALIGNED_EMPIRICAL_RISK_NOT_COVERAGE_GUARANTEE" if base.supported
                               else "V2_OUT_OF_DOMAIN_OR_PRIOR_ONLY"))


def select_quality_feasible(predictions, current, task, budget, cost, prefer_cost):
    """Equal quality/domain/budget gates; only the objective differs.

    Do not collect indefinitely where the frozen model predicts no action can
    meet the task. That stop is a failure/unsupported result, not efficiency.
    """
    candidates = []
    for p in predictions:
        if not isinstance(p, Prediction):
            raise ValueError("Explicit candidate predictions required")
        c = cost.estimate(current, p.action)
        if (p.supported and not p.saturation_risk and p.min_snr >= 5
                and p.upper_error <= task.tolerance
                and budget.permits(c["frames"], c["ms"])):
            objective = (c["ms"], c["frames"], p.upper_error) if prefer_cost else (p.upper_error,)
            candidates.append((objective, p, c))
    if not candidates:
        return None
    return min(candidates, key=lambda x: x[0])[1].action


class QualityConstrainedController:
    METHODS = ("v2_quality_only", "v2_cost_feasible")

    def __init__(self, method="v2_cost_feasible"):
        if method not in self.METHODS:
            raise ValueError(method)
        self.method = method

    def choose(self, executor):
        if not isinstance(executor.model, MetricAlignedTaskModel):
            raise ValueError("V2 requires an explicitly metric-aligned model, not silent reuse of V1 semantics")
        if not executor.measurements:
            raise ValueError("Fresh observations required")
        if all(o["qualified"] for o in executor.measurements):
            return Decision(None, executor.task.version, executor.observation_seq, "TASK_PROXY_MET")
        points = executor.measurements[-1]["points"]
        predictions = tuple(executor.model.predict(m.action, executor.task, points, executor.current)
                            for m in executor.model.actions.values())
        action = select_quality_feasible(predictions, executor.current, executor.task,
                                         executor.budget, executor.cost,
                                         self.method == "v2_cost_feasible")
        reason = ("NO_PREDICTED_QUALITY_FEASIBLE_CANDIDATE" if action is None
                  else "V2_QUALITY_CONSTRAINT_THEN_COST" if self.method == "v2_cost_feasible"
                  else "V2_QUALITY_CONSTRAINT_THEN_MINIMUM_RISK")
        return Decision(action, executor.task.version, executor.observation_seq, reason, predictions)
