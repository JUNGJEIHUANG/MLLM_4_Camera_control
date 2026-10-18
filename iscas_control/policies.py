from __future__ import annotations

import numpy as np
from .types import Decision, TaskKind


class Controller:
    METHODS = ("fixed", "roi_ae", "task_rule", "full", "no_cost", "no_task_model")

    def __init__(self, method="full", cost_weight=1., hysteresis=.05):
        if method not in self.METHODS:
            raise ValueError(method)
        self.method, self.cost_weight, self.hysteresis = method, cost_weight, hysteresis
        self.probes = 0

    def choose(self, executor):
        model, task, current = executor.model, executor.task, executor.current
        obs = executor.measurements
        if not obs:
            raise ValueError("A fresh initial observation is required")
        points = obs[-1]["points"]
        prior = tuple(model.predict(m.action, task) for m in model.actions.values())
        predictions = tuple(model.predict(m.action, task, points, current) for m in model.actions.values())
        def decision(action, reason):
            return Decision(action, task.version, executor.observation_seq, reason, predictions)
        if all(o["qualified"] for o in obs):
            return decision(None, "TASK_PROXY_MET")
        if self.method == "fixed":
            best = min(prior, key=lambda p: p.loss)
            return decision(best.action, "DEVELOPMENT_FIXED")
        if self.method == "roi_ae":
            best = min(prior, key=lambda p: p.loss)
            target = np.mean(model.actions[best.action.key].mean)
            actual_background = np.mean([p.background for p in points])
            scale = target / max(actual_background, 1.)
            desired = np.mean(model.actions[current.key].mean) * scale
            m = min(model.actions.values(), key=lambda m: abs(np.mean(m.mean)-desired))
            return decision(m.action, "TASK_TUNED_BRIGHTNESS")
        if self.method == "task_rule":
            # Strong task-aware baseline: same temporal-noise inputs and legal model.
            # Greedy rule uses signal/noise, not candidate error prediction/cost objective.
            def quality(m):
                return min(p.snr for p in model.estimator.measure(m.mean, m.variance))
            best = max(model.actions.values(), key=quality)
            return decision(best.action, "GREEDY_TASK_SIGNAL_NOISE")
        # Full and no_cost have IDENTICAL predictions, legal domains and budgets.
        # Only action objective / hysteresis differ for the K2 ablation.
        feasible = []
        for p in predictions:
            c = executor.cost.estimate(current, p.action)
            if p.supported and not p.saturation_risk and p.min_snr >= 5 and executor.budget.permits(c["frames"], c["ms"]):
                loss = p.loss
                if self.method == "no_task_model":
                    loss = (5/max(p.min_snr, .01))**2
                weight = 0. if self.method == "no_cost" else self.cost_weight
                objective = loss + weight*c["ms"]/executor.budget.max_ms
                feasible.append((objective, p, c))
        if not feasible:
            # At most one bounded prior-based probe; never report it as measurement success.
            choices = [p for p in prior if not p.saturation_risk and p.action != current]
            if self.probes == 0 and choices:
                best = min(choices, key=lambda p: p.loss)
                cost = executor.cost.estimate(current, best.action)
                if executor.budget.permits(cost["frames"], cost["ms"]):
                    self.probes += 1
                    return decision(best.action, "BOUNDED_PRIOR_PROBE")
            return decision(None, "NO_SUPPORTED_BUDGET_FEASIBLE_CANDIDATE")
        _, best, _ = min(feasible, key=lambda item: item[0])
        hold = next((v for v in feasible if v[1].action == current), None)
        h = 0. if self.method == "no_cost" else self.hysteresis
        if hold and best.action != current and hold[0]-min(feasible, key=lambda v: v[0])[0] <= h:
            return decision(current, "HYSTERESIS_HOLD")
        return decision(best.action, "QUALITY_ONLY" if self.method == "no_cost" else "QUALITY_AND_ACTION_COST")
