from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import numpy as np

from .measurement import PatchEstimator, temporal_variance
from .types import Action, Prediction, Task, TaskKind


@dataclass
class ActionModel:
    action: Action
    mean: np.ndarray
    variance: np.ndarray
    flux: np.ndarray
    rate_response: np.ndarray
    location_bias: np.ndarray
    contrast_bias: np.ndarray
    location_factor: float = 1.5
    contrast_factor: float = 1.5
    calibration_frames: int = 0


class TaskModel:
    def __init__(self, estimator, actions, anchor_rate, provenance=None):
        self.estimator = estimator
        self.actions = {m.action.key: m for m in actions}
        self.anchor_rate = np.asarray(anchor_rate, float)
        self.provenance = provenance or {}

    def measure(self, patches, action):
        return self.estimator.measure(patches, self.actions[action.key].variance)

    def errors(self, points, action, kind):
        m = self.actions[action.key]
        if kind == TaskKind.LOCALIZATION:
            e = np.array([p.centroid_sigma for p in points])
            return np.sqrt(e**2 + m.location_bias**2) * m.location_factor
        flux = np.array([p.flux for p in points])
        sigma = np.array([p.flux_sigma for p in points])
        denom = action.nominal_ms * m.rate_response
        relative_sigma = sigma / np.maximum(abs(denom * self.anchor_rate), 1e-9)
        return np.sqrt(relative_sigma**2 + m.contrast_bias**2) * m.contrast_factor

    def observed(self, patches, action, task):
        if task.kind == TaskKind.RELATIVE_INTENSITY and self.estimator.polarity != 1:
            raise ValueError("Bright-target intensity requires a positive-polarity model; dark contrast is not its substitute")
        points = self.measure(patches, action)
        errors = self.errors(points, action, task.kind)
        return {"points": points, "error_upper": float(max(errors)),
                "qualified": all(p.valid for p in points) and bool(max(errors) <= task.tolerance)}

    def predict(self, action, task, current_points=None, current_action=None):
        if task.kind == TaskKind.RELATIVE_INTENSITY and self.estimator.polarity != 1:
            raise ValueError("Bright-target intensity requires its own positive-polarity development model")
        m = self.actions[action.key]
        scale, supported = 1., True
        if current_points is not None:
            current = self.actions[current_action.key]
            ratios = [p.flux / f for p, f in zip(current_points, current.flux)
                      if p.snr >= 3 and f > 0]
            if len(ratios) >= max(1, len(current_points) // 2):
                scale = float(np.median(ratios))
                supported = .5 <= scale <= 2.
            # A low-SNR observation cannot justify zeroing all candidate signals.
            # Use a table-prior probe, explicitly not a reliable adaptive estimate.
            else:
                supported = False
        # Bounded empirical signal template; this is a prediction, never a RAW result.
        e = self.estimator
        patches = []
        for p in m.mean:
            xr = e.design[e.ring]
            beta = np.linalg.lstsq(xr, p[e.ring], rcond=None)[0]
            bg = e.design @ beta
            patches.append(bg + scale * (p - bg))
        variance = m.variance * max(1., scale)
        pts = e.measure(np.stack(patches), variance)
        errors = self.errors(pts, action, task.kind)
        upper = float(max(errors))
        min_snr = float(min(p.snr for p in pts))
        sat = any(p.clipped for p in pts)
        loss = (upper / task.tolerance)**2 + 4 * float(min_snr < 5 or sat)
        return Prediction(action, upper / max(m.location_factor if task.kind == TaskKind.LOCALIZATION
                                              else m.contrast_factor, 1.), upper, min_snr,
                          supported, sat, float(loss),
                          "bounded_empirical_prediction" if supported else "OUT_OF_DOMAIN_OR_PRIOR_ONLY")

    def save(self, path):
        p = Path(path)
        rows = []
        for m in self.actions.values():
            d = asdict(m)
            for k, v in list(d.items()):
                if isinstance(v, np.ndarray):
                    d[k] = v.tolist()
            rows.append(d)
        p.write_text(json.dumps({"seeds": self.estimator.seeds, "polarity": self.estimator.polarity,
                                 "anchor_rate": self.anchor_rate.tolist(), "actions": rows,
                                 "provenance": self.provenance}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        j = json.loads(Path(path).read_text(encoding="utf-8"))
        models = []
        for d in j["actions"]:
            d["action"] = Action(**d["action"])
            for k in ("mean", "variance", "flux", "rate_response", "location_bias", "contrast_bias"):
                d[k] = np.asarray(d[k], float)
            models.append(ActionModel(**d))
        return cls(PatchEstimator(j["seeds"], polarity=j["polarity"]), models,
                   j["anchor_rate"], j["provenance"])


def fit_real_summary(summary_path, seeds, hash_check=True, polarity=-1):
    """Fit ONLY real development frames; references distinct from fitted images.

    n>=8: fit first 3, calibrate factors next 2, leave last >=3 unused.
    n<8: initialization fit only; no held-out quality claim.
    """
    path = Path(summary_path)
    j = json.loads(path.read_text(encoding="utf-8"))
    estimator = PatchEstimator(seeds, polarity=polarity)
    receipts, arrays = [], {}
    for i, f in enumerate(j["frames"]):
        b = Path(f["raw"]).read_bytes()
        h = hashlib.sha256(b).hexdigest()
        if len(b) != 8912896 or (hash_check and h != f["sha256"]):
            raise ValueError("RAW integrity mismatch")
        image = np.frombuffer(b, dtype="<u2").reshape(2048, 2176)
        if image.max() > 4095:
            raise ValueError("Invalid 12-bit container")
        arrays[i] = estimator.patches(image)
        receipts.append({"raw": f["raw"], "sha256": h})
    refs = [arrays[i] for i, f in enumerate(j["frames"])
            if f["label"] == "reference_before" and f["frame_index"] > 0]
    if len(refs) < 3:
        raise ValueError("Independent development reference repetitions required")
    ref_mean = np.mean(refs, axis=0)
    ref_var = temporal_variance(refs)
    ref_points = estimator.measure(ref_mean, ref_var / len(refs))
    if not all(p.valid for p in ref_points):
        raise ValueError("Development reference target invalid: no ROI reselection in fitter")
    ref_spec = j["plan"].get("reference_action", {"gain": 75.2, "lines": 256})
    ref_action = Action(**ref_spec)
    anchor_rate = np.array([p.flux for p in ref_points]) / ref_action.nominal_ms
    models, splits, unsupported = [], {}, []
    for sid, (gain, lines) in enumerate(j["plan"]["settings_gain_lines"]):
        indices = [i for c in j["cells"] if c["setting_id"] == sid for i in c["frame_indices"][1:]]
        samples = np.stack([arrays[i] for i in indices])
        if len(samples) < 3:
            raise ValueError("Three development observations per action required")
        nf = 3 if len(samples) >= 8 else len(samples)
        fit = samples[:nf]
        variance = temporal_variance(fit)
        mean = fit.mean(axis=0)
        points = estimator.measure(mean, variance / len(fit))
        flux = np.array([p.flux for p in points])
        action = Action(gain, lines)
        if (flux <= 0).any() or any(p.clipped for p in points):
            unsupported.append({"action": action.key, "frame_indices": indices,
                                "reason": "nonpositive signal or saturated development patch; RAW retained"})
            continue
        response = flux / (action.nominal_ms * anchor_rate)
        bias = np.array([np.hypot(p.x-r.x, p.y-r.y) for p, r in zip(points, ref_points)])
        models.append(ActionModel(action, mean, variance, flux, response, bias,
                                  np.zeros(len(points)), calibration_frames=nf))
        splits[action.key] = {"fit": indices[:nf], "calibration": indices[3:5] if len(samples) >= 8 else [],
                              "validation": indices[5:] if len(samples) >= 8 else []}
    if not models:
        raise ValueError("No supported action models; keep acquisition, do not start controller trials")
    model = TaskModel(estimator, models, anchor_rate,
                      {"development_summary": str(path), "summary_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "receipts": receipts, "splits": splits, "unsupported_actions": unsupported,
                       "reference_action": ref_spec,
                       "scope": "scene-specific relative DN model, not electron calibration or independent-layout evidence"})
    # Factors calibrated on disjoint calibration samples, never reported as validation.
    for m in models:
        loc_ratios, photo_ratios = [], []
        for i in splits[m.action.key]["calibration"]:
            pts = model.measure(arrays[i], m.action)
            for pid, (p, r) in enumerate(zip(pts, ref_points)):
                raw_loc = np.sqrt(p.centroid_sigma**2 + m.location_bias[pid]**2)
                err = np.hypot(p.x-r.x, p.y-r.y)
                loc_ratios.append(err / max(raw_loc, 1e-6))
                rate = p.flux / (m.action.nominal_ms * m.rate_response[pid])
                rel = abs(rate / anchor_rate[pid] - 1)
                sig = p.flux_sigma / max(m.action.nominal_ms*m.rate_response[pid]*anchor_rate[pid], 1e-6)
                photo_ratios.append(rel / max(sig, 1e-6))
        if loc_ratios:
            m.location_factor = max(1., float(np.quantile(loc_ratios, .9)))
            m.contrast_factor = max(1., float(np.quantile(photo_ratios, .9)))
    validation = []
    for m in models:
        for i in splits[m.action.key]["validation"]:
            pts = model.measure(arrays[i], m.action)
            for kind in (TaskKind.LOCALIZATION, TaskKind.RELATIVE_CONTRAST if polarity == -1 else TaskKind.RELATIVE_INTENSITY):
                predicted = model.errors(pts, m.action, kind)
                actual = [np.hypot(p.x-r.x, p.y-r.y) if kind == TaskKind.LOCALIZATION
                          else abs(p.flux/(m.action.nominal_ms*m.rate_response[k]*anchor_rate[k])-1)
                          for k, (p, r) in enumerate(zip(pts, ref_points))]
                validation.append({"action": m.action.key, "frame_index": i, "task": kind.value,
                                   "valid": [p.valid for p in pts], "predicted_upper": predicted.tolist(),
                                   "actual_error_vs_reference": list(map(float, actual))})
    return model, {"split": splits, "validation": validation, "unsupported_actions": unsupported,
                   "reference_points": [p.to_dict() for p in ref_points],
                   "not_independent_layouts": True,
                   "validation_count": len(validation)}
