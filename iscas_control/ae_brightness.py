"""Brightness-AE prototypes. No hardware imports, no novelty/benefit guarantee."""
from dataclasses import dataclass, asdict
import math
import numpy as np

ROI = (768, 1280, 832, 1344)  # y0,y1,x0,x1; common metering, not sensor/output ROI
GRID = (40, 80, 128, 192, 256, 384, 512)


def measure(image, roi=ROI):
    y0, y1, x0, x1 = roi
    a = np.asarray(image)[y0:y1, x0:x1]
    if a.size != (y1-y0)*(x1-x0) or not a.size:
        raise ValueError("Invalid common metering ROI")
    return {"mean_dn": float(a.mean()), "clip_fraction": float(np.mean(a >= 4095)),
            "p95_dn": float(np.percentile(a, 95))}


def qualified(obs, cfg):
    return (abs(obs["mean_dn"]-cfg["target_dn"]) <= cfg["tolerance_dn"]
            and obs["clip_fraction"] <= cfg["max_roi_clip_fraction"])


def monotone(values):
    """Equal-weight pool-adjacent-violators; development curve only."""
    blocks = []
    for value in values:
        blocks.append([float(value), 1])
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            b, a = blocks.pop(), blocks.pop()
            n = a[1]+b[1]
            blocks.append([(a[0]*a[1]+b[0]*b[1])/n, n])
    return np.array([value for value, n in blocks for _ in range(n)])


@dataclass
class ResponseModel:
    lines: list
    mean_dn: list
    offset_dn: float
    fit_scope: str = "same-scene development RAW response, not radiometric calibration"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def fit(cls, rows):
        rows = sorted(rows, key=lambda r: r["lines"])
        x = np.array([r["lines"] for r in rows], float)
        y = np.array([r["mean_dn"] for r in rows], float)
        if len(x) < 4 or np.any(np.diff(x) <= 0) or not np.isfinite(y).all():
            raise ValueError("At least four distinct finite development settings required")
        slope, intercept = np.polyfit(x, y, 1)
        if slope <= 0 or y.max()-y.min() < 40:
            raise ValueError("Development response not usable for this brightness AE")
        # Intercept is DN-domain empirical bias, never an accepted E0 calibration.
        offset = float(np.clip(intercept, 0., y.min()-1.))
        return cls(x.tolist(), monotone(y).tolist(), offset)


class BrightnessAE:
    METHODS = ("ratio", "offset_ratio", "response_model")

    def __init__(self, method, model, cfg):
        if method not in self.METHODS:
            raise ValueError(method)
        self.method, self.model, self.cfg = method, model, cfg

    def choose(self, lines, obs):
        c, m = self.cfg, self.model
        mean = obs["mean_dn"]
        if not math.isfinite(mean) or mean < 0:
            raise ValueError("Finite nonnegative observation required")
        if qualified(obs, c):
            return lines
        if obs["clip_fraction"] > c["max_roi_clip_fraction"]:
            desired = lines/2  # identical saturation safety branch for all policies
        elif self.method == "ratio":
            factor = (c["target_dn"]/max(mean, 1.))**c["ratio_alpha"]
            desired = lines*factor
        elif self.method == "offset_ratio":
            desired = lines*(c["target_dn"]-m.offset_dn)/max(mean-m.offset_dn, 1.)
        else:
            reference = float(np.interp(lines, m.lines, m.mean_dn))
            scale = (mean-m.offset_dn)/max(reference-m.offset_dn, 1.)
            target_reference = m.offset_dn+(c["target_dn"]-m.offset_dn)/max(scale, .01)
            desired = float(np.interp(target_reference, m.mean_dn, m.lines))
        desired = float(np.clip(desired, lines/c["max_step_factor"], lines*c["max_step_factor"]))
        new = int(np.clip(round(desired), c["min_lines"], c["max_lines"]))
        if new == lines and not qualified(obs, c):
            direction = 1 if mean < c["target_dn"] else -1
            new = int(np.clip(lines+direction, c["min_lines"], c["max_lines"]))
        return new


def tune_ratio_offline(model, cfg):
    """Development-only response replay; output is NOT a real AE result."""
    candidates = []
    for alpha in (.5, .7, 1., 1.5, 2.):
        test = {**cfg, "ratio_alpha": alpha}
        cases = []
        for start in (cfg["min_lines"], cfg["max_lines"]):
            t, updates, trajectory = start, 0, []
            for _ in range(12):
                mean = float(np.interp(t, model.lines, model.mean_dn))
                trajectory.append(mean)
                obs = {"mean_dn": mean, "clip_fraction": 0.}
                if qualified(obs, test):
                    break
                new = BrightnessAE("ratio", model, test).choose(t, obs)
                if new == t:
                    break
                t, updates = new, updates+1
            ok = qualified(obs, test)
            cases.append({"start": start, "updates": updates, "success": ok,
                          "trajectory_dn": trajectory})
        rank = (sum(not r["success"] for r in cases), sum(r["updates"] for r in cases),
                sum(abs(r["trajectory_dn"][-1]-cfg["target_dn"]) for r in cases))
        candidates.append({"alpha": alpha, "rank": list(rank), "cases": cases})
    best = min(candidates, key=lambda r: tuple(r["rank"]))
    return best["alpha"], {"purpose": "baseline tuning on development replay, NOT measured convergence", "candidates": candidates}
