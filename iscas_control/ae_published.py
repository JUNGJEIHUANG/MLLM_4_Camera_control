"""Published exposure-update laws, adapted to the common RAW-DN executor.

Su & Kuo, ICCE 2015, Eq. (5), DOI 10.1109/ICCE.2015.7066300.
Cvetkovic et al., EURASIP JIVP 2010, Eq. (6), DOI 10.1155/2010/197194.
These are independently implemented update laws, not author software or full
reproductions of either original camera pipeline. No hardware imports.
"""
import math
import numpy as np
from .ae_brightness import BrightnessAE, qualified


class PublishedAE:
    METHODS = ("cvetkovic2010", "su2015")

    def __init__(self, method, cfg, published):
        if method not in self.METHODS:
            raise ValueError(method)
        self.method, self.cfg, self.published = method, cfg, published
        self.last_decision = None
        if not 0 <= published["recursive_k"] <= 1:
            raise ValueError("Paper Eq. (6) requires 0 <= k <= 1")
        if published["su_tmax_lines"] < cfg["max_lines"]:
            raise ValueError("Su anchor cannot be inside the allowed exposure interval")
        if published["su_yfixed_dn"] <= 4095:
            raise ValueError("Su upper ordinate must exceed the 12-bit DN range")

    def choose(self, lines, obs):
        c, p, mean = self.cfg, self.published, obs["mean_dn"]
        if not math.isfinite(mean) or mean < 0:
            raise ValueError("Finite nonnegative observation required")
        if qualified(obs, c):
            self.last_decision = {"branch": "common_hold", "new_lines": lines}
            return lines
        if obs["clip_fraction"] > c["max_roi_clip_fraction"]:
            desired, branch = lines/2, "common_clipping_safety"
        elif self.method == "cvetkovic2010":
            # Eq. (6): t_next = t * (1 + k * (Y_wanted - Y_average)/Y_average).
            desired = lines*(1+p["recursive_k"]*(c["target_dn"]-mean)/max(mean, 1.))
            branch = "cvetkovic2010_eq6"
        else:
            tmax, upper = p["su_tmax_lines"], p["su_yfixed_dn"]
            if lines == tmax:
                # The paper moves the endpoint by one time unit. Physically
                # acquire at the nudged setting before using its brightness.
                desired, branch = lines-1, "su2015_endpoint_nudge"
            else:
                # Eq. (5), equivalently the fixed-anchor secant in X = 1/t.
                denominator = (c["target_dn"]-mean)*lines+(upper-c["target_dn"])*tmax
                if denominator <= 0:
                    raise ValueError("Su modified-secant denominator is nonpositive")
                desired = (upper-mean)*lines*tmax/denominator
                branch = "su2015_eq5"
        unconstrained = float(desired)
        desired = float(np.clip(desired, lines/c["max_step_factor"], lines*c["max_step_factor"]))
        new = int(np.clip(round(desired), c["min_lines"], c["max_lines"]))
        if new == lines:
            direction = 1 if mean < c["target_dn"] else -1
            new = int(np.clip(lines+direction, c["min_lines"], c["max_lines"]))
        self.last_decision = {"branch": branch, "observed_mean_dn": mean,
                              "requested_lines": unconstrained, "new_lines": new}
        return new


def make_policy(method, model, protocol):
    if method in PublishedAE.METHODS:
        return PublishedAE(method, protocol["cfg"], protocol["published"])
    return BrightnessAE(method, model, protocol["cfg"])
