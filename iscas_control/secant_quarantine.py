"""Cold-start online secant-offset AE and transition-quarantine execution ablation.

Preserves frozen hashes of existing iscas_control modules by residing in an
isolated module. No hardware imports on module load.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import time
import numpy as np

from .ae_brightness import BrightnessAE, ResponseModel, measure, qualified
from .ae_published import PublishedAE, make_policy as make_base_policy
from .ae_experiments import stripped
from .execution import PAYLOAD_BYTES
from .types import Action


@dataclass(frozen=True)
class ShortExposureAction:
    gain: float
    lines: int

    def __post_init__(self):
        if self.gain not in (1.04, 13.43, 75.2):
            raise ValueError("Gain outside verified STD nominal settings")
        if (
            isinstance(self.lines, bool)
            or not isinstance(self.lines, (int, np.integer))
            or not 8 <= int(self.lines) <= 512
        ):
            raise ValueError("Exposure lines outside candidate 8..512 envelope")

    @property
    def nominal_ms(self) -> float:
        return round(int(self.lines) * 0.0065, 6)

    @property
    def key(self) -> str:
        return f"g{self.gain:g}_l{int(self.lines)}"


@dataclass
class SecantEstimate:
    t_prev: int
    y_prev: float
    t_curr: int
    y_curr: float
    slope_dn_per_line: float
    raw_offset_dn: float
    projected_offset_dn: float
    valid_positive_slope: bool

    def to_dict(self):
        return asdict(self)


class SecantOffsetAE:
    """Calibration-free two-point online secant-offset exposure controller.

    Requires zero pre-scan development frames (N_dev = 0):
    - Step 1 (only 1 unclipped exposure observed): takes a bounded ratio step
      with exponent init_alpha (default 1.0, parameter-free uncorrected ratio).
    - Step >= 2 (at least 2 distinct unclipped exposures observed): estimates
      the local affine offset b_hat from the two most recent distinct unclipped
      observations (t_a, y_a) and (t_b, y_b), projects b_hat onto the physically
      admissible interval [min_offset_dn, min(y_a, y_b) - 1], and applies the
      offset-corrected ratio update.
    """
    METHODS = ("secant_offset", "secant_offset_alpha2")

    def __init__(self, cfg, init_alpha=1.0, min_offset_dn=0.0):
        if init_alpha <= 0 or not math.isfinite(init_alpha):
            raise ValueError("Positive finite init_alpha required")
        self.cfg = cfg
        self.init_alpha = float(init_alpha)
        self.min_offset_dn = float(min_offset_dn)
        self.history: list[tuple[int, float]] = []
        self.last_estimate: SecantEstimate | None = None
        self.last_decision: dict | None = None
        self.valid_offset_dn: float | None = None

    def reset(self):
        self.history.clear()
        self.last_estimate = None
        self.last_decision = None
        self.valid_offset_dn = None

    def _record_unclipped(self, lines: int, mean_dn: float):
        # Update or append so the last two entries always have distinct exposure lines.
        if self.history and self.history[-1][0] == lines:
            self.history[-1] = (int(lines), float(mean_dn))
        else:
            # Remove any older duplicate of `lines` to keep recent distinct pairs clean
            self.history = [(t, y) for t, y in self.history if t != lines]
            self.history.append((int(lines), float(mean_dn)))

    def choose(self, lines: int, obs: dict) -> int:
        c = self.cfg
        mean = float(obs["mean_dn"])
        clip = float(obs["clip_fraction"])
        if not math.isfinite(mean) or mean < 0:
            raise ValueError("Finite nonnegative observation required")
        if clip <= c["max_roi_clip_fraction"]:
            self._record_unclipped(lines, mean)
        if qualified(obs, c):
            self.last_decision = {"branch": "common_hold", "new_lines": lines,
                                  "offset_hat_dn": self.valid_offset_dn}
            return lines
        if clip > c["max_roi_clip_fraction"]:
            desired = lines / 2.0
            branch = "common_clipping_safety"
            est_dict = None
        elif len(self.history) < 2:
            factor = (c["target_dn"] / max(mean, 1.0)) ** self.init_alpha
            desired = lines * factor
            branch = "cold_start_ratio_probe"
            est_dict = None
        else:
            (t_a, y_a), (t_b, y_b) = self.history[-2], self.history[-1]
            dt = float(t_b - t_a)
            dy = float(y_b - y_a)
            slope = dy / dt
            raw_b = y_b - slope * t_b
            upper_b = max(self.min_offset_dn, min(y_a, y_b) - 1.0)
            if slope > 1e-6:
                b_hat = float(np.clip(raw_b, self.min_offset_dn, upper_b))
                self.valid_offset_dn = b_hat
                valid_slope = True
                branch = "online_secant_offset"
            else:
                # Nonpositive slope occurs if stale unquarantined feedback is fed in
                # or under severe noise on a 1-line step; fall back safely.
                b_hat = self.valid_offset_dn if self.valid_offset_dn is not None else self.min_offset_dn
                b_hat = float(np.clip(b_hat, self.min_offset_dn, upper_b))
                valid_slope = False
                branch = "secant_nonpositive_slope_fallback"
            est = SecantEstimate(
                t_prev=t_a, y_prev=y_a, t_curr=t_b, y_curr=y_b,
                slope_dn_per_line=slope, raw_offset_dn=raw_b,
                projected_offset_dn=b_hat, valid_positive_slope=valid_slope
            )
            self.last_estimate = est
            est_dict = est.to_dict()
            desired = lines * (c["target_dn"] - b_hat) / max(mean - b_hat, 1.0)

        unconstrained = float(desired)
        desired = float(np.clip(desired, lines / c["max_step_factor"], lines * c["max_step_factor"]))
        new = int(np.clip(round(desired), c["min_lines"], c["max_lines"]))
        if new == lines and not qualified(obs, c):
            direction = 1 if mean < c["target_dn"] else -1
            new = int(np.clip(lines + direction, c["min_lines"], c["max_lines"]))
        self.last_decision = {
            "branch": branch,
            "observed_mean_dn": mean,
            "requested_lines": unconstrained,
            "new_lines": new,
            "secant_estimate": est_dict,
        }
        return new


def make_extended_policy(method: str, model: ResponseModel, protocol: dict):
    """Factory supporting the 5 baseline policies plus cold-start secant policies."""
    if method == "secant_offset":
        return SecantOffsetAE(protocol["cfg"], init_alpha=1.0)
    if method == "secant_offset_alpha2":
        return SecantOffsetAE(protocol["cfg"], init_alpha=protocol["cfg"].get("ratio_alpha", 2.0))
    return make_base_policy(method, model, protocol)


def run_episode_quarantine_ablation(
    camera,
    method: str,
    start_lines: int,
    protocol: dict,
    eid: str,
    quarantine_mode: str = "always",
    action_cls=ShortExposureAction,
):
    """Run an AE episode under explicit transition-quarantine rules.

    quarantine_mode:
    - 'always': Standard execution; every parameter update captures 1 quarantined
      transition frame (N = G + 2U + W).
    - 'none': Unquarantined execution; no transition frame is isolated after
      camera.apply(), so the first post-command frame is treated directly as an
      evaluation frame for qualification and feedback control.
    - 'gain_only': Selective quarantine; isolates 1 transition frame if and only
      if the command changed hardware gain (command['gain_changed'] is True).
    """
    if quarantine_mode not in ("always", "none", "gain_only"):
        raise ValueError(f"Unsupported quarantine_mode: {quarantine_mode}")
    cfg = protocol["cfg"]
    model = ResponseModel(**protocol["model"])
    policy = make_extended_policy(method, model, protocol)
    camera.apply(action_cls(cfg["gain"], start_lines))
    reset = [stripped(camera.capture(eid + "_common_reset")) for _ in range(2)]
    current = start_lines
    start = time.perf_counter_ns()
    ep = {
        "episode_id": eid,
        "method": method,
        "quarantine_mode": quarantine_mode,
        "start_lines": start_lines,
        "frames": [],
        "commands": [],
        "common_reset": reset,
        "start_tick_ns": start,
    }
    run, first_entry, stop = 0, None, "UNKNOWN"
    stale_feedback_events = 0
    slope_collapse_events = 0
    while True:
        elapsed = (time.perf_counter_ns() - start) / 1e6
        if len(ep["frames"]) + 1 > cfg["max_online_frames"] or elapsed >= cfg["wall_budget_ms"]:
            stop = "BUDGET"
            break
        f = camera.capture(eid + "_evaluation")
        obs = measure(f["image"], tuple(cfg["roi"]))
        is_stale_frame = bool(f.get("stale_transition_unquarantined", False))
        if is_stale_frame:
            stale_feedback_events += 1
        rec = {
            **stripped(f),
            "measurement": obs,
            "qualified": qualified(obs, cfg),
            "lines": current,
            "role_online": "evaluation",
            "stale_transition_unquarantined": is_stale_frame,
        }
        ep["frames"].append(rec)
        run = run + 1 if rec["qualified"] else 0
        now_ms = (time.perf_counter_ns() - start) / 1e6
        if run >= cfg["first_accept_frames"] and first_entry is None:
            first_entry = {"ms": now_ms, "frames": len(ep["frames"])}
        if run >= cfg["stable_frames"]:
            stop = "STABLE_TARGET_REACHED"
            break
        if rec["qualified"]:
            continue
        if len(ep["commands"]) >= cfg["max_updates"]:
            stop = "UPDATE_LIMIT"
            break
        new = policy.choose(current, obs)
        if getattr(policy, "last_decision", None) is not None:
            rec["policy_decision"] = policy.last_decision
            if policy.last_decision.get("branch") == "secant_nonpositive_slope_fallback":
                slope_collapse_events += 1
        if new == current:
            stop = "ACTION_LIMIT_OUTSIDE_TARGET"
            break
        need_quarantine = (
            quarantine_mode == "always"
            or (quarantine_mode == "gain_only" and False)  # exposure-only within fixed gain
        )
        required_frames = 2 if need_quarantine else 1
        if len(ep["frames"]) + required_frames > cfg["max_online_frames"]:
            stop = "FRAME_BUDGET_BEFORE_CHANGE"
            break
        command = camera.apply(action_cls(cfg["gain"], new))
        ep["commands"].append(command)
        current = new
        if quarantine_mode == "always" or (quarantine_mode == "gain_only" and command.get("gain_changed", False)):
            q = camera.capture(eid + "_transition")
            ep["frames"].append({
                **stripped(q),
                "measurement": measure(q["image"], tuple(cfg["roi"])),
                "role_online": "transition",
                "lines": current,
                "qualified": False,
            })
        run = 0

    end = time.perf_counter_ns()
    evaluations = [f for f in ep["frames"] if f["role_online"] == "evaluation"]
    final = evaluations[-1] if evaluations else None
    elapsed = (end - start) / 1e6
    data_bytes = sum(f["payload_bytes"] + f["header_bytes"] for f in ep["frames"])
    max_bytes = cfg["max_online_frames"] * (PAYLOAD_BYTES + 20)
    overrun = {"ms": max(0.0, elapsed - cfg["wall_budget_ms"]), "bytes": max(0, data_bytes - max_bytes)}
    good = [f for f in evaluations if f["qualified"]]
    changes = [int(r["new"].split("_l")[1]) - int(r["old"].split("_l")[1]) for r in ep["commands"]]
    reversals = sum(a * b < 0 for a, b in zip(changes[:-1], changes[1:]))
    tail = evaluations[-cfg["stable_frames"]:]
    tail_has_stale = any(f.get("stale_transition_unquarantined", False) for f in tail)
    ep.update({
        "end_tick_ns": end,
        "elapsed_ms": elapsed,
        "stop_reason": stop,
        "success": stop == "STABLE_TARGET_REACHED" and not any(overrun.values()),
        "true_settled_success": (
            stop == "STABLE_TARGET_REACHED"
            and not any(overrun.values())
            and not tail_has_stale
        ),
        "first_entry": first_entry,
        "online_frames": len(ep["frames"]),
        "data_bytes": data_bytes,
        "budget_overrun": overrun,
        "updates": len(ep["commands"]),
        "direction_reversals": reversals,
        "stale_feedback_events": stale_feedback_events,
        "slope_collapse_events": slope_collapse_events,
        "transition_frames": sum(f["role_online"] == "transition" for f in ep["frames"]),
        "effective_frames": len(good),
        "frame_cost_per_effective": len(ep["frames"]) / len(good) if good else None,
        "final_lines": current,
        "output": final["raw"] if final else None,
        "final_mean_dn": final["measurement"]["mean_dn"] if final else None,
        "tail_std_dn": float(np.std([f["measurement"]["mean_dn"] for f in tail], ddof=1)) if len(tail) > 1 else None,
        "tail_mae_dn": float(np.mean([abs(f["measurement"]["mean_dn"] - cfg["target_dn"]) for f in tail])) if tail else None,
    })
    return ep
