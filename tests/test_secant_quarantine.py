"""Unit and mechanism tests for SecantOffsetAE and transition-quarantine ablation."""
import hashlib
import json
from pathlib import Path
import unittest
import numpy as np

from iscas_control.ae_experiments import DEFAULT
from iscas_control.execution import PAYLOAD_BYTES
from iscas_control.secant_quarantine import (
    SecantOffsetAE,
    make_extended_policy,
    run_episode_quarantine_ablation,
)
from tests.test_ae_brightness import FakeCamera, fixture

PARAMS = {"recursive_k": 1.0, "su_tmax_lines": 512, "su_yfixed_dn": 4096}


class LaggyFakeCamera(FakeCamera):
    """Synthetic camera with configurable 1-frame pipeline lag on exposure/gain changes."""

    def __init__(self, lag_frames=1, offset_dn=160.0, slope_dn=2.0):
        super().__init__()
        self.lag_frames = int(lag_frames)
        self.offset_dn = float(offset_dn)
        self.slope_dn = float(slope_dn)
        self.prev_lines = 308
        self.stale_remaining = 0

    def apply(self, action):
        old_lines = self.current.lines
        rec = super().apply(action)
        if action.lines != old_lines:
            self.prev_lines = old_lines
            self.stale_remaining = self.lag_frames
        return rec

    def capture(self, role):
        self.seq += 1
        if self.stale_remaining > 0:
            effective_lines = self.prev_lines
            self.stale_remaining -= 1
            is_stale = True
        else:
            effective_lines = self.current.lines
            is_stale = False
        value = int(np.clip(round(self.offset_dn + self.slope_dn * effective_lines), 0, 4095))
        image = np.full((2048, 2176), value, dtype=np.uint16)
        return {
            "raw": f"fake_{self.seq}.bin",
            "sha256": "0" * 64,
            "payload_bytes": PAYLOAD_BYTES,
            "header_bytes": 20,
            "role": role,
            "commanded_action": self.current.key,
            "stale_transition_unquarantined": is_stale and role.endswith("evaluation"),
            "image": image,
        }


class SecantQuarantineTests(unittest.TestCase):
    def test_step1_matches_cvetkovic_k1_and_step2_matches_exact_secant(self):
        cfg = {**DEFAULT, "target_dn": 300.0, "tolerance_dn": 15.0, "min_lines": 8, "max_lines": 512}
        p = SecantOffsetAE(cfg, init_alpha=1.0)
        # Step 1: only (t0=8, y0=191.0) known -> uncorrected ratio 8 * (300 / 191) = 12.565 -> 13
        t1 = p.choose(8, {"mean_dn": 191.0, "clip_fraction": 0.0})
        self.assertEqual(t1, 13)
        self.assertEqual(p.last_decision["branch"], "cold_start_ratio_probe")
        # Step 2: observe (t1=13, y1=206.75) -> slope = 15.75/5 = 3.15, b_hat = 191 - 3.15*8 = 165.8
        # Exact target lines = (300 - 165.8) / 3.15 = 42.603 -> 43
        t2 = p.choose(13, {"mean_dn": 206.75, "clip_fraction": 0.0})
        self.assertEqual(t2, 43)
        self.assertEqual(p.last_decision["branch"], "online_secant_offset")
        self.assertAlmostEqual(p.last_estimate.projected_offset_dn, 165.8, places=4)

    def test_secant_equivalence_to_two_point_linear_inversion(self):
        cfg = {**DEFAULT, "target_dn": 300.0, "tolerance_dn": 15.0, "min_lines": 8, "max_lines": 512}
        p = SecantOffsetAE(cfg, init_alpha=1.0)
        p.choose(10, {"mean_dn": 200.0, "clip_fraction": 0.0})
        p.choose(20, {"mean_dn": 235.0, "clip_fraction": 0.0})
        # Two-point linear root: t = 20 + (300 - 235) * (20 - 10) / (235 - 200) = 38.5714286
        expected = 20.0 + (300.0 - 235.0) * (20.0 - 10.0) / (235.0 - 200.0)
        self.assertAlmostEqual(p.last_decision["requested_lines"], expected, places=6)

    def test_clipping_and_nonpositive_slope_safety(self):
        cfg = {**DEFAULT, "target_dn": 300.0, "tolerance_dn": 15.0, "min_lines": 8, "max_lines": 512}
        p = SecantOffsetAE(cfg, init_alpha=1.0)
        # Clipped observation must halve exposure and not pollute unclipped history
        self.assertEqual(p.choose(512, {"mean_dn": 4065.0, "clip_fraction": 0.95}), 256)
        self.assertEqual(len(p.history), 0)
        # First unclipped observation
        p.choose(10, {"mean_dn": 200.0, "clip_fraction": 0.0})
        # Second unclipped observation with nonpositive slope (e.g. stale feedback y1 == y0)
        t_fallback = p.choose(15, {"mean_dn": 200.0, "clip_fraction": 0.0})
        self.assertEqual(p.last_decision["branch"], "secant_nonpositive_slope_fallback")
        self.assertGreater(t_fallback, 15)

    def test_quarantine_prevents_stale_feedback_oscillation(self):
        cfg = {**DEFAULT, "target_dn": 300.0, "tolerance_dn": 15.0, "min_lines": 8, "max_lines": 512}
        protocol = {"cfg": cfg, "model": fixture().to_dict(), "published": PARAMS}
        # Under 1-frame pipeline lag, quarantined execution converges cleanly in 1 update (0 reversals)
        ep_q = run_episode_quarantine_ablation(
            LaggyFakeCamera(lag_frames=1), "offset_ratio", 16, protocol, "q_always", quarantine_mode="always"
        )
        self.assertTrue(ep_q["success"])
        self.assertEqual(ep_q["direction_reversals"], 0)
        self.assertEqual(ep_q["stale_feedback_events"], 0)
        # Without quarantine under 1-frame pipeline lag, stale feedback triggers double-stepping overshoot
        ep_u = run_episode_quarantine_ablation(
            LaggyFakeCamera(lag_frames=1), "offset_ratio", 16, protocol, "q_none", quarantine_mode="none"
        )
        self.assertGreater(ep_u["stale_feedback_events"], 0)
        self.assertGreater(ep_u["updates"], ep_q["updates"])
        self.assertGreater(ep_u["direction_reversals"], 0)

    def test_frozen_module_hashes_untouched(self):
        plan_path = Path(r"P:\13_VLM_4_Camera_Control\T_review_fixes_20261016\T_independent_validation_plan.json")
        if plan_path.exists():
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            for path_str, expected_hash in plan["protocol"]["source_hashes"].items():
                actual_hash = hashlib.sha256(Path(path_str).read_bytes()).hexdigest()
                self.assertEqual(actual_hash, expected_hash, f"Hash mismatch for frozen file {path_str}")


if __name__ == "__main__":
    unittest.main()
