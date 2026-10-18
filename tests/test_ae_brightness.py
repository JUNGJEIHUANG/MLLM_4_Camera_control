"""Synthetic software tests only; fixtures are not real AE experiment results."""
import unittest
import numpy as np
from iscas_control.ae_brightness import ResponseModel, BrightnessAE, qualified, monotone, measure, tune_ratio_offline
from iscas_control.ae_experiments import DEFAULT, run_episode
from iscas_control.types import Action


def fixture():
    return ResponseModel.fit([{"lines": t, "mean_dn": 160+t} for t in (40, 80, 128, 256, 384, 512)])


class FakeCamera:
    def __init__(self):
        self.current, self.seq = Action(75.2, 40), 0
    def apply(self, action):
        old = self.current; self.current = action
        return {"old": old.key, "new": action.key}
    def capture(self, role):
        self.seq += 1
        return {"image": np.full((2048, 2176), 160+self.current.lines, dtype=np.uint16),
                "raw": f"T_synthetic_{self.seq}", "sha256": "test_only", "role": role,
                "payload_bytes": 8912896, "header_bytes": 20}


class AEBrightnessTests(unittest.TestCase):
    def test_pava(self):
        np.testing.assert_allclose(monotone([2, 1, 3]), [1.5, 1.5, 3])
    def test_offset_fit(self):
        self.assertAlmostEqual(fixture().offset_dn, 160)
    def test_common_quality(self):
        self.assertTrue(qualified({"mean_dn": 380, "clip_fraction": 0}, DEFAULT))
        self.assertFalse(qualified({"mean_dn": 400, "clip_fraction": .00001}, DEFAULT))
    def test_both_model_baselines_get_same_development_offset(self):
        model = fixture()
        for method in ("offset_ratio", "response_model"):
            self.assertEqual(BrightnessAE(method, model, DEFAULT).choose(40, {"mean_dn": 200., "clip_fraction": 0}), 240)
    def test_saturation_safety_shared(self):
        for method in BrightnessAE.METHODS:
            self.assertEqual(BrightnessAE(method, fixture(), DEFAULT).choose(200, {"mean_dn": 4000., "clip_fraction": .2}), 100)
    def test_bounds(self):
        for method in BrightnessAE.METHODS:
            n = BrightnessAE(method, fixture(), DEFAULT).choose(512, {"mean_dn": 100., "clip_fraction": 0})
            self.assertEqual(n, 512)
    def test_hold_inside_common_deadband(self):
        for method in BrightnessAE.METHODS:
            self.assertEqual(BrightnessAE(method, fixture(), DEFAULT).choose(200, {"mean_dn": 400., "clip_fraction": 0}), 200)
    def test_replay_is_not_measured_tuning(self):
        alpha, audit = tune_ratio_offline(fixture(), DEFAULT)
        self.assertIn(alpha, [.5, .7, 1., 1.5, 2.])
        self.assertEqual(len(audit["candidates"]), 5)
    def test_bad_mean(self):
        with self.assertRaises(ValueError):
            BrightnessAE("ratio", fixture(), DEFAULT).choose(40, {"mean_dn": float('nan'), "clip_fraction": 0})
    def test_episode_common_confirmation_and_transition_cost(self):
        protocol = {"cfg": DEFAULT, "model": fixture().to_dict()}
        ep = run_episode(FakeCamera(), "offset_ratio", 40, protocol, "software_only")
        self.assertTrue(ep["success"])
        self.assertEqual(ep["updates"], 1)
        self.assertEqual(ep["transition_frames"], 1)
        self.assertEqual(ep["online_frames"], 8)  # initial + transition + six confirmations
        self.assertEqual(ep["first_entry"]["frames"], 4)
        self.assertEqual(ep["effective_frames"], 6)
        self.assertEqual(ep["output"], ep["frames"][-1]["raw"])
    def test_episode_unreachable_preserves_failure(self):
        protocol = {"cfg": {**DEFAULT, "target_dn": 1000.}, "model": fixture().to_dict()}
        ep = run_episode(FakeCamera(), "offset_ratio", 512, protocol, "software_only_failure")
        self.assertFalse(ep["success"])
        self.assertEqual(ep["stop_reason"], "ACTION_LIMIT_OUTSIDE_TARGET")
        self.assertEqual(ep["online_frames"], 1)
