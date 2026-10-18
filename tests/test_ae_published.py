"""Synthetic correctness tests, never empirical camera results."""
import inspect
import unittest
import numpy as np
from iscas_control.ae_brightness import BrightnessAE, ResponseModel
from iscas_control.ae_experiments import DEFAULT, run_episode as old_episode
from iscas_control.ae_published import PublishedAE
from iscas_control.ae_published_experiments import run_episode
from tests.test_ae_brightness import FakeCamera, fixture

PARAMS = {"recursive_k": 1., "su_tmax_lines": 512, "su_yfixed_dn": 4096}


class PublishedTests(unittest.TestCase):
    def policy(self, method, **params):
        return PublishedAE(method, DEFAULT, {**PARAMS, **params})
    def test_cvetkovic_equation6(self):
        p = self.policy("cvetkovic2010", recursive_k=.5)
        self.assertEqual(p.choose(100, {"mean_dn": 200., "clip_fraction": 0}), 150)
        self.assertEqual(self.policy("cvetkovic2010").choose(100, {"mean_dn": 200., "clip_fraction": 0}), 200)
    def test_su_equation5_matches_reciprocal_exposure_secant(self):
        for t, y in [(40, 231), (384, 780), (146, 600), (80, 280)]:
            p = self.policy("su2015")
            new = p.choose(t, {"mean_dn": y, "clip_fraction": 0})
            xn = 1/512+(1/t-1/512)*(4096-400)/(4096-y)
            self.assertAlmostEqual(p.last_decision["requested_lines"], 1/xn)
            self.assertEqual(new, round(1/xn))
    def test_su_endpoint_requires_fresh_nudged_exposure(self):
        p = self.policy("su2015")
        self.assertEqual(p.choose(512, {"mean_dn": 1000., "clip_fraction": 0}), 511)
        self.assertEqual(p.last_decision["branch"], "su2015_endpoint_nudge")
    def test_common_bounds_clipping_and_hold(self):
        for method in PublishedAE.METHODS:
            p = self.policy(method)
            self.assertEqual(p.choose(200, {"mean_dn": 4000., "clip_fraction": .1}), 100)
            self.assertEqual(p.choose(100, {"mean_dn": 400., "clip_fraction": 0}), 100)
            self.assertTrue(40 <= p.choose(40, {"mean_dn": 0., "clip_fraction": 0}) <= 240)
    def test_invalid_observation_and_parameters(self):
        for method in PublishedAE.METHODS:
            for y in (float('nan'), -1):
                with self.assertRaises(ValueError):
                    self.policy(method).choose(100, {"mean_dn": y, "clip_fraction": 0})
        with self.assertRaises(ValueError): self.policy("cvetkovic2010", recursive_k=1.1)
    def test_original_executor_preserved_except_factory_and_trace(self):
        source = inspect.getsource(old_episode).replace('policy = BrightnessAE(method, model, cfg)', 'policy = make_policy(method, model, protocol)')
        source = source.replace('new = policy.choose(current, obs)', "new = policy.choose(current, obs)\n        if getattr(policy, 'last_decision', None) is not None:\n            rec['policy_decision'] = policy.last_decision")
        self.assertEqual(inspect.getsource(run_episode).strip(), source.strip())
    def test_same_proposed_actions_and_cost_as_old_executor(self):
        protocol = {"cfg": DEFAULT, "model": fixture().to_dict(), "published": PARAMS}
        a = old_episode(FakeCamera(), "response_model", 40, protocol, "software_only")
        b = run_episode(FakeCamera(), "response_model", 40, protocol, "software_only")
        for key in ("online_frames", "updates", "transition_frames", "effective_frames", "final_lines", "success"):
            self.assertEqual(a[key], b[key])
    def test_recursive_success_and_su_failure_retained(self):
        protocol = {"cfg": DEFAULT, "model": fixture().to_dict(), "published": PARAMS}
        a = run_episode(FakeCamera(), "cvetkovic2010", 40, protocol, "synthetic_recursive")
        self.assertTrue(a["success"])
        b = run_episode(FakeCamera(), "su2015", 40, protocol, "synthetic_su")
        self.assertFalse(b["success"])
        self.assertEqual(b["stop_reason"], "UPDATE_LIMIT")
        self.assertEqual(b["online_frames"], 21)
