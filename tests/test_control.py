"""Software tests use synthetic arrays; none are camera experiment results."""
import tempfile
from pathlib import Path
import unittest
import numpy as np

from iscas_control.measurement import PatchEstimator, temporal_variance
from iscas_control.models import TaskModel, ActionModel
from iscas_control.types import Action, Task, TaskKind, Decision
from iscas_control.execution import Budget, CostModel, Executor, Phase, Rejected, PAYLOAD_BYTES
from iscas_control.policies import Controller


def patch(signal=100., dx=0., dy=0., background=500.):
    yy, xx = np.mgrid[-10:11, -10:11]
    return background + .2*xx - .1*yy - signal*np.exp(-((xx-dx)**2+(yy-dy)**2)/8)


def model_fixture():
    e = PatchEstimator([(16, 16)])
    rows = []
    for a, strength in [(Action(1.04, 80), 10.), (Action(1.04, 308), 100.)]:
        q = patch(strength)[None]
        v = np.full_like(q, 4.)
        p = e.measure(q, v)[0]
        rows.append(ActionModel(a, q, v, np.array([p.flux]),
                                np.array([p.flux/a.nominal_ms/100]), np.array([.02]), np.array([.001])))
    return TaskModel(e, rows, [100.])


class FakeBackend:
    def __init__(self, image=None, fail=False):
        self.writes = self.captures = 0
        self.image = np.full((33, 33), 500.) if image is None else image
        self.fail = fail

    def apply(self, action):
        self.writes += 1
        if self.fail:
            raise RuntimeError("Injected command timeout; not real hardware")
        return {"gain_readback": True}

    def capture(self, role):
        self.captures += 1
        return {"image": self.image, "payload_bytes": PAYLOAD_BYTES,
                "header_bytes": 20, "role": role, "raw": f"T_fake_{self.captures}", "sha256": "test-only"}


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        self.e = PatchEstimator([(20, 20)])
        self.v = np.full((1, 21, 21), 4.)

    def test_dark_centroid_translation(self):
        p = self.e.measure(patch(dx=.6, dy=-.4)[None], self.v)[0]
        self.assertTrue(p.valid)
        self.assertAlmostEqual(p.x, 20.6, delta=.08)
        self.assertAlmostEqual(p.y, 19.6, delta=.08)

    def test_plane_background_offset(self):
        a = self.e.measure(patch()[None], self.v)[0]
        b = self.e.measure(patch(background=800)[None], self.v)[0]
        self.assertAlmostEqual(a.flux, b.flux, places=8)
        self.assertAlmostEqual(a.x, b.x, places=8)

    def test_noise_changes_uncertainty(self):
        a = self.e.measure(patch()[None], self.v)[0]
        b = self.e.measure(patch()[None], self.v*9)[0]
        self.assertAlmostEqual(b.flux_sigma/a.flux_sigma, 3., places=8)
        self.assertGreater(b.centroid_sigma, a.centroid_sigma)

    def test_spatial_texture_not_temporal_variance(self):
        a = patch()
        yy, xx = np.mgrid[-10:11, -10:11]
        b = a + 5*np.cos(xx*2)*self.e.ring
        p = self.e.measure(a[None], self.v)[0]
        q = self.e.measure(b[None], self.v)[0]
        self.assertAlmostEqual(p.flux_sigma, q.flux_sigma)
        self.assertGreater(q.spatial_ring_mad, p.spatial_ring_mad)

    def test_signed_photometry_not_rectified(self):
        p = self.e.measure(np.full((1, 21, 21), 500.), self.v)[0]
        self.assertAlmostEqual(p.flux, 0., delta=1e-8)
        self.assertFalse(p.valid)

    def test_bright_polarity(self):
        e = PatchEstimator([(20, 20)], polarity=1)
        p = e.measure((1000-patch())[None], self.v)[0]
        self.assertTrue(p.valid)
        self.assertGreater(p.flux, 0)

    def test_saturation_rejected(self):
        q = patch(); q[0, 0] = 4095
        self.assertFalse(self.e.measure(q[None], self.v)[0].valid)

    def test_bad_noise_input(self):
        with self.assertRaises(ValueError):
            self.e.measure(patch()[None], np.zeros_like(self.v))

    def test_temporal_variance_floor(self):
        v = temporal_variance(np.ones((3, 1, 21, 21)))
        self.assertTrue(np.all(v > 0))
        with self.assertRaises(ValueError):
            temporal_variance(np.ones((2, 1, 21, 21)))

    def test_roi_bounds(self):
        with self.assertRaises(ValueError):
            self.e.patches(np.ones((22, 22)))


class ModelTests(unittest.TestCase):
    def test_development_alignment_known_shift(self):
        from iscas_control.alignment import align_development_seeds
        yy, xx = np.mgrid[:80, :80]
        image = 500-100*np.exp(-((xx-46)**2+(yy-38)**2)/8)
        aligned, _ = align_development_seeds(image, [(40, 40)])
        self.assertEqual(aligned, [(46, 38)])

    def test_development_alignment_blank_rejected(self):
        from iscas_control.alignment import align_development_seeds
        with self.assertRaises(ValueError):
            align_development_seeds(np.full((80, 80), 500.), [(40, 40)])

    def test_round_trip(self):
        model = model_fixture()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/"T_model.json"; model.save(p); other = TaskModel.load(p)
            a = Action(1.04, 308); t = Task(TaskKind.LOCALIZATION, .3)
            self.assertEqual(model.predict(a, t), other.predict(a, t))

    def test_tasks_are_distinct(self):
        m = model_fixture(); a = Action(1.04, 308)
        x = m.predict(a, Task(TaskKind.LOCALIZATION, .3))
        y = m.predict(a, Task(TaskKind.RELATIVE_CONTRAST, .05))
        self.assertNotEqual(x.loss, y.loss)

    def test_dark_contrast_not_called_bright_intensity(self):
        m = model_fixture()
        with self.assertRaises(ValueError):
            m.predict(Action(1.04, 308), Task(TaskKind.RELATIVE_INTENSITY, .05))

    def test_bright_intensity_model_accepted(self):
        m = model_fixture()
        m.estimator = PatchEstimator([(16, 16)], polarity=1)
        for row in m.actions.values():
            row.mean = 1000-row.mean
        self.assertTrue(np.isfinite(m.predict(Action(1.04, 308), Task(TaskKind.RELATIVE_INTENSITY, .05)).loss))

    def test_unknown_action(self):
        with self.assertRaises(KeyError):
            model_fixture().predict(Action(75.2, 256), Task(TaskKind.LOCALIZATION, .3))

    def test_action_bounds(self):
        for gain, lines in [(1.04, 10000), (99, 80), (1.04, 80.5)]:
            with self.assertRaises(ValueError):
                Action(gain, lines)

    def test_unit_type_required(self):
        with self.assertRaises(ValueError):
            Task("localization", .3)


class ExecutionTests(unittest.TestCase):
    def make(self, backend=None, budget_bytes=200_000_000):
        m = model_fixture(); b = Budget(30000, budget_bytes); b.begin()
        return Executor(backend or FakeBackend(), m, Task(TaskKind.LOCALIZATION, .3),
                        Action(1.04, 80), b, CostModel(frame_ms=1, command_ms=1))

    def test_stale_task_no_io(self):
        e = self.make()
        with self.assertRaises(Rejected):
            e.execute(Decision(Action(1.04, 308), 2, 0, "test"))
        self.assertEqual(e.backend.writes+e.backend.captures, 0)

    def test_stale_observation_no_io(self):
        e = self.make()
        with self.assertRaises(Rejected):
            e.execute(Decision(Action(1.04, 308), 1, 99, "test"))
        self.assertEqual(e.backend.writes+e.backend.captures, 0)

    def test_budget_reject_no_io(self):
        e = self.make(budget_bytes=1)
        with self.assertRaises(Rejected):
            e.execute(Decision(Action(1.04, 308), 1, 0, "test"))
        self.assertEqual(e.backend.writes+e.backend.captures, 0)

    def test_domain_reject_no_io(self):
        e = self.make()
        with self.assertRaises(Rejected):
            e.execute(Decision(Action(75.2, 256), 1, 0, "test"))
        self.assertEqual(e.backend.writes+e.backend.captures, 0)

    def test_changed_action_quarantines_and_counts(self):
        e = self.make()
        e.execute(Decision(Action(1.04, 308), 1, 0, "test"))
        self.assertEqual(e.backend.captures, 3)
        self.assertEqual(e.frames[0]["role"], "transition_quarantine")
        self.assertEqual(e.budget.frames, 3)
        self.assertGreater(e.budget.data_bytes, 3*PAYLOAD_BYTES)

    def test_hold_does_not_write(self):
        e = self.make()
        e.execute(Decision(e.current, 1, 0, "hold"))
        self.assertEqual(e.backend.writes, 0)
        self.assertEqual(e.backend.captures, 2)

    def test_fault_prevents_further_capture(self):
        e = self.make(FakeBackend(fail=True))
        with self.assertRaises(RuntimeError):
            e.execute(Decision(Action(1.04, 308), 1, 0, "test"))
        self.assertEqual(e.phase, Phase.FAULT)
        self.assertEqual(e.backend.captures, 0)
        with self.assertRaises(Rejected):
            e.observe()

    def test_commit_last_not_best(self):
        e = self.make(); e.observe(); committed = e.commit()
        self.assertEqual(committed["raw"], "T_fake_2")

    def test_task_update_invalidates_old_decision(self):
        e = self.make(); d = Decision(e.current, 1, 0, "old")
        e.change_task(Task(TaskKind.RELATIVE_CONTRAST, .05, 2))
        with self.assertRaises(Rejected):
            e.execute(d)

    def test_cost_categories(self):
        c = CostModel(1, 1); a = Action(1.04, 80)
        self.assertEqual(c.estimate(a, a)["frames"], 2)
        self.assertEqual(c.estimate(a, Action(1.04, 308))["category"], "exposure")
        self.assertEqual(c.estimate(a, Action(75.2, 160))["category"], "gain")

    def test_measured_cost_profile_round_trip(self):
        c = CostModel(2, 3)
        c.observe_command(Action(1.04, 80), Action(75.2, 256), 9.)
        other = CostModel.from_dict(c.to_dict())
        self.assertEqual(c.estimate(Action(1.04, 80), Action(75.2, 256)),
                         other.estimate(Action(1.04, 80), Action(75.2, 256)))

    def test_apply_fault_invalidates_previous_output(self):
        e = self.make(); e.observe(); e.backend.fail = True
        with self.assertRaises(RuntimeError):
            e.execute(Decision(Action(1.04, 308), 1, e.observation_seq, "test"))
        self.assertIsNone(e.commit())

    def test_partial_transfer_charged(self):
        from iscas_control.hardware import CaptureError
        class Partial(FakeBackend):
            def capture(self, role):
                raise CaptureError("software-injected partial", 1234)
        e = self.make(Partial())
        with self.assertRaises(CaptureError):
            e.observe()
        self.assertGreaterEqual(e.budget.data_bytes, 1234)

    def test_no_cost_ablation_uses_same_predictions(self):
        e = self.make(); e.observe()
        a = Controller("full").choose(e)
        b = Controller("no_cost").choose(e)
        self.assertEqual(a.predictions, b.predictions)


if __name__ == "__main__":
    unittest.main()
