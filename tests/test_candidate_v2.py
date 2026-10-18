"""Synthetic mechanism tests, not real-camera advantage evidence."""
import unittest
import tempfile
from pathlib import Path
import numpy as np

from iscas_control.candidate_v2 import aggregate_task_error, select_quality_feasible, MetricAlignedTaskModel, QualityConstrainedController
from iscas_control.execution import Budget, CostModel, PAYLOAD_BYTES
from iscas_control.measurement import PatchEstimator
from iscas_control.models import ActionModel, TaskModel
from iscas_control.types import Action, Prediction, Task, TaskKind


class V2CandidateTests(unittest.TestCase):
    def setUp(self):
        self.a, self.b = Action(1.04, 308), Action(13.43, 308)
        self.task = Task(TaskKind.LOCALIZATION, .30)
        self.budget = Budget(30000, 100_000_000)
        self.cost = CostModel(200., 2.)

    def pred(self, action, risk, snr=10., supported=True, clipped=False):
        return Prediction(action, risk, risk, snr, supported, clipped, (risk/.3)**2)

    def test_localization_uses_same_rms_statistic_as_scoring(self):
        self.assertAlmostEqual(aggregate_task_error([.1, .3], TaskKind.LOCALIZATION), np.sqrt(.05))

    def test_photometry_keeps_maximum_point_error(self):
        self.assertEqual(aggregate_task_error([.1, .3], TaskKind.RELATIVE_CONTRAST), .3)

    def test_bad_errors_rejected(self):
        for e in ([], [-1], [np.nan], [[.1]]):
            with self.assertRaises(ValueError):
                aggregate_task_error(e, TaskKind.LOCALIZATION)
        with self.assertRaises(ValueError):
            aggregate_task_error([.1], "localization")

    def test_cost_and_quality_differ_only_when_both_actions_meet_same_task(self):
        p = [self.pred(self.a, .2), self.pred(self.b, .1)]
        self.assertEqual(select_quality_feasible(p, self.a, self.task, self.budget, self.cost, True), self.a)
        self.assertEqual(select_quality_feasible(p, self.a, self.task, self.budget, self.cost, False), self.b)

    def test_cost_cannot_choose_cheap_but_inadequate_measurement(self):
        p = [self.pred(self.a, .31), self.pred(self.b, .1)]
        for mode in (False, True):
            self.assertEqual(select_quality_feasible(p, self.a, self.task, self.budget, self.cost, mode), self.b)

    def test_infeasible_stop_is_not_faked_success(self):
        p = [self.pred(self.a, .31), self.pred(self.b, .5)]
        for mode in (False, True):
            self.assertIsNone(select_quality_feasible(p, self.a, self.task, self.budget, self.cost, mode))

    def test_domain_detection_saturation_gates_shared(self):
        for bad in (self.pred(self.b, .1, snr=4), self.pred(self.b, .1, supported=False),
                    self.pred(self.b, .1, clipped=True)):
            for mode in (False, True):
                self.assertIsNone(select_quality_feasible([bad], self.a, self.task, self.budget, self.cost, mode))

    def test_same_budget_and_real_transition_frame_cost(self):
        p = [self.pred(self.a, .2), self.pred(self.b, .1)]
        b = Budget(30000, 2*(PAYLOAD_BYTES+20+16384))
        for mode in (False, True):
            self.assertEqual(select_quality_feasible(p, self.a, self.task, b, self.cost, mode), self.a)
        self.assertEqual(self.cost.estimate(self.a, self.a)["frames"], 2)
        self.assertEqual(self.cost.estimate(self.a, self.b)["frames"], 3)


class V2ModelTests(unittest.TestCase):
    def make(self, cls=MetricAlignedTaskModel):
        estimator = PatchEstimator([(20, 20), (50, 20), (20, 50), (50, 50)])
        yy, xx = np.mgrid[-10:11, -10:11]
        p = 500. - 100*np.exp(-(xx**2+yy**2)/8)
        patches = np.repeat(p[None], 4, axis=0)
        var = np.full_like(patches, 4.)
        points = estimator.measure(patches, var)
        action = Action(1.04, 308)
        row = ActionModel(action, patches, var, np.array([p.flux for p in points]),
                          np.ones(4), np.array([.01, .1, .2, .3]), np.zeros(4))
        return cls(estimator, [row], np.ones(4)), action

    def test_model_observation_is_rms_but_v1_remains_max(self):
        v2, a = self.make()
        task = Task(TaskKind.LOCALIZATION, .3)
        obs = v2.observed(v2.actions[a.key].mean, a, task)
        errors = v2.errors(obs["points"], a, task.kind)
        self.assertAlmostEqual(obs["error_upper"], np.sqrt(np.mean(errors**2)))
        v1, _ = self.make(TaskModel)
        self.assertAlmostEqual(v1.observed(v1.actions[a.key].mean, a, task)["error_upper"], errors.max())

    def test_prior_candidate_metric_matches_observation_metric(self):
        m, a = self.make()
        task = Task(TaskKind.LOCALIZATION, .3)
        pred = m.predict(a, task)
        obs = m.observed(m.actions[a.key].mean, a, task)
        self.assertAlmostEqual(pred.upper_error, obs["error_upper"])

    def test_photometry_prior_preserves_v1_maximum(self):
        v2, a = self.make()
        v1, _ = self.make(TaskModel)
        task = Task(TaskKind.RELATIVE_CONTRAST, .15)
        self.assertAlmostEqual(v2.predict(a, task).upper_error, v1.predict(a, task).upper_error)

    def test_explicit_v2_loading_retains_v2_class_and_predictions(self):
        m, a = self.make()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"T_unvalidated_v2_model.json"
            m.save(path)
            other = MetricAlignedTaskModel.load(path)
            self.assertIsInstance(other, MetricAlignedTaskModel)
            t = Task(TaskKind.LOCALIZATION, .3)
            self.assertEqual(m.predict(a, t), other.predict(a, t))

    def test_controller_rejects_silent_v1_model_use(self):
        from types import SimpleNamespace
        m, _ = self.make(TaskModel)
        with self.assertRaises(ValueError):
            QualityConstrainedController().choose(SimpleNamespace(model=m))


if __name__ == "__main__":
    unittest.main()
