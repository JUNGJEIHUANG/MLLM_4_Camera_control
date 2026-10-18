"""Ephemeral mocked-camera integration tests. NOT real camera evidence."""
from argparse import Namespace
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch as mock_patch
import numpy as np

from iscas_control.experiments import run_trials
from iscas_control.types import Action
from test_control import model_fixture, patch


class SyntheticCamera:
    def __init__(self, output, actions):
        self.output = Path(output)
        self.output.mkdir(); (self.output/"log").mkdir()
        self.records = []
        self.current = actions[0]
        self.rng = np.random.default_rng(12)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def apply(self, action):
        self.current = action
        return {"mock_only": True}

    def capture(self, role):
        a = np.full((33, 33), 500.)
        q = patch(100 if self.current.lines >= 256 else 10)
        a[6:27, 6:27] = q
        a += self.rng.normal(0, 2, a.shape)
        f = {"image": a, "payload_bytes": 8912896, "header_bytes": 20, "role": role,
             "raw": "T_MOCK_ONLY_"+str(len(self.records)), "sha256": "MOCK_NOT_A_REAL_FILE"}
        self.records.append(f)
        return f

    def restore(self):
        pass


class PipelineTests(unittest.TestCase):
    def files(self, td, confirmed=True, kind="localization"):
        root = Path(td)
        model_fixture().save(root/"T_model.json")
        (root/"T_tasks.json").write_text(json.dumps({"units_confirmed": confirmed,
                    "tasks": [{"kind": kind, "tolerance": .3}]}), encoding="utf-8")
        return Namespace(model=root/"T_model.json", tasks=root/"T_tasks.json", output=root,
                         methods="full,no_cost", repetitions=1, execute=True)

    def test_unconfirmed_units_never_open_camera(self):
        with tempfile.TemporaryDirectory() as td:
            args = self.files(td, False)
            with mock_patch("iscas_control.hardware.OfficialCamera") as camera:
                with self.assertRaises(ValueError):
                    run_trials(args)
                camera.assert_not_called()

    def test_wrong_target_polarity_never_open_camera(self):
        with tempfile.TemporaryDirectory() as td:
            args = self.files(td, True, "relative_intensity")
            with mock_patch("iscas_control.hardware.OfficialCamera") as camera:
                with self.assertRaises(ValueError):
                    run_trials(args)
                camera.assert_not_called()

    def test_mocked_trial_export_and_scoring(self):
        with tempfile.TemporaryDirectory() as td:
            args = self.files(td)
            with mock_patch("iscas_control.hardware.OfficialCamera", SyntheticCamera):
                with contextlib.redirect_stdout(io.StringIO()):
                    run_trials(args)
            results = list(Path(td).glob("T_K1K2_trials_*/T_results.json"))
            self.assertEqual(len(results), 1)
            r = json.loads(results[0].read_text(encoding="utf-8"))
            self.assertEqual(r["status"], "TASK_TRIALS_CAPTURE_COMPLETE")
            self.assertEqual(len(r["episodes"]), 4)
            self.assertTrue(all(not e["control_reference_access"] for e in r["episodes"]))
            self.assertTrue(all(e["output"]["raw"].startswith("T_MOCK_ONLY") for e in r["episodes"]))
            # TemporaryDirectory deletes all fixture result files; none enter delivered evidence.


if __name__ == "__main__":
    unittest.main()
