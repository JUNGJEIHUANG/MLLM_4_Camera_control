import unittest
import numpy as np

from iscas_control.vlm_contract_compiler import (
    SEMANTIC_ROIS,
    SemanticContract,
    compute_roi_telemetry,
    parse_vlm_contract_json,
    parse_vlm_direct_lines,
    render_visual_representation,
    simulate_dual_layer_episode,
)


class TestVLMContractCompiler(unittest.TestCase):
    def test_parse_contract_json_and_bounds(self):
        raw = '{"roi_code": "JUNCTION_BOX", "target_dn": 450, "tolerance_dn": 22.5, "max_clip_fraction": 0.0, "confirm_frames": 6, "rationale": "Shadowed label"}'
        c = parse_vlm_contract_json(raw, version=2, observation_seq=4, task_id="T1")
        self.assertEqual(c.version, 2)
        self.assertEqual(c.roi_code, "JUNCTION_BOX")
        self.assertEqual(c.roi_box, SEMANTIC_ROIS["JUNCTION_BOX"])
        self.assertAlmostEqual(c.target_dn, 450.0)

    def test_parse_contract_clamps_below_offset(self):
        raw = '{"roi_code": "CENTER", "target_dn": 90}'
        c = parse_vlm_contract_json(raw, offset_floor_dn=166.5)
        self.assertGreaterEqual(c.target_dn, 186.5)

    def test_parse_direct_lines(self):
        self.assertEqual(parse_vlm_direct_lines('{"next_lines": 128}', 16), 128)
        self.assertEqual(parse_vlm_direct_lines('Step to 64 lines', 16), 64)

    def test_visual_representations(self):
        rng = np.random.default_rng(42)
        img = rng.integers(165, 600, size=(2048, 2176), dtype=np.uint16)
        img[0:200, 400:700] = 4095
        lin = render_visual_representation(img, mode="linear", downsample=8)
        mm = render_visual_representation(img, mode="minmax", downsample=8)
        dual = render_visual_representation(img, mode="dual_offset_grid", downsample=8)
        self.assertEqual(lin.shape, (256, 272, 3))
        self.assertEqual(mm.shape, (256, 272, 3))
        self.assertEqual(dual.shape, (256, 272 * 2 + 6, 3))

    def test_simulate_dual_layer_episode_physical_and_explicit_direct_replay(self):
        curve = {}
        for lines in (8, 16, 32, 64, 128, 256, 512):
            mean_dn = min(4095.0, 166.5 + 2.086 * lines)
            curve[lines] = {
                "mean_dn": mean_dn,
                "std_dn": 4.0,
                "p99_dn": min(4095.0, mean_dn + 15.0),
                "clip_fraction": 1.0 if mean_dn >= 4095.0 else 0.0,
                "net_signal_dn": max(0.0, mean_dn - 166.5),
                "snr": max(0.0, mean_dn - 166.5) / 4.0,
                "grad_snr": max(0.0, mean_dn - 166.5) / 6.0,
            }
        contract = SemanticContract(
            version=1,
            observation_seq=1,
            task_id="T_center",
            roi_code="CENTER",
            roi_box=SEMANTIC_ROIS["CENTER"],
            target_dn=300.0,
            tolerance_dn=15.0,
        )
        res_sec = simulate_dual_layer_episode(
            curve, contract, start_lines=8, fast_policy="secant_offset", quarantine_mode="always"
        )
        res_dir = simulate_dual_layer_episode(
            curve, contract, start_lines=8, fast_policy="direct_vlm", quarantine_mode="always",
            direct_vlm_step_sequence=[16, 32, 64],
        )
        self.assertTrue(res_sec["success"])
        self.assertEqual(res_sec["vlm_calls"], 1)
        self.assertTrue(res_dir["success"])
        self.assertEqual(res_dir["vlm_calls"], 3)
        # A synthetic replay tests mechanics, not empirical model superiority.


if __name__ == "__main__":
    unittest.main()
