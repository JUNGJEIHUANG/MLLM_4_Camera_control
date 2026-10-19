import unittest
import numpy as np

from iscas_control.vlm_contract_compiler import (
    SEMANTIC_ROIS,
    SemanticContract,
    coco_xywh_to_xyxy,
    compute_bbox_iou_xyxy,
    compute_roi_telemetry,
    convert_pred_bbox_to_pixel_xyxy,
    parse_vlm_contract_json,
    parse_vlm_direct_lines,
    parse_vlm_free_bbox,
    render_visual_representation,
    simulate_dual_layer_episode,
    summarize_rec_evaluation,
    xyxy_pixel_to_sensor_roi_box,
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

    def test_free_bbox_conversion_and_strict_rec_scoring(self):
        # 1. Ref-L4 / COCO [x, y, w, h] -> [x1, y1, x2, y2]
        gt_xyxy = coco_xywh_to_xyxy([580.6, 179.5, 93.6, 112.1])
        self.assertAlmostEqual(gt_xyxy[2], 674.2)
        self.assertAlmostEqual(gt_xyxy[3], 291.6)

        # 2. Qwen2.5-VL resized pixel coordinate conversion -> original pixel
        pred = parse_vlm_free_bbox(
            '```json\n{"bbox_2d": [290.3, 89.75, 337.1, 145.8], "label": "antique camera"}\n```',
            orig_width=1024,
            orig_height=741,
            resized_width=512,
            resized_height=370.5,
            default_coord_space="qwen25_resized_pixel",
        )
        self.assertTrue(pred["valid"])
        iou = compute_bbox_iou_xyxy(pred["bbox_xyxy"], gt_xyxy)
        self.assertGreater(iou, 0.99)

        # 3. Fixed alias output WITHOUT a 4D bbox must fail strictly (never repaired to CENTER)
        bad = parse_vlm_free_bbox(
            '{"roi_code": "DARK_WALL", "grid_cell": "R3C0", "target_dn": 260.0}',
            orig_width=1024,
            orig_height=741,
        )
        self.assertFalse(bad["valid"])
        self.assertEqual(compute_bbox_iou_xyxy(bad["bbox_xyxy"], gt_xyxy), 0.0)

        # 4. Sensor ROI mapping and REC summary denominator check
        roi_box = xyxy_pixel_to_sensor_roi_box(pred["bbox_xyxy"], 1024, 741)
        self.assertEqual(len(roi_box), 4)
        summary = summarize_rec_evaluation([
            {"valid": True, "iou": iou},
            {"valid": False, "iou": 0.0},
        ])
        self.assertEqual(summary["num_samples"], 2)
        self.assertEqual(summary["valid_bbox_count"], 1)
        self.assertAlmostEqual(summary["acc_at_iou_050"], 0.5)


if __name__ == "__main__":
    unittest.main()

