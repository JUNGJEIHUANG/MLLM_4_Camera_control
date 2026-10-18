"""Live-adapter HOST protection preflight, not scientific task-success testing.

The 0.3/.05 values below are illustrative software metadata only. No scientific
success is scored against them and no tolerance-unit confirmation is inferred.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

from .execution import Budget, CostModel, Executor, Rejected
from .experiments import unique_path, source_receipt, write_json
from .hardware import OfficialCamera, BASELINE
from .models import TaskModel
from .types import Action, Task, TaskKind, Decision


def run(model_path, output_root):
    model = TaskModel.load(model_path)
    run = unique_path(output_root, "K2_live_guards")
    camera = OfficialCamera(run, [m.action for m in model.actions.values()])
    source_receipt(run)
    result = {"status": "FAIL", "scope": "host-control preflight on real adapter; not full E5/E4 or task accuracy evidence",
              "scientific_tolerance_values_used_for_scoring": False, "cases": []}
    try:
        with camera:
            budget = Budget(30000, 200_000_000); budget.begin()
            ex = Executor(camera, model, Task(TaskKind.LOCALIZATION, .3), BASELINE, budget, CostModel())
            def rejected(name, decision, executor=ex):
                before = (len(camera.commands), len(camera.records))
                try:
                    executor.execute(decision)
                except Rejected as exc:
                    after = (len(camera.commands), len(camera.records))
                    if before != after:
                        raise AssertionError("Rejected request caused device action")
                    result["cases"].append({"name": name, "result": "REJECTED_BEFORE_DEVICE_IO",
                                             "exception": str(exc), "command_delta": 0, "frame_delta": 0})
                else:
                    raise AssertionError("Invalid request was accepted")
            rejected("stale_task", Decision(BASELINE, 99, 0, "injected"))
            rejected("stale_observation", Decision(BASELINE, 1, 99, "injected"))
            rejected("outside_frozen_model", Decision(Action(75.2, 512), 1, 0, "injected"))
            tiny = Budget(30000, 1); tiny.begin()
            ex_tiny = Executor(camera, model, ex.task, BASELINE, tiny)
            rejected("exhausted_data_budget", Decision(Action(75.2, 256), 1, 0, "injected"), ex_tiny)
            ex.change_task(Task(TaskKind.RELATIVE_CONTRAST, .05, 2))
            rejected("old_request_after_task_switch", Decision(BASELINE, 1, 0, "injected"))
            count = len(camera.records)
            ex.execute(Decision(BASELINE, 2, 0, "normal_hold_control"))
            if len(camera.records)-count != 2:
                raise AssertionError("Normal hold did not acquire two frames")
            result["cases"].append({"name": "valid_current_hold", "result": "TWO_REAL_FRAMES_CAPTURED",
                                     "raws": [f["raw"] for f in ex.frames]})
            rejected("stale_after_new_frames", Decision(BASELINE, 2, 0, "injected"))
            camera.restore()
            result["status"] = "HOST_GUARD_PREFLIGHT_PASS"
    except Exception as exc:
        result["error"] = repr(exc)
    result["all_real_frames"] = len(camera.records)
    result["camera_receipt"] = str(run/"T_camera_receipt.json")
    write_json(run/"T_guard_results.json", result)
    print(json.dumps({"run": str(run), "status": result["status"], "cases": len(result["cases"]),
                      "all_real_frames": len(camera.records), "error": result.get("error")}))
    if result["status"] != "HOST_GUARD_PREFLIGHT_PASS":
        raise RuntimeError(result.get("error", result["status"]))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    if a.execute:
        run(a.model, a.output)
    else:
        print("DRY RUN: six rejected-request paths and one normal hold; 5 images including restoration; no task scoring")
