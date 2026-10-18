"""Reproducible real-data fitting, fresh development acquisition and task trials.

Run with -m iscas_control.experiments. Hardware commands require --execute.
Fresh scoring references are never provided to Controller.choose().
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback
import numpy as np

from .types import Action, Task, TaskKind
from .measurement import PatchEstimator, temporal_variance
from .models import TaskModel, fit_real_summary
from .execution import Budget, CostModel, Executor, Rejected, PAYLOAD_BYTES
from .policies import Controller


ACTIONS = [Action(1.04, 80), Action(1.04, 308), Action(13.43, 80),
           Action(13.43, 308), Action(75.2, 80), Action(75.2, 160), Action(75.2, 256)]
DEFAULT_SEEDS = [(2001, 442), (2077, 457), (2002, 643), (2079, 643)]


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def source_receipt(run):
    log = Path(run)/"log"
    log.mkdir(exist_ok=True)
    hashes = {}
    for p in Path(__file__).parent.glob("*.py"):
        data = p.read_bytes()
        (log/("T_source_"+p.name)).write_bytes(data)
        hashes[str(p)] = hashlib.sha256(data).hexdigest()
    write_json(log/"T_runtime.json", {"python": sys.version, "numpy": np.__version__,
                                     "machine": platform.platform(), "processor": platform.processor(),
                                     "sources": hashes, "model_runs_on": "host CPU, synchronous Python"})


def unique_path(parent, prefix):
    return Path(parent)/("T_"+prefix+"_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))


def collect(args):
    ref_action = Action(args.reference_gain or 75.2, args.reference_lines or 256)
    plan = {"experiment": "K1_K2_CURRENT_SCENE_DEVELOPMENT", "settings_gain_lines":
            [[a.gain, a.lines] for a in ACTIONS], "seeds": args.seeds,
            "repeats": args.repeats, "polarity": args.polarity, "fit_calibration_validation": [3, 2, args.repeats-5],
            "reference_frames_per_side": 8, "reference_action": asdict(ref_action),
            "scene": "existing dark-hole targets" if args.polarity == -1 else "operator-provided bright-target ROIs",
            "photometry_scope": "relative local contrast, not original bright-spot validation" if args.polarity == -1
                                 else "relative bright-target intensity, not absolute/electron-rate validation",
            "aperture": "last operator report f/8; not electronically readable",
            "purpose": "real temporal-noise and task-error development, not full E0 certification"}
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2)); return
    from .hardware import OfficialCamera, BASELINE
    run = unique_path(args.output, "K1K2_dev")
    camera = OfficialCamera(run, ACTIONS+[ref_action])
    source_receipt(run)
    write_json(run/"T_plan.json", plan)
    result = {"status": "FAIL", "start_utc": datetime.now(timezone.utc).isoformat(),
              "plan": plan, "frames": [], "cells": []}
    estimator = PatchEstimator(args.seeds, polarity=args.polarity)
    def snap(label, k, action):
        f = camera.capture(label)
        result["frames"].append({k2: v for k2, v in f.items() if k2 != "image"} |
                                {"label": label, "frame_index": k, "gain_nominal": action.gain,
                                 "lines": action.lines})
        write_json(run/"T_progress.json", result)
        return f
    try:
        with camera:
            camera.apply(ref_action)
            refs = [snap("reference_before", k, ref_action) for k in range(9)]
            stacks = np.stack([estimator.patches(f["image"]) for f in refs[1:]])
            presence = estimator.measure(stacks.mean(axis=0), temporal_variance(stacks)/8)
            result["presence_gate"] = [p.to_dict() for p in presence]
            if not all(p.valid for p in presence):
                result["status"] = "TARGET_UNAVAILABLE_NO_CANDIDATE_SCAN"
                camera.restore()
            else:
                order = np.random.default_rng(20261015).permutation(len(ACTIONS)).tolist()
                result["order"] = order
                for sid in order:
                    action = ACTIONS[sid]
                    camera.apply(action)
                    indices = []
                    for k in range(args.repeats+1):
                        snap(f"setting_{sid}", k, action)
                        indices.append(len(result["frames"])-1)
                    result["cells"].append({"setting_id": sid, "frame_indices": indices,
                                            "gain_nominal": action.gain, "lines": action.lines})
                    print(f"DEV {len(result['cells'])}/{len(ACTIONS)} {action.key}", flush=True)
                camera.apply(ref_action)
                for k in range(9):
                    snap("reference_after", k, ref_action)
                camera.restore()
                result["status"] = "DEVELOPMENT_CAPTURE_COMPLETE"
    except Exception as exc:
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
    finally:
        result["end_utc"] = datetime.now(timezone.utc).isoformat()
        result["camera_receipt"] = str(run/"T_camera_receipt.json")
        result["all_camera_frames_including_restoration"] = len(camera.records)
        write_json(run/"T_summary.json", result)
    print(json.dumps({"run": str(run), "status": result["status"], "task_frames": len(result["frames"]),
                      "all_frames": len(camera.records), "error": result.get("error")}, ensure_ascii=False))
    if result["status"] != "DEVELOPMENT_CAPTURE_COMPLETE":
        raise RuntimeError(result.get("error", result["status"]))
    model, audit = fit_real_summary(run/"T_summary.json", args.seeds, polarity=args.polarity)
    model.save(run/"T_task_model.json")
    write_json(run/"T_model_validation.json", audit)
    costs = CostModel.from_camera_receipt(json.loads((run/"T_camera_receipt.json").read_text()))
    write_json(run/"T_action_cost_model.json", costs.to_dict())
    print("MODEL_AND_DISJOINT_VALIDATION_SAVED", flush=True)


def run_trials(args):
    model = TaskModel.load(args.model)
    cfg = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    cost_path = Path(args.model).parent/"T_action_cost_model.json"
    cost_profile = json.loads(cost_path.read_text()) if cost_path.is_file() else None
    if not cfg.get("units_confirmed"):
        raise ValueError("Task tolerance units must be explicitly confirmed before scored trials")
    tasks = [Task(TaskKind(x["kind"]), x["tolerance"], i+1) for i, x in enumerate(cfg["tasks"])]
    for task in tasks:
        model.predict(next(iter(model.actions.values())).action, task)  # reject wrong physical target BEFORE opening USB
    methods = args.methods.split(",")
    if any(m not in Controller.METHODS for m in methods):
        raise ValueError("Unknown method")
    plan = {"experiment": "K1K2_ONE_SCENE_PAIRED_DEVELOPMENT_TRIALS", "tasks": cfg,
            "methods": methods, "repetitions": args.repetitions,
            "frame_equivalent_budgets": [8, 14], "wall_budget_ms": 30000,
            "budget_data_scope": "RAW payload plus logical header plus online JSONL audit bytes; not USB wire measurement",
            "model_sha256": hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
            "cost_model_source": str(cost_path) if cost_profile else "explicit cold-start default; not yet measured",
            "output": "last evaluation frame; no retrospective best frame",
            "limits": ["single physical scene", "relative contrast photometry, not bright-point/electron-rate validation",
                       "no hardware sensor-frame parameter tags", "empirical proxy not error guarantee"]}
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2)); return
    from .hardware import OfficialCamera
    actions = [m.action for m in model.actions.values()]
    run = unique_path(args.output, "K1K2_trials")
    ref_action = Action(**model.provenance.get("reference_action", {"gain": 75.2, "lines": 256}))
    camera = OfficialCamera(run, actions+[ref_action])
    source_receipt(run)
    write_json(run/"T_plan.json", plan)
    (run/"T_model_snapshot.json").write_bytes(Path(args.model).read_bytes())
    result = {"status": "FAIL", "plan": plan, "episodes": [], "references": [],
              "start_utc": datetime.now(timezone.utc).isoformat()}
    refs = {}
    def reference(label):
        action = Action(**model.provenance.get("reference_action", {"gain": 75.2, "lines": 256}))
        camera.apply(action)
        fs = [camera.capture(label) for _ in range(9)]
        patches = np.stack([model.estimator.patches(f["image"]) for f in fs[1:]])
        qs = model.estimator.measure(patches.mean(axis=0), temporal_variance(patches)/8)
        if not all(p.valid for p in qs):
            camera.restore()
            raise Rejected("REFERENCE_TARGET_INVALID: stop all policies, not a method failure")
        rec = {"label": label, "raws": [f["raw"] for f in fs], "points": [p.to_dict() for p in qs]}
        result["references"].append(rec)
        refs[label] = qs
        return label
    try:
        with camera:
            before = reference("score_reference_start")
            block_id = 0
            for task in tasks:
                for frame_budget in (8, 14):
                    for rep in range(args.repetitions):
                        block_id += 1
                        order = methods[(block_id-1) % len(methods):]+methods[:(block_id-1) % len(methods)]
                        for method in order:
                            eid = f"b{block_id}_{task.kind.value}_{method}"
                            start_action = actions[0 if rep % 2 == 0 else min(2, len(actions)-1)]
                            camera.apply(start_action)
                            for _ in range(2):
                                camera.capture(eid+"_common_reset")
                            max_bytes = frame_budget*(PAYLOAD_BYTES+20+16384)+65536
                            budget = Budget(30000, max_bytes)
                            journal_path = run/("T_"+eid+"_events.jsonl")
                            with journal_path.open("wb") as log:
                                executor = Executor(camera, model, task, start_action, budget,
                                                    CostModel.from_dict(cost_profile) if cost_profile else CostModel(),
                                                    journal=lambda data: (log.write(data), log.flush()))
                                controller = Controller(method)
                                budget.begin()
                                stop = "UNKNOWN"
                                try:
                                    executor.observe(2)
                                    for _ in range(8):
                                        decision = controller.choose(executor)
                                        if decision.action is None:
                                            executor.execute(decision); stop = decision.reason; break
                                        executor.execute(decision)
                                    else:
                                        stop = "DECISION_LIMIT"
                                except Rejected as exc:
                                    stop = str(exc)
                                output = executor.commit()
                                elapsed = budget.elapsed_ms
                            # The controller receives no reference arrays or positions.
                            final = executor.measurements[-1] if executor.measurements else None
                            truth = refs[before]
                            task_error = None
                            detected = 0
                            if final:
                                pts = final["points"]
                                detected = sum(p.valid for p in pts)
                                if detected == len(pts):
                                    if task.kind == TaskKind.LOCALIZATION:
                                        task_error = float(np.sqrt(np.mean([(p.x-r.x)**2+(p.y-r.y)**2
                                                                          for p, r in zip(pts, truth)])))
                                    else:
                                        m = model.actions[executor.current.key]
                                        # Matched independently acquired anchor contrast, not stored training truth.
                                        ref_ms = Action(**model.provenance.get("reference_action", {"gain": 75.2, "lines": 256})).nominal_ms
                                        errors = [abs((p.flux/(executor.current.nominal_ms*m.rate_response[i])) /
                                                      (r.flux/ref_ms)-1) for i, (p, r) in enumerate(zip(pts, truth))]
                                        task_error = float(max(errors))
                            overrun = budget.overrun
                            success = task_error is not None and task_error <= task.tolerance and not any(overrun.values())
                            ep = {"episode_id": eid, "block_id": block_id, "task": task.kind.value,
                                  "tolerance": task.tolerance, "method": method, "order": order,
                                  "frame_equivalent_budget": frame_budget, "byte_budget": max_bytes,
                                  "elapsed_ms": elapsed, "data_bytes": budget.data_bytes,
                                  "frames": budget.frames, "overrun": overrun,
                                  "updates": sum(e["event"] == "COMMAND" for e in executor.events),
                                  "stop_reason": stop, "output": output, "final_action": executor.current.key,
                                  "detected_points": detected, "offline_error": task_error,
                                  "offline_success": bool(success), "scoring_reference": before,
                                  "event_log": str(journal_path), "online_error_upper": final["error_upper"] if final else None,
                                  "online_proxy_met": all(o["qualified"] for o in executor.measurements) if final else False,
                                  "control_reference_access": False,
                                  "frame_receipts": [{k: v for k, v in f.items() if k != "image"} for f in executor.frames]}
                            result["episodes"].append(ep)
                            write_json(run/("T_"+eid+"_episode.json"), ep)
                            write_json(run/"T_progress.json", result)
                            print(f"EP {eid} success={success} err={task_error} frames={budget.frames} ms={elapsed:.1f}", flush=True)
                        before = reference("score_reference_after_b"+str(block_id))
            camera.restore()
            result["status"] = "TASK_TRIALS_CAPTURE_COMPLETE"
    except Exception as exc:
        result["error"] = repr(exc); result["traceback"] = traceback.format_exc()
    finally:
        result["end_utc"] = datetime.now(timezone.utc).isoformat()
        result["all_camera_frames"] = len(camera.records)
        write_json(run/"T_results.json", result)
    print(json.dumps({"run": str(run), "status": result["status"], "episodes": len(result["episodes"]),
                      "all_frames": len(camera.records), "error": result.get("error")}, ensure_ascii=False))
    if result["status"] != "TASK_TRIALS_CAPTURE_COMPLETE":
        raise RuntimeError(result.get("error", result["status"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--seeds", type=json.loads, default=DEFAULT_SEEDS)
    c.add_argument("--repeats", type=int, default=8)
    c.add_argument("--polarity", type=int, choices=[-1, 1], default=-1)
    c.add_argument("--reference-gain", type=float, choices=[1.04, 13.43, 75.2])
    c.add_argument("--reference-lines", type=int)
    c.add_argument("--execute", action="store_true")
    t = sub.add_parser("trials")
    t.add_argument("--model", type=Path, required=True)
    t.add_argument("--tasks", type=Path, required=True)
    t.add_argument("--output", type=Path, required=True)
    t.add_argument("--methods", default=",".join(Controller.METHODS))
    t.add_argument("--repetitions", type=int, default=1)
    t.add_argument("--execute", action="store_true")
    f = sub.add_parser("fit")
    f.add_argument("--summary", type=Path, required=True)
    f.add_argument("--seeds", type=json.loads, default=DEFAULT_SEEDS)
    f.add_argument("--polarity", type=int, choices=[-1, 1], default=-1)
    f.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.cmd == "collect":
        if args.repeats < 8:
            p.error("Fresh collection needs >=8 repeats for disjoint fit/calibration/validation")
        if args.polarity == 1 and (args.reference_gain is None or args.reference_lines is None):
            p.error("Bright-target collection needs explicit non-saturating reference gain and exposure")
        collect(args)
    elif args.cmd == "trials":
        run_trials(args)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        model, audit = fit_real_summary(args.summary, args.seeds, polarity=args.polarity)
        model.save(args.output/"T_task_model.json")
        write_json(args.output/"T_model_validation.json", audit)


if __name__ == "__main__":
    main()
