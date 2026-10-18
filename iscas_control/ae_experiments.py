"""Minimal fixed-gain brightness AE: development and frozen all-run comparisons."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import time
import traceback
import numpy as np

from .types import Action
from .ae_brightness import GRID, ROI, ResponseModel, BrightnessAE, measure, qualified, tune_ratio_offline
from .experiments import source_receipt, unique_path, write_json
from .execution import PAYLOAD_BYTES

DEFAULT = {"target_dn": 400., "tolerance_dn": 20., "max_roi_clip_fraction": 0.,
           "min_lines": 40, "max_lines": 512, "gain": 75.2, "max_step_factor": 6.,
           "ratio_alpha": 1., "stable_frames": 6, "first_accept_frames": 2,
           "max_online_frames": 28, "max_updates": 10, "wall_budget_ms": 15000,
           "roi": list(ROI)}


def stripped(frame):
    return {k: v for k, v in frame.items() if k != "image"}


def collect(args):
    plan = {"experiment": "AE_MAINLINE_DEVELOPMENT_NOT_E0", "cfg": DEFAULT,
            "grid_lines": list(GRID), "repeats_per_setting": 5, "fit_first": 3,
            "validate_last": 2, "target": "raw ROI brightness, not localization/intensity accuracy",
            "optics": "unchanged; last operator record f/8, not electronically confirmed"}
    if not args.execute:
        print(json.dumps(plan, indent=2)); return
    from .hardware import OfficialCamera
    run = unique_path(args.output, "AE_dev")
    camera = OfficialCamera(run, [Action(75.2, n) for n in GRID])
    source_receipt(run)
    write_json(run/"T_plan.json", plan)
    result = {"status": "FAIL", "plan": plan, "frames": [], "cells": [],
              "start_utc": datetime.now(timezone.utc).isoformat()}
    try:
        with camera:
            # Deterministic interleaving avoids a simple monotonic-time grid confound.
            for n in (192, 40, 384, 80, 512, 128, 256):
                camera.apply(Action(75.2, n))
                ids = []
                for k in range(6):
                    f = camera.capture(f"development_l{n}_"+("transition" if k == 0 else "evaluation"))
                    rec = {**stripped(f), "lines": n, "sample_index": k, "measurement": measure(f["image"])}
                    ids.append(len(result["frames"]))
                    result["frames"].append(rec)
                    write_json(run/"T_progress.json", result)
                result["cells"].append({"lines": n, "frame_indices": ids})
                print(f"DEV {n}: mean={np.mean([result['frames'][i]['measurement']['mean_dn'] for i in ids[1:]]):.2f}", flush=True)
            camera.restore()
            result["status"] = "AE_DEVELOPMENT_CAPTURE_COMPLETE"
    except Exception as exc:
        result["error"], result["traceback"] = repr(exc), traceback.format_exc()
    finally:
        result["all_frames"] = len(camera.records)
        result["end_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(run/"T_summary.json", result)
    print(json.dumps({"run": str(run), "status": result["status"], "all_frames": len(camera.records),
                      "error": result.get("error")}, indent=2))
    if result["status"] != "AE_DEVELOPMENT_CAPTURE_COMPLETE":
        raise RuntimeError(result.get("error", result["status"]))


def freeze(args):
    if (args.output/"T_frozen_protocol.json").exists():
        raise FileExistsError("Frozen protocol already exists; use a new version folder, do not overwrite after results")
    source = Path(args.development)
    j = json.loads((source/"T_summary.json").read_text())
    if j["status"] != "AE_DEVELOPMENT_CAPTURE_COMPLETE":
        raise ValueError("Complete fresh development required")
    for frame in j["frames"]:
        data = Path(frame["raw"]).read_bytes()
        if len(data) != PAYLOAD_BYTES or hashlib.sha256(data).hexdigest() != frame["sha256"]:
            raise ValueError("Development RAW hash/size mismatch")
        raw_stats = measure(np.frombuffer(data, dtype="<u2").reshape(2048, 2176))
        if not all(np.isclose(raw_stats[k], frame["measurement"][k], atol=1e-9) for k in raw_stats):
            raise ValueError("Development RAW/measurement mismatch")
    rows, validation = [], []
    for cell in j["cells"]:
        samples = [j["frames"][i] for i in cell["frame_indices"][1:]]
        if any(s["measurement"]["clip_fraction"] > 0 for s in samples):
            continue
        rows.append({"lines": cell["lines"], "mean_dn": float(np.mean([s["measurement"]["mean_dn"] for s in samples[:3]]))})
        validation.extend({"lines": cell["lines"], "raw": s["raw"], "mean_dn": s["measurement"]["mean_dn"]} for s in samples[3:])
    model = ResponseModel.fit(rows)
    cfg = dict(DEFAULT)
    if not min(model.mean_dn) < cfg["target_dn"]-cfg["tolerance_dn"] < cfg["target_dn"]+cfg["tolerance_dn"] < max(model.mean_dn):
        raise ValueError("Current brightness target not bracketed in frozen safe exposure domain; preserve scan and adjust physical condition, not E0")
    alpha, tuning = tune_ratio_offline(model, cfg)
    cfg["ratio_alpha"] = alpha
    protocol = {"version": "AE_MAINLINE_V1", "frozen_utc": datetime.now(timezone.utc).isoformat(),
                "development": str(source), "development_sha256": hashlib.sha256((source/"T_summary.json").read_bytes()).hexdigest(),
                "model": model.to_dict(), "cfg": cfg,
                "methods": list(BrightnessAE.METHODS), "repetitions": 4,
                "phases": {"static": "starts 40 and 512; relative under/overbrightness, not guaranteed clipping",
                           "brighter": "one operator aperture change; common pre-change steady exposure reconstructed for every policy",
                           "darker": "one reverse aperture change; common pre-change steady exposure reconstructed for every policy"},
                "recovery_timing": "first online post-change frame request to settled output, excludes operator movement/wait; not repeated in-flight physical steps",
                "selection_policy": "retain all formal runs, no best-of-repeats filtering",
                "fairness": "same ROI, gain, exposure domain, step limits, confirmation, backend and budgets; offset baseline receives same fitted offset",
                "claim_limit": "brightness AE scope shift, not validation of original K1 localization/intensity or independent K2 novelty"}
    for v in validation:
        v["prediction_dn"] = float(np.interp(v["lines"], model.lines, model.mean_dn))
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output/"T_frozen_protocol.json", protocol)
    write_json(args.output/"T_development_validation.json", {"fit": rows, "heldout_temporal_samples": validation, "not_independent_scenes": True})
    write_json(args.output/"T_baseline_tuning.json", tuning)
    print(json.dumps({"protocol": str(args.output/"T_frozen_protocol.json"), "offset_dn": model.offset_dn,
                      "ratio_alpha": alpha, "range_dn": [min(model.mean_dn), max(model.mean_dn)]}, indent=2))


def run_episode(camera, method, start_lines, protocol, eid):
    cfg = protocol["cfg"]
    model = ResponseModel(**protocol["model"])
    policy = BrightnessAE(method, model, cfg)
    camera.apply(Action(cfg["gain"], start_lines))
    reset = [stripped(camera.capture(eid+"_common_reset")) for _ in range(2)]
    current = start_lines
    start = time.perf_counter_ns()
    ep = {"episode_id": eid, "method": method, "start_lines": start_lines, "frames": [],
          "commands": [], "common_reset": reset, "start_tick_ns": start,
          "initial_condition": "relative to frozen DN target, not a claim of ADC clipping"}
    run, first_entry, stop = 0, None, "UNKNOWN"
    while True:
        elapsed = (time.perf_counter_ns()-start)/1e6
        if len(ep["frames"])+1 > cfg["max_online_frames"] or elapsed >= cfg["wall_budget_ms"]:
            stop = "BUDGET"; break
        f = camera.capture(eid+"_evaluation")
        obs = measure(f["image"], tuple(cfg["roi"]))
        rec = {**stripped(f), "measurement": obs, "qualified": qualified(obs, cfg), "lines": current,
               "role_online": "evaluation"}
        ep["frames"].append(rec)
        run = run+1 if rec["qualified"] else 0
        now_ms = (time.perf_counter_ns()-start)/1e6
        if run >= cfg["first_accept_frames"] and first_entry is None:
            first_entry = {"ms": now_ms, "frames": len(ep["frames"])}
        if run >= cfg["stable_frames"]:
            stop = "STABLE_TARGET_REACHED"; break
        if rec["qualified"]:
            continue  # all methods get the identical common deadband / hold rule
        if len(ep["commands"]) >= cfg["max_updates"]:
            stop = "UPDATE_LIMIT"; break
        new = policy.choose(current, obs)
        if new == current:
            stop = "ACTION_LIMIT_OUTSIDE_TARGET"; break
        # A changed action needs one isolated frame plus a useful evaluation.
        if len(ep["frames"])+2 > cfg["max_online_frames"]:
            stop = "FRAME_BUDGET_BEFORE_CHANGE"; break
        command = camera.apply(Action(cfg["gain"], new))
        ep["commands"].append(command)
        current = new
        q = camera.capture(eid+"_transition")
        ep["frames"].append({**stripped(q), "measurement": measure(q["image"], tuple(cfg["roi"])),
                             "role_online": "transition", "lines": current, "qualified": False})
        run = 0
    end = time.perf_counter_ns()
    evaluations = [f for f in ep["frames"] if f["role_online"] == "evaluation"]
    final = evaluations[-1] if evaluations else None
    elapsed = (end-start)/1e6
    data_bytes = sum(f["payload_bytes"]+f["header_bytes"] for f in ep["frames"])
    max_bytes = cfg["max_online_frames"]*(PAYLOAD_BYTES+20)
    overrun = {"ms": max(0., elapsed-cfg["wall_budget_ms"]), "bytes": max(0, data_bytes-max_bytes)}
    good = [f for f in evaluations if f["qualified"]]
    changes = [int(r["new"].split('_l')[1])-int(r["old"].split('_l')[1]) for r in ep["commands"]]
    reversals = sum(a*b < 0 for a, b in zip(changes[:-1], changes[1:]))
    tail = evaluations[-cfg["stable_frames"]:]
    ep.update({"end_tick_ns": end, "elapsed_ms": elapsed, "stop_reason": stop,
               "success": stop == "STABLE_TARGET_REACHED" and not any(overrun.values()),
               "first_entry": first_entry, "online_frames": len(ep["frames"]), "data_bytes": data_bytes,
               "budget_overrun": overrun, "updates": len(ep["commands"]), "direction_reversals": reversals,
               "transition_frames": sum(f["role_online"] == "transition" for f in ep["frames"]),
               "effective_frames": len(good), "frame_cost_per_effective": len(ep["frames"])/len(good) if good else None,
               "final_lines": current, "output": final["raw"] if final else None,
               "final_mean_dn": final["measurement"]["mean_dn"] if final else None,
               "tail_std_dn": float(np.std([f["measurement"]["mean_dn"] for f in tail], ddof=1)) if len(tail)>1 else None,
               "tail_mae_dn": float(np.mean([abs(f["measurement"]["mean_dn"]-cfg["target_dn"]) for f in tail])) if tail else None})
    return ep


def trials(args):
    protocol_path = Path(args.protocol)
    protocol = json.loads(protocol_path.read_text())
    starts = [int(x) for x in args.starts.split(",")]
    if not starts or any(x < protocol["cfg"]["min_lines"] or x > protocol["cfg"]["max_lines"] for x in starts):
        raise ValueError("Initial exposure outside frozen envelope")
    plan = {"phase": args.phase, "condition_label": args.condition_label, "starts": starts,
            "repetitions": args.repetitions, "protocol": protocol,
            "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            "expected_episodes": len(starts)*args.repetitions*len(protocol["methods"])}
    if not args.execute:
        print(json.dumps(plan, indent=2)); return
    from .hardware import OfficialCamera
    run = unique_path(args.output, "AE_"+args.phase)
    camera = OfficialCamera(run, [Action(protocol["cfg"]["gain"], n) for n in range(40, 513)])
    source_receipt(run)
    write_json(run/"T_plan.json", plan)
    result = {"status": "FAIL", "plan": plan, "episodes": [], "start_utc": datetime.now(timezone.utc).isoformat()}
    try:
        with camera:
            block = 0
            for rep in range(args.repetitions):
                for start in starts:
                    methods = protocol["methods"]
                    order = methods[block % len(methods):]+methods[:block % len(methods)]
                    block += 1
                    for method in order:
                        eid = f"b{block}_{method}_l{start}"
                        ep = run_episode(camera, method, start, protocol, eid)
                        ep.update({"block_id": block, "rep": rep, "phase": args.phase, "order": order})
                        result["episodes"].append(ep)
                        write_json(run/("T_"+eid+"_episode.json"), ep)
                        write_json(run/"T_progress.json", result)
                        print(f"EP {eid} success={ep['success']} frames={ep['online_frames']} updates={ep['updates']} mean={ep['final_mean_dn']:.2f} ms={ep['elapsed_ms']:.1f}", flush=True)
            camera.restore()
            result["status"] = "AE_TRIALS_CAPTURE_COMPLETE"
    except Exception as exc:
        result["error"], result["traceback"] = repr(exc), traceback.format_exc()
    finally:
        result["all_frames"] = len(camera.records)
        result["end_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(run/"T_results.json", result)
    print(json.dumps({"run": str(run), "status": result["status"], "episodes": len(result["episodes"]),
                      "all_frames": len(camera.records), "error": result.get("error")}, indent=2))
    if result["status"] != "AE_TRIALS_CAPTURE_COMPLETE":
        raise RuntimeError(result.get("error", result["status"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest="cmd", required=True)
    c = s.add_parser("collect")
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--execute", action="store_true")
    f = s.add_parser("freeze")
    f.add_argument("--development", type=Path, required=True)
    f.add_argument("--output", type=Path, required=True)
    t = s.add_parser("trials")
    t.add_argument("--protocol", type=Path, required=True)
    t.add_argument("--output", type=Path, required=True)
    t.add_argument("--phase", choices=["development_check", "static", "brighter", "darker"], required=True)
    t.add_argument("--condition-label", default="physical aperture last reported f/8; not electronic readback")
    t.add_argument("--starts", default="40,512")
    t.add_argument("--repetitions", type=int, default=4)
    t.add_argument("--execute", action="store_true")
    a = p.parse_args()
    if a.cmd == "collect": collect(a)
    elif a.cmd == "freeze": freeze(a)
    else: trials(a)


if __name__ == "__main__": main()
