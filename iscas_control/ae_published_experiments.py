"""New paired published-baseline cohort; old AE sources and results are unchanged."""
from datetime import datetime, timezone
from pathlib import Path
import argparse, hashlib, json, time, traceback
import numpy as np
from .types import Action
from .ae_brightness import ResponseModel, measure, qualified
from .ae_published import make_policy
from .ae_experiments import stripped
from .experiments import unique_path, source_receipt, write_json
from .execution import PAYLOAD_BYTES

def run_episode(camera, method, start_lines, protocol, eid):
    cfg = protocol["cfg"]
    model = ResponseModel(**protocol["model"])
    policy = make_policy(method, model, protocol)
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
        if getattr(policy, 'last_decision', None) is not None:
            rec['policy_decision'] = policy.last_decision
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


def freeze(args):
    original = Path(args.original)
    protocol = json.loads(original.read_text(encoding="utf-8"))
    protocol.update({
        "version": "AE_PUBLISHED_COMPARISON_V1",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "parent_protocol": str(original.resolve()),
        "parent_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "methods": ["cvetkovic2010", "su2015", "response_model"],
        "published": {"recursive_k": 1.0, "su_tmax_lines": 512, "su_yfixed_dn": 4096,
                      "recursive_parameter_choice": "fastest k in published allowed [0,1], not typical smoothing k<0.2",
                      "su_endpoint": "command Tmax-1 and acquire fresh brightness before Eq5 if current == Tmax",
                      "su_source": "https://mcl.usc.edu/wp-content/uploads/2016/01/13.pdf",
                      "su_doi": "10.1109/ICCE.2015.7066300", "su_equation": 5,
                      "recursive_source": "https://link.springer.com/article/10.1155/2010/197194",
                      "recursive_doi": "10.1155/2010/197194", "recursive_equation": 6,
                      "scope": "independent RAW-mean update-law adaptations; not author code or full original systems"},
        "phases": {"static": {"starts": [40, 384], "aperture": "f/8 operator confirmed"},
                   "brighter": {"starts": [146], "aperture": "f/4 operator action required"},
                   "darker": {"starts": [80], "aperture": "f/8 operator action required"}},
        "start_selection": "static 40/384 bracket the target in pre-existing development and avoid the Su Tmax singular anchor; recovery 146/80 inherited from prior cohort and fixed before any new formal result",
        "fairness": "fresh matched interleaved cohort; same RAW ROI, DN target, gain, limits, metering, transition isolation, confirmation and accounting; fitted response/offset reused unchanged for proposed method",
        "claim_limit": "one scene; bounded RAW brightness protocol; baseline adaptations not original pipelines; no test tuning; old 48 episodes are a separate cohort",
        "selection_policy": "all formal episodes retained including failures; no best-of-repeat filtering",
    })
    args.protocol.parent.mkdir(parents=True, exist_ok=True)
    if args.protocol.exists():
        raise FileExistsError(args.protocol)
    write_json(args.protocol, protocol)
    print(json.dumps({"frozen_protocol": str(args.protocol), "methods": protocol["methods"]}, indent=2))


def acquire(args):
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    phase = args.phase
    starts = protocol["phases"][phase]["starts"] if phase != "preflight" else [40, 384]
    expected = 0 if phase == "preflight" else len(starts)*protocol["repetitions"]*len(protocol["methods"])
    plan = {"phase": phase, "condition_label": args.condition_label,
            "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
            "protocol": protocol, "starts": starts, "expected_episodes": expected}
    if not args.execute:
        print(json.dumps(plan, indent=2)); return
    from .hardware import OfficialCamera
    run = unique_path(args.output, "AE_published_"+phase)
    camera = OfficialCamera(run, [Action(protocol["cfg"]["gain"], n) for n in range(40, 513)])
    source_receipt(run)
    write_json(run/"T_plan.json", plan)
    result = {"status": "FAIL", "plan": plan, "episodes": [],
              "start_utc": datetime.now(timezone.utc).isoformat()}
    try:
        with camera:
            if phase == "preflight":
                observations = []
                for lines in starts:
                    camera.apply(Action(protocol["cfg"]["gain"], lines))
                    camera.capture("preflight_transition_l"+str(lines))
                    rows = []
                    for _ in range(2):
                        f = camera.capture("preflight_evaluation_l"+str(lines))
                        rows.append({**stripped(f), "measurement": measure(f["image"], tuple(protocol["cfg"]["roi"]))})
                    observations.append({"lines": lines, "frames": rows,
                                         "mean_dn": float(np.mean([r["measurement"]["mean_dn"] for r in rows]))})
                result["preflight"] = observations
                target, tol = protocol["cfg"]["target_dn"], protocol["cfg"]["tolerance_dn"]
                gate = (observations[0]["mean_dn"] < target-tol and observations[1]["mean_dn"] > target+tol
                        and all(r["measurement"]["clip_fraction"] == 0 for cell in observations for r in cell["frames"]))
                result["target_bracketed_without_clipping"] = gate
                camera.restore()
                result["status"] = "PREFLIGHT_PASS" if gate else "PREFLIGHT_SCENE_GATE_FAILED"
            else:
                block = 0
                for rep in range(protocol["repetitions"]):
                    for initial in starts:
                        methods = protocol["methods"]
                        order = methods[block % len(methods):]+methods[:block % len(methods)]
                        block += 1
                        for method in order:
                            eid = f"b{block}_{method}_l{initial}"
                            ep = run_episode(camera, method, initial, protocol, eid)
                            ep.update({"block_id": block, "rep": rep, "phase": phase, "order": order})
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
    if result["status"] not in ("AE_TRIALS_CAPTURE_COMPLETE", "PREFLIGHT_PASS"):
        raise RuntimeError(result.get("error", result["status"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest="command", required=True)
    f = s.add_parser("freeze")
    f.add_argument("--original", type=Path, required=True)
    f.add_argument("--protocol", type=Path, required=True)
    a = s.add_parser("acquire")
    a.add_argument("--protocol", type=Path, required=True)
    a.add_argument("--phase", choices=["preflight", "static", "brighter", "darker"], required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--condition-label", required=True)
    a.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if args.command == "freeze": freeze(args)
    else: acquire(args)


if __name__ == "__main__":
    main()

