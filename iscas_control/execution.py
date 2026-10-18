from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import json
import re
import time
import numpy as np

from .types import Action, Decision


PAYLOAD_BYTES = 8912896


class Rejected(RuntimeError):
    pass


class Phase(str, Enum):
    READY = "READY"
    APPLYING = "APPLYING"
    QUARANTINE = "QUARANTINE"
    OBSERVING = "OBSERVING"
    FAULT = "FAULT"
    STOPPED = "STOPPED"


@dataclass
class Budget:
    max_ms: float
    max_bytes: int
    start: float | None = None
    data_bytes: int = 0
    frames: int = 0
    clock: object = time.perf_counter

    def begin(self):
        if self.start is not None:
            raise Rejected("Budget already started")
        self.start = self.clock()

    @property
    def elapsed_ms(self):
        return 0. if self.start is None else (self.clock() - self.start) * 1000

    def permits(self, frames, predicted_ms=0):
        # Reserve metadata headroom rather than silently budgeting only RAW.
        return (self.data_bytes + frames * (PAYLOAD_BYTES + 20 + 16384) <= self.max_bytes
                and self.elapsed_ms + predicted_ms <= self.max_ms)

    def charge(self, nbytes, frame=False):
        if nbytes < 0:
            raise ValueError("Negative resource charge")
        self.data_bytes += int(nbytes)
        self.frames += int(frame)

    @property
    def overrun(self):
        return {"ms": max(0., self.elapsed_ms-self.max_ms),
                "bytes": max(0, self.data_bytes-self.max_bytes)}


class CostModel:
    """Measured request-to-completed-processing cycles; not a sum of pipelines."""
    def __init__(self, frame_ms=500., command_ms=50.):
        self.frame_samples = [float(frame_ms)]
        self.command_samples = defaultdict(lambda: [float(command_ms)])
        self.receipts = []

    @staticmethod
    def category(old, new):
        if old == new:
            return "hold"
        return "gain" if old.gain != new.gain else "exposure"

    def observe_frame(self, elapsed_ms):
        if elapsed_ms <= 0:
            raise ValueError("Positive measured cycle required")
        self.frame_samples.append(float(elapsed_ms))

    def observe_command(self, old, new, elapsed_ms):
        self.command_samples[self.category(old, new)].append(float(elapsed_ms))
        self.receipts.append({"old": old.key, "new": new.key, "elapsed_ms": elapsed_ms})

    def to_dict(self):
        return {"frame_samples_ms": self.frame_samples,
                "command_samples_ms": dict(self.command_samples), "receipts": self.receipts,
                "scope": "empirical completed-host cycles and command quantiles, not hardware latency bounds"}

    @classmethod
    def from_dict(cls, data):
        model = cls()
        if data.get("frame_samples_ms"):
            model.frame_samples = list(map(float, data["frame_samples_ms"]))
        for key, values in data.get("command_samples_ms", {}).items():
            model.command_samples[key] = list(map(float, values))
        model.receipts = data.get("receipts", [])
        return model

    @classmethod
    def from_camera_receipt(cls, data):
        model = cls()
        cycles = []
        fs = data["frames"]
        for a, b in zip(fs[:-1], fs[1:]):
            if a["role"] == b["role"] and a["commanded_action"] == b["commanded_action"]:
                dt = (b["trigger_tick_ns"]-a["trigger_tick_ns"])/1e6
                if dt > 0:
                    cycles.append(dt)
        if cycles:
            model.frame_samples = cycles
        grouped = defaultdict(list)
        def action(key):
            m = re.fullmatch(r"g([\d.]+)_l(\d+)", key)
            if not m:
                raise ValueError("Unknown action receipt")
            return Action(float(m[1]), int(m[2]))
        for row in data["commands"]:
            old, new = action(row["old"]), action(row["new"])
            category = model.category(old, new)
            if category != "hold":
                grouped[category].append((row["end_tick_ns"]-row["start_tick_ns"])/1e6)
        for k, values in grouped.items():
            model.command_samples[k] = values
        model.receipts = [{"source": "real camera command/trigger receipt", "frame_cycles": len(cycles),
                           "category_samples": {k: len(v) for k, v in grouped.items()}}]
        return model

    def estimate(self, old, new, evaluation_frames=2):
        category = self.category(old, new)
        count = evaluation_frames + int(category != "hold")
        frame = float(np.quantile(self.frame_samples[-64:], .95))
        command = 0. if category == "hold" else float(np.quantile(self.command_samples[category][-32:], .95))
        return {"frames": count, "ms": count*frame+command,
                "bytes_upper": count*(PAYLOAD_BYTES+20+16384), "category": category,
                "scope": "empirical upper quantile, not deterministic deadline guarantee"}


class Executor:
    """One writer, bounded actions, version rejection, transition quarantine.

    Host tags do NOT certify optical integration or sensor-frame settings.
    The backend must validate AA55 headers, RAW length and current configuration.
    """
    def __init__(self, backend, model, task, initial_action, budget, cost=None, journal=None):
        self.backend, self.model, self.task = backend, model, task
        self.current, self.budget = initial_action, budget
        self.cost = cost or CostModel()
        self.journal = journal
        self.phase = Phase.READY
        self.observation_seq = 0
        self.events, self.frames = [], []
        self.measurements = []

    def emit(self, kind, **fields):
        event = {"event": kind, "task_version": self.task.version,
                 "host_observation_seq": self.observation_seq, **fields}
        data = (json.dumps(event, sort_keys=True, allow_nan=False)+"\n").encode()
        self.budget.charge(len(data))
        self.events.append(event)
        if self.journal:
            self.journal(data)

    def _capture(self, role):
        start = time.perf_counter()
        try:
            frame = self.backend.capture(role)
        except Exception as exc:
            retained = getattr(exc, "captured_bytes", 0)
            if retained:
                self.budget.charge(retained)
                self.emit("PARTIAL_CAPTURE", retained_bytes=retained)
            raise
        self.budget.charge(frame["payload_bytes"] + frame.get("header_bytes", 20), frame=True)
        self.observation_seq += 1
        frame["host_observation_seq"] = self.observation_seq
        frame["task_version"] = self.task.version
        frame["commanded_action"] = self.current.key
        frame["parameter_attribution"] = "host_command_and_quarantine_not_sensor_hardware_tag"
        self.frames.append(frame)
        patches = self.model.estimator.patches(frame["image"])
        observation = self.model.observed(patches, self.current, self.task)
        self.emit("FRAME", role=role, raw=frame.get("raw"), sha256=frame.get("sha256"),
                  commanded_action=self.current.key,
                  measurement=[p.to_dict() for p in observation["points"]])
        self.cost.observe_frame((time.perf_counter()-start)*1000)
        return observation

    def observe(self, evaluation_frames=2):
        if self.phase in (Phase.FAULT, Phase.STOPPED):
            raise Rejected("Executor not observable")
        if not self.budget.permits(evaluation_frames):
            raise Rejected("OBSERVATION_BUDGET")
        self.measurements = []
        self.phase = Phase.OBSERVING
        observations = [self._capture("evaluation") for _ in range(evaluation_frames)]
        self.phase = Phase.READY
        self.measurements = observations
        return observations

    def execute(self, decision):
        # All validation precedes any device command or capture.
        if self.phase != Phase.READY:
            raise Rejected("Executor state not READY")
        if decision.task_version != self.task.version:
            self.emit("REJECT_STALE_TASK")
            raise Rejected("STALE_TASK_VERSION")
        if decision.observation_seq != self.observation_seq:
            self.emit("REJECT_STALE_OBSERVATION")
            raise Rejected("STALE_OBSERVATION")
        if decision.action is None:
            self.phase = Phase.STOPPED
            self.emit("STOP", reason=decision.reason)
            return []
        if decision.action.key not in self.model.actions:
            self.emit("REJECT_OUTSIDE_MODEL")
            raise Rejected("ACTION_OUTSIDE_MODEL")
        estimate = self.cost.estimate(self.current, decision.action)
        if not self.budget.permits(estimate["frames"], estimate["ms"]):
            self.emit("REJECT_BUDGET", estimate=estimate)
            raise Rejected("INSUFFICIENT_BUDGET")
        self.emit("DECISION_ACCEPTED", action=decision.action.key, reason=decision.reason,
                  cost_prediction=estimate)
        try:
            if decision.action != self.current:
                old = self.current
                self.measurements = []  # old-action measurements can never become new-action output
                self.phase = Phase.APPLYING
                start = time.perf_counter()
                receipt = self.backend.apply(decision.action)
                self.cost.observe_command(old, decision.action, (time.perf_counter()-start)*1000)
                self.current = decision.action
                self.emit("COMMAND", receipt=receipt)
                self.phase = Phase.QUARANTINE
                self._capture("transition_quarantine")
            result = self.observe(2)
            self.emit("PARAMETER_EVIDENCE", evidence="gain_readback_and_quarantine",
                      optical_integration_confirmed=False, hardware_frame_tag_available=False)
            return result
        except Exception as exc:
            self.phase = Phase.FAULT
            self.emit("FAULT", exception=repr(exc))
            raise

    def change_task(self, task):
        if task.version <= self.task.version or self.phase not in (Phase.READY, Phase.STOPPED):
            raise Rejected("Task update requires increasing version and idle executor")
        self.task = task
        self.phase = Phase.READY
        self.measurements = []
        self.emit("TASK_UPDATE", task_kind=task.kind.value)

    def commit(self):
        if not self.measurements:
            return None
        # No offline-best or past-frame search.
        frame = next(f for f in reversed(self.frames) if f["role"] == "evaluation")
        self.emit("OUTPUT_COMMITTED", raw=frame.get("raw"), action=self.current.key)
        self.phase = Phase.STOPPED
        return {"raw": frame.get("raw"), "action": self.current.key,
                "host_observation_seq": frame["host_observation_seq"]}
