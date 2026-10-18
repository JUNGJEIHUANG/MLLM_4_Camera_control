"""Adapter to the locally verified HJJ official AA55 transport.

No programming, power cycling, CSV editing, or acquisition occurs on import.
Output RAWs and receipts are new files; vendor sources are imported read-only.
"""
from __future__ import annotations

from dataclasses import asdict
import contextlib
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
import time
import numpy as np

from .types import Action


SOURCE = Path(r"E:\01_G6504_060\HJJ_developed\PC_Sensor_comm_scripts")
BASELINE = Action(1.04, 308)
LOCK = Path(r"P:\13_VLM_4_Camera_Control\T_camera_capture.lock")


class CaptureError(RuntimeError):
    def __init__(self, message, captured_bytes):
        super().__init__(message)
        self.captured_bytes = captured_bytes


class OfficialCamera:
    def __init__(self, output, actions, source=SOURCE):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output/"log").mkdir()
        self.allowed = set(actions) | {BASELINE, Action(75.2, 256)}
        self.source = Path(source)
        self.session = None
        self.current = BASELINE
        self.records, self.commands, self.hashes = [], [], set()
        self.stages = {"program": "not programmed by this adapter; current identity requires separate receipt"}
        self.lock_owned = False
        self.driver_bytes = 0

    def _save(self):
        data = {"status": self.stages, "frames": self.records, "commands": self.commands,
                "source": str(self.source), "current_commanded_action": self.current.key}
        (self.output/"T_camera_receipt.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    def operations(self, ops):
        with (self.output/"log"/"T_transport.log").open("a", encoding="utf-8") as log:
            with contextlib.redirect_stdout(log):
                return self.ctl.run_operations(self.session, ops)

    def read_sensor(self, address, count):
        return self.operations([
            ("read range", self.ctl.fpga_write_packet([(1, address), (2, count)]), -1, 0),
            ("read SPI", self.ctl.system_command_packet(5, ack_time=1), 0x84, count)])

    def read_state(self):
        return {"mode": self.read_sensor(0, 6), "line_time": self.read_sensor(0x70, 2),
                "gain": self.read_sensor(0xA0, 5)}

    def __enter__(self):
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, json.dumps({"pid": os.getpid(), "output": str(self.output)}).encode())
            os.close(fd)
            self.lock_owned = True
            known = {"cc2405_official_control.py": "d52ae539e5a2481c96555f5f2df3defe70fa885793fa0485f775d1217c2cdf12",
                     "sensor_aa55_packet.py": "762eb9185c346d0570dbd8d4bce4086ee2477433fde0504e36e4eaef23341c3d",
                     "sensor_aa55_usb_config.py": "3b9f5fc62f2b6ee285d9c44d99d8572d331e581be43ac20b790b24d314fad587",
                     "auto_exposure_std.py": "4951431235be1abe83a429ad8013f53889abde8eab74a8d733e3cd13c07cd4eb"}
            for name, expected in known.items():
                p = self.source/name
                if hashlib.sha256(p.read_bytes()).hexdigest() != expected:
                    raise RuntimeError(f"Vendor source changed: {p}; no device opened")
                (self.output/"log"/("T_source_"+name)).write_bytes(p.read_bytes())
            sys.path.insert(0, str(self.source))
            self.ctl = importlib.import_module("cc2405_official_control")
            self.ae = importlib.import_module("auto_exposure_std")
            self.session = self.ctl.DllSession(self.source/"third_party/cobra_dlls/gp_native_fx3.dll", 0, 10000)
            self.stages["usb_session"] = "OPEN"
            initial = self.read_state()
            self.stages["initial_readback"] = initial
            if initial["mode"][:4] != [1, 6, 56, 0] or initial["line_time"] != [69, 1] or initial["gain"][:2] != [1, 8]:
                raise RuntimeError("Current STD baseline mismatch; stop without exposure/gain writes")
            self.stages["config"] = "READBACK_MATCH"
            # Fresh training ONLY, no POWER_OFF/POWER_ON or mode rewrite.
            training = self.operations(self.ctl.build_mode_operations("std_gain1p04")[-1:])
            if len(training) != 112 or any(not training[i+6] for i in range(0, 112, 7)):
                raise RuntimeError("Training failed: stop before capture")
            self.stages["training"] = {"channels": 16, "failed": 0, "fresh": True}
            self.operations([("single frame", self.ctl.fpga_write_packet([(3, 1)]), -1, 0)])
            self.operations(self.ctl.build_sensor_write_operations([(0x20, 1), (0x21, 0)]))
            self.apply(BASELINE, force_exposure=True)
            self._save()
            return self
        except Exception as exc:
            self.stages["error"] = repr(exc)
            self._save()
            self.close()
            raise

    def apply(self, action, force_exposure=False):
        if action not in self.allowed:
            raise ValueError("Action not in frozen session domain")
        start = time.perf_counter_ns()
        old = self.current
        gain_changed = action.gain != old.gain
        if gain_changed:
            specs = {1.04: (1, 8), 13.43: (1, 29), 75.2: (0, 29)}
            c04 = self.read_sensor(4, 1)[0]
            a0, a1 = self.read_sensor(0xA0, 2)
            bit, code = specs[action.gain]
            seq = self.ae.build_sensor_gain_sequence_from_setting(
                self.ae.GainSetting(action.gain, bit, code), c04, a0, a1)
            self.operations(self.ctl.build_sensor_write_operations(list(seq)))
            got = self.read_sensor(0xA0, 2)
            if got != [(a0 & ~1) | bit, (a1 & ~31) | code]:
                raise RuntimeError("Gain readback mismatch; no blind retry")
        else:
            got = None
        if action.lines != old.lines or force_exposure or gain_changed:
            self.operations([("bounded exposure", self.ctl.fpga_write_packet(
                [(0x103, action.lines), (0x104, 0), (0x105, 0)]), -1, 0)])
        self.current = action
        receipt = {"old": old.key, "new": action.key, "gain_changed": gain_changed,
                   "exposure_changed": action.lines != old.lines or force_exposure,
                   "gain_readback": got, "start_tick_ns": start, "end_tick_ns": time.perf_counter_ns(),
                   "optical_integration_verified": False}
        self.commands.append(receipt)
        self._save()
        return receipt

    def capture(self, role):
        if self.session is None:
            raise RuntimeError("Session closed")
        seq = len(self.records)
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", role)
        prefix = f"T_{seq:05d}_{safe}"
        start = time.perf_counter_ns()
        packet = self.ctl.frame_request_packet()
        if self.session.write(packet) != len(packet):
            raise RuntimeError("Short FRAME_REQ")
        data = bytearray()
        first = b""
        try:
            first = self.session.read(262144)
            header, payload = self.ctl.parse_frame_header(first)
            if (header.columns, header.rows, header.frames, header.pixel_width) != (2176, 2048, 1, 12):
                raise RuntimeError("STD frame geometry mismatch")
            data.extend(payload[:header.image_bytes])
            while len(data) < header.image_bytes:
                remaining = header.image_bytes - len(data)
                block = self.session.read(262144)
                if len(block) < min(remaining, 262144):
                    raise RuntimeError("Short image payload")
                data.extend(block[:remaining])
        except Exception as exc:
            retained = first[:20]+data
            (self.output/f"{prefix}_PARTIAL.bin").write_bytes(retained)
            raise CaptureError(str(exc), len(retained)) from exc
        received = time.perf_counter_ns()
        raw = self.output/f"{prefix}_raw.bin"
        raw.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        if digest in self.hashes:
            raise CaptureError("Byte-identical frame; freshness unconfirmed", len(data)+20)
        self.hashes.add(digest)
        image = np.frombuffer(data, dtype="<u2").reshape(2048, 2176)
        if image.max() > 4095:
            raise CaptureError("Nonzero high container bits", len(data)+20)
        record = {"raw": str(raw), "sha256": digest, "role": role,
                  "payload_bytes": len(data), "header_bytes": 20,
                  "header_hex": first[:20].hex(), "header": asdict(header),
                  "trigger_tick_ns": start, "image_received_tick_ns": received,
                  "receive_ms": (received-start)/1e6, "commanded_action": self.current.key,
                  "full_clip_fraction": float(np.mean(image == 4095)),
                  "host_frame_sequence_only": seq}
        self.records.append(record)
        self.stages["last_readout"] = "AA55_ACK86_GEOMETRY_LENGTH_VALUE_RANGE_VALID"
        self._save()
        record["capture_return_tick_ns"] = time.perf_counter_ns()
        return {**record, "image": image}

    def restore(self):
        self.apply(BASELINE, force_exposure=True)
        for _ in range(3):
            self.capture("session_restore")
        self.operations(self.ctl.build_sensor_write_operations([(4, 18)]))
        final = self.read_state()
        self.stages["final_readback"] = final
        if final != self.stages["initial_readback"]:
            raise RuntimeError("Final sensor configuration differs from initial")
        self.stages["restoration"] = "BASELINE_COMMANDED_AND_SENSOR_READBACK_MATCH"
        self._save()

    def close(self):
        if self.session:
            self.session.close()
            self.session = None
        if self.lock_owned:
            LOCK.unlink()
            self.lock_owned = False

    def __exit__(self, exc_type, exc, tb):
        if exc:
            self.stages["error"] = repr(exc)
            self.stages["failure_policy"] = "Stopped; no blind retry, power cycle, or automatic restoration after transport fault"
        self._save()
        self.close()
