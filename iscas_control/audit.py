"""Offline independent validation of newly saved camera receipts and RAWs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def verify_camera_receipt(path):
    path = Path(path)
    j = json.loads(path.read_text(encoding="utf-8"))
    seen = set()
    previous_tick = -1
    for f in j["frames"]:
        p = Path(f["raw"])
        data = p.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if len(data) != 8912896 or len(data) != f["payload_bytes"] or digest != f["sha256"]:
            raise ValueError("RAW size/hash mismatch: "+str(p))
        if not p.name.startswith("T_") or digest in seen:
            raise ValueError("Output naming/freshness violation")
        seen.add(digest)
        header = bytes.fromhex(f["header_hex"])
        if len(header) != 20 or int.from_bytes(header[:4], "little") != 0xAA55 or header[7] != 0x86:
            raise ValueError("Header sync/ACK mismatch")
        if (int.from_bytes(header[8:10], "little"), int.from_bytes(header[10:12], "little"),
                int.from_bytes(header[12:14], "little"), header[16]) != (2176, 2048, 1, 12):
            raise ValueError("Header geometry mismatch")
        if np.frombuffer(data, dtype="<u2").max() > 4095:
            raise ValueError("Nonzero upper container bits")
        if f["trigger_tick_ns"] <= previous_tick or f["image_received_tick_ns"] <= f["trigger_tick_ns"]:
            raise ValueError("Invalid host timestamp ordering")
        previous_tick = f["trigger_tick_ns"]
    return {"status": "SAVED_RAW_AND_HEADER_AUDIT_PASS", "receipt": str(path),
            "frames_verified": len(j["frames"]),
            "sensor_readback_restored": j["status"].get("initial_readback") == j["status"].get("final_readback"),
            "not_certified": ["absolute optical integration", "hardware sensor-frame parameter tags",
                              "task accuracy or method superiority"]}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receipt", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    result = verify_camera_receipt(a.receipt)
    if a.output:
        a.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
