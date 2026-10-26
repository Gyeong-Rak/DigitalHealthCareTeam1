#!/usr/bin/env python3
"""Stop only the verified source-study driver, letting it reap its torchrun group.

This file is standalone so it can be read from FETCH_HEAD before updating the
working checkout used by a running experiment. Never sends a broad pkill.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import time


def process_matches(pid, source):
    try:
        args = Path("/proc/{}/cmdline".format(pid)).read_bytes().decode().split("\0")
    except FileNotFoundError:
        return False
    scripts = {"run_calibration_study.py", "run_calibration_followup.py"}
    if not any(Path(arg).name in scripts for arg in args if arg):
        return False
    try:
        return Path(args[args.index("--output-dir") + 1]).expanduser().resolve() == source
    except (ValueError, IndexError):
        return False


def stop_study(source, timeout=40):
    source = Path(source).expanduser().resolve()
    state = json.loads((source / "status.json").read_text())
    lock_path = source / ".lock"
    if not lock_path.is_file():
        raise ValueError("Source lock file is missing; cannot safely identify an active driver")
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            print("Source driver is already stopped: " + str(source), flush=True)
            return
        except BlockingIOError:
            pass
        pid = state.get("pid")
        if type(pid) is not int or pid <= 1 or not process_matches(pid, source):
            raise ValueError("PID does not match this study; no signal was sent")
        current = state.get("current") or ""
        if not (current.startswith("combination_seed0_") or current in ("check_combination", "summarize")):
            raise ValueError("Refusing to stop an unexpected source stage: " + current)
        print("Stopping source driver PID {} ({}); completed conditions are preserved".format(pid, current), flush=True)
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                print("Source driver released its lock after cleanup", flush=True)
                return
            except BlockingIOError:
                time.sleep(.5)
        raise RuntimeError("Source did not stop within {}s. No new training was launched.".format(timeout))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        stop_study(args.output_dir)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, "Stop failed: {}\n".format(error))
