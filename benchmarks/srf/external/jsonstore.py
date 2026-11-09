"""Locked, atomic read-modify-write of the benchmark result JSONs (several concurrent jobs may store keys of the same
file concurrently): an fcntl lock on <file>.lock around load / merge / write, the write to a temp file and os.replace."""
from __future__ import annotations

import fcntl
import json
import os
import tempfile
from pathlib import Path


def load(path) -> dict:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def update(path, key, val, merge: bool = False) -> None:
    """d[key] = val (merge=True: d[key].update(val) for dict values) under the lock, written atomically."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(str(p) + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        d = load(p)
        if merge and isinstance(d.get(key), dict):
            d[key].update(val)
        else:
            d[key] = val
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=p.name + ".")
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(d, indent=1, default=float) + "\n")
        os.replace(tmp, p)
