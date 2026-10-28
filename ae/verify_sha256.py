#!/usr/bin/env python3
"""Verify a GNU-style SHA256SUMS file without external utilities."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


def main() -> None:
    manifest = Path(sys.argv[1]).resolve()
    failures: list[str] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        path = manifest.parent / relative.lstrip("*")
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(block)
        digest = hasher.hexdigest()
        if digest != expected:
            failures.append(str(path))
    if failures:
        raise SystemExit("checksum mismatch: " + ", ".join(failures))
    print("PASS: data checksums")


if __name__ == "__main__":
    main()
