"""Full validation must require built simulators, even when compilers exist."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_env


def test_missing_astra_binary_fails_full_preflight(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(check_env, "__file__", str(tmp_path / "ae/check_env.py"))
    monkeypatch.setattr(check_env, "MODULES", [])
    monkeypatch.setattr(check_env.platform, "system", lambda: "Linux")
    monkeypatch.setattr(check_env.resource, "getrlimit", lambda _: (8192, 8192))
    monkeypatch.setattr(check_env.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(sys, "argv", ["check_env.py", "--full"])
    for directory in ("astra-sim", "third_party/ramulator2"):
        path = tmp_path / directory
        path.mkdir(parents=True)
        (path / "source.txt").write_text("source present")
    binary = tmp_path / "build/third_party/ramulator2-build/ramulator2"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\nexit 1\n")
    binary.chmod(0o755)
    with pytest.raises(SystemExit, match="1"):
        check_env.main()
    assert "missing required ASTRA-Sim binary" in capsys.readouterr().err
