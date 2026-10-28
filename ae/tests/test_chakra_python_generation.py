from __future__ import annotations

import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from generate_chakra_python import generate_binding


def test_chakra_binding_can_be_generated_with_system_protoc(tmp_path: Path) -> None:
    schema_dir = tmp_path / "protobuf"
    schema_dir.mkdir()
    shutil.copy2(
        REPO_ROOT
        / "astra-sim/extern/graph_frontend/chakra/schema/protobuf/et_def.proto",
        schema_dir / "et_def.proto",
    )

    output = generate_binding(
        schema_dir,
        grpc_protoc=None,
        protoc_executable="protoc",
    )

    assert output == schema_dir / "et_def_pb2.py"
    assert output.is_file()
    assert "DESCRIPTOR" in output.read_text(encoding="utf-8")
