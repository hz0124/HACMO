#!/usr/bin/env python3
"""Generate the Python binding for the pinned Chakra execution-trace schema."""

from __future__ import annotations

import subprocess
from pathlib import Path

try:
    from grpc_tools import protoc as _grpc_protoc
except ImportError:  # Optional when a compatible system protoc is installed.
    _grpc_protoc = None


def generate_binding(
    schema_dir: Path,
    *,
    grpc_protoc=_grpc_protoc,
    protoc_executable: str = "protoc",
) -> Path:
    proto = schema_dir / "et_def.proto"
    output = schema_dir / "et_def_pb2.py"

    if not proto.is_file():
        raise FileNotFoundError(f"missing pinned Chakra schema: {proto}")
    if output.is_file() and output.stat().st_mtime_ns >= proto.stat().st_mtime_ns:
        return output

    if grpc_protoc is not None:
        result = grpc_protoc.main(
            [
                "grpc_tools.protoc",
                f"--proto_path={schema_dir}",
                f"--python_out={schema_dir}",
                str(proto),
            ]
        )
    else:
        result = subprocess.run(
            [
                protoc_executable,
                f"--proto_path={schema_dir}",
                f"--python_out={schema_dir}",
                str(proto),
            ],
            check=False,
        ).returncode
    if result != 0 or not output.is_file():
        raise RuntimeError(f"failed to generate Chakra Python binding (protoc exit {result})")
    return output


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    schema_dir = root / "astra-sim" / "extern" / "graph_frontend" / "chakra" / "schema" / "protobuf"
    output = generate_binding(schema_dir)
    print(f"[chakra-python] Ready: {output.relative_to(root)}")


if __name__ == "__main__":
    main()
