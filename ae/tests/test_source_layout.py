from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "relative_path",
    [
        "astra-sim/build/astra_analytical/build.sh",
        "astra-sim/build/astra_analytical/CMakeLists.txt",
        "astra-sim/extern/helper/fmt/include/fmt/core.h",
        "astra-sim/extern/helper/spdlog/include/spdlog/fmt/bundled/core.h",
        "third_party/ramulator2/src/frontend/impl/processor/simpleO3/core.cpp",
        "third_party/ramulator2/src/frontend/impl/processor/simpleO3/core.h",
    ],
)
def test_source_snapshot_contains_astra_build_entrypoints(relative_path: str) -> None:
    assert (REPO_ROOT / relative_path).is_file(), (
        "Required ASTRA source build entrypoint is missing from the repository: "
        f"{relative_path}"
    )
