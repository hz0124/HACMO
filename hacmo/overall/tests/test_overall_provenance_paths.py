from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_overall_rebuttal import pack_relative


def test_an_absolute_artifact_directory_is_named_by_pack_and_leaf() -> None:
    # This string is published in simulated_layers.csv, so it must not carry
    # the directory the authors happened to run in.
    named = pack_relative(
        "/authors/HACMO/hacmo/moe/outputs/tp2_pack_x/16w_4x4_rail_500"
    )

    assert named == "tp2_pack_x/16w_4x4_rail_500"
    assert not named.startswith("/")


def test_a_reviewer_run_and_an_author_run_name_the_same_input() -> None:
    author = pack_relative("/authors/HACMO/outputs/tp2_pack_x/16w_4x4_rail_500")
    reviewer = pack_relative("/reviewer/HACMO-AE/ae/work/tp2_pack_x/16w_4x4_rail_500")

    assert author == reviewer


def test_a_bare_name_and_an_empty_value_survive() -> None:
    assert pack_relative("metrics.csv") == "metrics.csv"
    assert pack_relative("") == ""
    assert pack_relative(None) == ""


def test_a_path_object_is_accepted() -> None:
    assert pack_relative(Path("/a/b/c/d.csv")) == "c/d.csv"
