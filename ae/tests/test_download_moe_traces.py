from __future__ import annotations

import sys
from pathlib import Path


AE_DIR = Path(__file__).resolve().parents[1]
if str(AE_DIR) not in sys.path:
    sys.path.insert(0, str(AE_DIR))

from download_moe_traces import REVISION, TARGET_PROFILES


PREFIX = "cognitivecomputations/DeepSeek-R1-AWQ/mmlu"

# The deepest index each domain is read at by the camera-ready TP2 protocol.
# The static-mapping library takes professional_law[0:256] and
# miscellaneous[0:256]; epoch-T telemetry takes professional_law[256:416]; and
# the mapping stage still asks for a one-request epoch-T+1 segment at
# professional_law[416:417] even though --mapping-only discards it. Downloading
# only 416 professional_law requests makes that segment empty and the mapping
# stage fails on the released trace subset while passing on a full mirror.
REQUIRED_REQUESTS = {
    "professional_law": 417,
    "miscellaneous": 256,
    "high_school_mathematics": 90,
    "elementary_mathematics": 90,
    "human_aging": 90,
}


def profile_counts(profile: str) -> dict[str, int]:
    return {
        prefix.rsplit("/", 1)[-1]: limit for prefix, limit in TARGET_PROFILES[profile]
    }


def test_the_pinned_dataset_revision_is_unchanged() -> None:
    assert REVISION == "27febb7b2d24169560a9f6f83d38eb53f94916cf"


def test_the_camera_ready_profile_covers_every_slice_the_protocol_reads() -> None:
    counts = profile_counts("camera-ready-v1")

    assert counts == REQUIRED_REQUESTS


def test_the_camera_ready_profile_names_only_mmlu_domains() -> None:
    for prefix, _ in TARGET_PROFILES["camera-ready-v1"]:
        assert prefix.startswith(PREFIX + "/"), prefix


def test_professional_law_reaches_past_the_epoch_t_telemetry_window() -> None:
    # Guards the specific off-by-one that only shows up on the real subset.
    counts = profile_counts("camera-ready-v1")

    epoch_t_end = 416
    assert counts["professional_law"] > epoch_t_end


def test_the_evaluated_ae_v2_profile_is_left_alone() -> None:
    # The immutable ae-v2 snapshot keeps its own selection.
    assert TARGET_PROFILES["ae-v2"] == [
        (f"{PREFIX}/professional_law", 96),
        (f"{PREFIX}/miscellaneous", 96),
        ("cognitivecomputations/DeepSeek-R1-AWQ/livecodebench/execution", 440),
    ]
