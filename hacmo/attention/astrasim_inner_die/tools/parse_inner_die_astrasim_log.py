#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

WALL_RE = re.compile(r"sys\[(\d+)\],\s*Wall time:\s*(\d+)")
COMM_RE = re.compile(r"sys\[(\d+)\],\s*Comm time:\s*(\d+)")
REMOTE_MEM_RE = re.compile(r"sys\[(\d+)\],\s*Remote mem time:\s*(\d+)")
# logs can interleave; tolerate extra non-digit chars before the cycle number
REPORT_RE = re.compile(r"sys\[(\d+)\]\s*report called at[^0-9]*(\d+)\s*cycles")
FINISHED_RE = re.compile(r"sys\[(\d+)\]\s*finished,\s*(\d+)\s*cycles")


def parse_log(text: str) -> dict:
    wall = {}
    comm = {}
    remote = {}
    report = {}
    finished = {}
    for m in WALL_RE.finditer(text):
        wall[int(m.group(1))] = int(m.group(2))
    for m in COMM_RE.finditer(text):
        comm[int(m.group(1))] = int(m.group(2))
    for m in REMOTE_MEM_RE.finditer(text):
        remote[int(m.group(1))] = int(m.group(2))
    for m in REPORT_RE.finditer(text):
        report[int(m.group(1))] = int(m.group(2))
    for m in FINISHED_RE.finditer(text):
        finished[int(m.group(1))] = int(m.group(2))

    ranks = sorted(set(wall) | set(comm) | set(remote) | set(report) | set(finished))
    per_rank = []
    for r in ranks:
        per_rank.append(
            {
                "rank": r,
                "wall_time_cycles": wall.get(r),
                "comm_time_cycles": comm.get(r),
                "remote_mem_time_cycles": remote.get(r),
                "report_tick_cycles": report.get(r),
                "finished_cycles": finished.get(r),
            }
        )

    def _summary(values: list[int]) -> dict:
        if not values:
            return {"max": None, "avg": None}
        return {"max": max(values), "avg": sum(values) / len(values)}

    wall_vals = list(wall.values())
    report_vals = list(report.values())
    finished_vals = list(finished.values())
    comm_vals = list(comm.values())
    remote_vals = list(remote.values())
    summary = {
        "num_ranks_with_wall": len(wall),
        "num_ranks_with_report_tick": len(report),
        "num_ranks_with_finished_cycles": len(finished),
        "num_ranks_with_comm": len(comm_vals),
        "num_ranks_with_remote_mem": len(remote_vals),
        "wall_cycles": _summary(wall_vals),
        "report_tick_cycles": _summary(report_vals),
        "finished_cycles": _summary(finished_vals),
        "comm_cycles": _summary(comm_vals),
        "remote_mem_cycles": _summary(remote_vals),
    }
    return {"summary": summary, "per_rank": per_rank}


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse ASTRA-Sim inner-die run log.")
    parser.add_argument("--input_log", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    text = Path(args.input_log).read_text(encoding="utf-8", errors="ignore")
    result = parse_log(text)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote parsed summary: {args.output_json}")


if __name__ == "__main__":
    main()
