#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
from pathlib import Path


def _load_ablation_rows(csv_path: Path) -> list[dict]:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    picked = []
    for r in rows:
        if r["baseline"] not in {"A", "B", "D", "O"}:
            continue
        picked.append(
            {
                "kind": "ablation",
                "label": f"{r['baseline']}_bs{r['batch_size']}",
                "baseline": r["baseline"],
                "batch_size": int(r["batch_size"]),
                "compute_us": float(r["T_comp_us"]),
                "select_hbm_bytes": int(round(2 * int(r["batch_size"]) * 64 * 32768)),
                "qkv_hbm_bytes": int(round(float(r["bytes_from_hbm"]))),
                "remote_sram_bytes": int(round(float(r["bytes_from_sram_remote"]))),
                "formula_total_us": float(r["T_attn_total_us"]),
                "formula_qkv_mem_us": float(r["T_qkv_mem_us"]),
            }
        )
    return picked


def _load_motivation_rows(csv_path: Path) -> list[dict]:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    picked = []
    topk = 2048
    selected_vector_bytes = 576
    for r in rows:
        b = int(r["batch_size"])
        hit = float(r["hit_rate"])
        v_qkv = 2 * b * selected_vector_bytes * topk
        qkv_hbm = int(round(v_qkv * (1.0 - hit)))
        remote = int(round(v_qkv * (hit * 0.7)))  # Baseline C split
        picked.append(
            {
                "kind": "motivation",
                "label": f"C{int(float(r['cache_ratio'])*100)}_bs{b}",
                "baseline": "C",
                "batch_size": b,
                "cache_ratio": float(r["cache_ratio"]),
                "compute_us": 0.0,
                "select_hbm_bytes": int(round(2 * b * 64 * 32768)),
                "qkv_hbm_bytes": qkv_hbm,
                "remote_sram_bytes": remote,
                "formula_qkv_mem_us": float(r["T_qkv_mem_us"]),
            }
        )
    return picked


def _run_cmd(cmd: list[str], cwd: Path, log_path: Path) -> None:
    env = dict(**__import__("os").environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{cwd}:{existing}" if existing else str(cwd)
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=f, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}; log={log_path}")


def _parse_case_json(path: Path) -> dict:
    pj = json.loads(path.read_text(encoding="utf-8"))
    s = pj["summary"]
    wall_max = s["wall_cycles"]["max"]
    report_max = s["report_tick_cycles"]["max"]
    finished_max = s["finished_cycles"]["max"]
    if s.get("num_ranks_with_finished_cycles", 0) > s.get("num_ranks_with_wall", 0):
        wall_max = finished_max
    elif s.get("num_ranks_with_report_tick", 0) > s.get("num_ranks_with_wall", 0):
        wall_max = report_max
    return {
        "wall_us": (wall_max / 1000.0) if wall_max is not None else None,
        "report_us": (report_max / 1000.0) if report_max is not None else None,
        "finished_us": (finished_max / 1000.0) if finished_max is not None else None,
        "comm_us": (s["comm_cycles"]["max"] / 1000.0) if s["comm_cycles"]["max"] is not None else None,
        "remote_mem_us": (s["remote_mem_cycles"]["max"] / 1000.0) if s["remote_mem_cycles"]["max"] is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inner-die ASTRA-Sim memory-driven attention pipeline.")
    parser.add_argument("--astra_sim_dir", type=str, required=True)
    parser.add_argument("--ablation_csv", type=str, required=True)
    parser.add_argument("--motivation_csv", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--network_config", type=str, required=True)
    parser.add_argument("--system_config", type=str, required=True)
    parser.add_argument("--remote_memory_config", type=str, required=True)
    parser.add_argument("--num_ranks", type=int, default=361)
    parser.add_argument("--active_ranks", type=int, default=360)
    parser.add_argument("--qkv_effective_payload_bytes", type=int, default=0)
    parser.add_argument("--qkv_transfer_granularity_bytes", type=int, default=0)
    parser.add_argument("--kinds", nargs="+", choices=["ablation", "motivation"], default=["ablation", "motivation"])
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=None)
    parser.add_argument("--max_cases", type=int, default=None)
    args = parser.parse_args()

    root = Path(args.output_root).resolve()
    traces = root / "traces"
    logs = root / "logs"
    parsed = root / "parsed"
    csv_dir = root / "csv"
    for p in [traces, logs, parsed, csv_dir]:
        p.mkdir(parents=True, exist_ok=True)

    astra_bin = Path(args.astra_sim_dir) / "build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware"
    if not astra_bin.exists():
        raise FileNotFoundError(f"ASTRA binary not found: {astra_bin}")

    cases = []
    if "ablation" in args.kinds:
        cases.extend(_load_ablation_rows(Path(args.ablation_csv)))
    if "motivation" in args.kinds:
        cases.extend(_load_motivation_rows(Path(args.motivation_csv)))
    if args.batch_sizes:
        bs_set = set(args.batch_sizes)
        cases = [c for c in cases if c["batch_size"] in bs_set]
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    generator = Path(__file__).resolve().parent / "generate_inner_die_attention_trace.py"
    parser_py = Path(__file__).resolve().parent / "parse_inner_die_astrasim_log.py"

    summary_rows = []
    for case in cases:
        trace_prefix = "inner_die_attn"
        # Variant 1: HBM-only (MEM_LOAD path)
        hbm_case_dir = traces / f"{case['label']}_hbm"
        gen_hbm_cmd = [
            "python3", str(generator),
            "--num_ranks", str(args.num_ranks),
            "--active_ranks", str(args.active_ranks),
            "--trace_prefix", trace_prefix,
            "--output_dir", str(hbm_case_dir),
            "--compute_us", str(case["compute_us"]),
            "--select_hbm_bytes", str(case["select_hbm_bytes"]),
            "--qkv_hbm_bytes", str(case["qkv_hbm_bytes"]),
            "--qkv_effective_payload_bytes", str(args.qkv_effective_payload_bytes),
            "--qkv_transfer_granularity_bytes", str(args.qkv_transfer_granularity_bytes),
            "--remote_sram_bytes", "0",
            "--remote_mode", "p2p",
        ]
        _run_cmd(gen_hbm_cmd, cwd=Path(args.astra_sim_dir), log_path=logs / f"{case['label']}_hbm_gen.log")
        hbm_run_log = logs / f"{case['label']}_hbm.log"
        hbm_run_cmd = [
            str(astra_bin),
            f"--workload-configuration={hbm_case_dir / trace_prefix}",
            f"--system-configuration={Path(args.system_config).resolve()}",
            f"--remote-memory-configuration={Path(args.remote_memory_config).resolve()}",
            f"--network-configuration={Path(args.network_config).resolve()}",
        ]
        _run_cmd(hbm_run_cmd, cwd=Path(args.astra_sim_dir), log_path=hbm_run_log)
        hbm_parsed_json = parsed / f"{case['label']}_hbm.json"
        _run_cmd(
            ["python3", str(parser_py), "--input_log", str(hbm_run_log), "--output_json", str(hbm_parsed_json)],
            cwd=Path(args.astra_sim_dir),
            log_path=logs / f"{case['label']}_hbm_parse.log",
        )
        hbm_metrics = _parse_case_json(hbm_parsed_json)

        # Variant 2: remote-SRAM-only (COMM path), avoids REMOTE_MEM+COMM crash in this branch.
        remote_metrics = {"wall_us": 0.0, "report_us": 0.0, "finished_us": 0.0, "comm_us": 0.0, "remote_mem_us": 0.0}
        if case["remote_sram_bytes"] > 0:
            remote_case_dir = traces / f"{case['label']}_remote"
            gen_remote_cmd = [
                "python3", str(generator),
                "--num_ranks", str(args.num_ranks),
                "--active_ranks", str(args.active_ranks),
                "--trace_prefix", trace_prefix,
                "--output_dir", str(remote_case_dir),
                "--compute_us", "0",
                "--select_hbm_bytes", "0",
                "--qkv_hbm_bytes", "0",
                "--remote_sram_bytes", str(case["remote_sram_bytes"]),
                "--remote_mode", "p2p",
            ]
            _run_cmd(gen_remote_cmd, cwd=Path(args.astra_sim_dir), log_path=logs / f"{case['label']}_remote_gen.log")
            remote_run_log = logs / f"{case['label']}_remote.log"
            # COMM-only run does not need remote-memory backend.
            no_mem_cfg = Path(args.remote_memory_config).resolve().parent / "no_memory_expansion.json"
            if not no_mem_cfg.exists():
                no_mem_cfg.write_text('{\n  "memory-type": "NO_MEMORY_EXPANSION"\n}\n', encoding="utf-8")
            remote_run_cmd = [
                str(astra_bin),
                f"--workload-configuration={remote_case_dir / trace_prefix}",
                f"--system-configuration={Path(args.system_config).resolve()}",
                f"--remote-memory-configuration={no_mem_cfg}",
                f"--network-configuration={Path(args.network_config).resolve()}",
            ]
            _run_cmd(remote_run_cmd, cwd=Path(args.astra_sim_dir), log_path=remote_run_log)
            remote_parsed_json = parsed / f"{case['label']}_remote.json"
            _run_cmd(
                ["python3", str(parser_py), "--input_log", str(remote_run_log), "--output_json", str(remote_parsed_json)],
                cwd=Path(args.astra_sim_dir),
                log_path=logs / f"{case['label']}_remote_parse.log",
            )
            remote_metrics = _parse_case_json(remote_parsed_json)

        combined_wall = (hbm_metrics["wall_us"] or 0.0) + (remote_metrics["wall_us"] or 0.0)
        summary_rows.append(
            {
                "kind": case["kind"],
                "label": case["label"],
                "baseline": case["baseline"],
                "batch_size": case["batch_size"],
                "cache_ratio": case.get("cache_ratio"),
                "compute_us_input": case["compute_us"],
                "select_hbm_bytes": case["select_hbm_bytes"],
                "qkv_hbm_bytes": case["qkv_hbm_bytes"],
                "remote_sram_bytes": case["remote_sram_bytes"],
                "astra_hbm_us": hbm_metrics["wall_us"],
                "astra_remote_sram_us": remote_metrics["wall_us"],
                "astra_wall_us": combined_wall,
                "astra_report_us": hbm_metrics["report_us"],
                "astra_finished_us": hbm_metrics["finished_us"],
                "astra_comm_us": remote_metrics["comm_us"],
                "astra_remote_mem_us": hbm_metrics["remote_mem_us"],
                "formula_total_us": case.get("formula_total_us"),
                "formula_qkv_mem_us": case.get("formula_qkv_mem_us"),
            }
        )
        print(f"[done] {case['label']}: astra_wall_us={summary_rows[-1]['astra_wall_us']}")

    out_csv = csv_dir / "inner_die_astrasim_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "kind", "label", "baseline", "batch_size", "cache_ratio",
            "compute_us_input", "select_hbm_bytes", "qkv_hbm_bytes", "remote_sram_bytes",
            "astra_hbm_us", "astra_remote_sram_us", "astra_wall_us",
            "astra_report_us", "astra_finished_us", "astra_comm_us", "astra_remote_mem_us",
            "formula_total_us", "formula_qkv_mem_us",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    (root / "raw_summary.json").write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
