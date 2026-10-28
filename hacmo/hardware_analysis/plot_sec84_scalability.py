import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ------------------------------
# Parameters from Section 8.4
# ------------------------------
C_RACK = 18
S_RACK = 18
R = 72
BW_SWITCH_PORT = 50.0   # GB/s
BW_WAFER_PORT = 900.0   # GB/s
P_OPTICAL = 17.0        # W per optical endpoint
P_OPTICAL_OPT = 8.0     # optimistic / future optics

# Sweeps
WAFER_SIDE = np.array([2, 3, 4, 5, 6])
WAFER_TOTAL = np.array([4, 8, 12, 16, 20, 24, 28, 32, 36])


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    here = Path(__file__).resolve().parent
    out_dir = here / "sec84_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------
    # Subplot (a): single-rack limit
    # ------------------------------
    n_dies = WAFER_SIDE ** 2
    labels = [f"{x}x{x}" for x in WAFER_SIDE]

    tau = np.ceil(n_dies / 20.0)  # Physical footprint equivalence
    w_pack = np.floor(C_RACK / tau).astype(int)
    w_net = (np.floor(R * BW_SWITCH_PORT / BW_WAFER_PORT) * np.ones_like(n_dies)).astype(int)
    w_max = np.minimum(w_pack, w_net).astype(int)
    s_req = (4 * np.sqrt(n_dies)).astype(int)
    is_feasible = (s_req <= S_RACK).astype(int)

    # ------------------------------
    # Subplot (b): cross-rack power tax
    # ------------------------------
    idx_4x4 = list(WAFER_SIDE).index(4)
    w_single = int(w_max[idx_4x4])
    ports_per_wafer = int(4 * math.sqrt(16))  # 16 for 4x4 wafer
    w_extra = np.maximum(0, WAFER_TOTAL - w_single)
    endpoints = 2 * ports_per_wafer * w_extra
    p_extra_kw = endpoints * P_OPTICAL / 1000.0
    p_extra_opt_kw = endpoints * P_OPTICAL_OPT / 1000.0

    # ------------------------------
    # CSV outputs (for PPT)
    # ------------------------------
    csv_left = out_dir / "sec84_scalability_left_panel.csv"
    rows_left = [
        [
            int(side), int(dies), f"{int(side)}x{int(side)}", int(t),
            int(pack), int(net), int(mx), int(sw), int(feasible),
        ]
        for side, dies, t, pack, net, mx, sw, feasible in zip(
            WAFER_SIDE, n_dies, tau, w_pack, w_net, w_max, s_req, is_feasible
        )
    ]
    write_csv(
        csv_left,
        [
            "wafer_side",
            "dies_per_wafer",
            "label",
            "tau_ceiling_N_over_20",
            "W_pack",
            "W_net",
            "W_max",
            "S_req",
            "is_switch_feasible_1_true",
        ],
        rows_left,
    )

    csv_right = out_dir / "sec84_scalability_right_panel.csv"
    rows_right = [
        [
            int(total), int(extra), int(endp),
            round(float(pk), 3), round(float(pk_opt), 3),
            int(w_single), int(ports_per_wafer),
        ]
        for total, extra, endp, pk, pk_opt in zip(
            WAFER_TOTAL, w_extra, endpoints, p_extra_kw, p_extra_opt_kw
        )
    ]
    write_csv(
        csv_right,
        [
            "total_wafers",
            "W_extra",
            "cross_rack_optical_endpoints",
            "extra_optical_power_kw_17w",
            "extra_optical_power_kw_8w",
            "single_rack_boundary_wafers",
            "ports_per_wafer_4x4",
        ],
        rows_right,
    )

    csv_full = out_dir / "sec84_scalability_full_for_ppt.csv"
    rows_full = []
    for row in rows_left:
        rows_full.append(["left_panel"] + row + ["", "", "", "", "", "", ""])
    for row in rows_right:
        rows_full.append(
            ["right_panel", "", "", "", "", "", "", "", ""] + row
        )
    write_csv(
        csv_full,
        [
            "panel",
            "wafer_side",
            "dies_per_wafer",
            "label",
            "tau_ceiling_N_over_20",
            "W_pack",
            "W_net",
            "W_max",
            "S_req",
            "is_switch_feasible_1_true",
            "total_wafers",
            "W_extra",
            "cross_rack_optical_endpoints",
            "extra_optical_power_kw_17w",
            "extra_optical_power_kw_8w",
            "single_rack_boundary_wafers",
            "ports_per_wafer_4x4",
        ],
        rows_full,
    )

    # ------------------------------
    # Plot
    # ------------------------------
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), dpi=320)

    color_pack = "#4E79A7"
    color_net = "#F28E2B"
    color_max = "#59A14F"
    color_switch = "#B07AA1"
    color_power = "#E15759"
    color_power_opt = "#76B7B2"

    # (a) single-rack limit
    ax_l = axes[0]
    x = np.arange(len(labels))
    ax_l.plot(x, w_pack, marker="o", linewidth=2.2, markersize=5, color=color_pack, label="Packaging limit")
    ax_l.plot(x, w_net, marker="s", linewidth=2.2, markersize=5, color=color_net, label="Network/radix limit")
    ax_l.plot(x, w_max, marker="^", linewidth=2.8, markersize=6, color=color_max, label="Actual rack limit")
    ax_l.set_xticks(x)
    ax_l.set_xticklabels(labels)
    ax_l.set_xlabel("Wafer scale (# dies per wafer)")
    ax_l.set_ylabel("Max wafers per rack")
    ax_l.set_title("(a) Single-rack scalability limit", fontsize=11)
    ax_l.set_ylim(0, max(max(w_pack), max(w_net)) + 4)
    ax_l.grid(True, alpha=0.25, linestyle="--", linewidth=0.7)

    ax_l2 = ax_l.twinx()
    ax_l2.plot(
        x, s_req, linestyle="--", linewidth=2.0, color=color_switch, label="Required switch trays",
    )
    ax_l2.axhline(S_RACK, linestyle=":", linewidth=2.0, color="#666666", label=f"Switch-tray budget = {S_RACK}")
    ax_l2.set_ylabel("Required switch trays/planes")
    ax_l2.set_ylim(6, max(max(s_req) + 6, S_RACK + 2))

    x4 = labels.index("4x4")
    x5 = labels.index("5x5")
    ax_l.annotate(
        "4x4 sweet spot",
        xy=(x4, w_max[x4]),
        xytext=(x4 - 1.1, w_max[x4] + 3.0),
        arrowprops=dict(arrowstyle="->", lw=1.2, color="#333333"),
        fontsize=9,
    )
    ax_l2.annotate(
        "5x5 infeasible",
        xy=(x5, s_req[x5]),
        xytext=(x5 - 1.0, s_req[x5] + 3.5),
        arrowprops=dict(arrowstyle="->", lw=1.2, color="#333333"),
        fontsize=9,
    )

    lines1, labels1 = ax_l.get_legend_handles_labels()
    lines2, labels2 = ax_l2.get_legend_handles_labels()
    ax_l.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right", frameon=True)

    # (b) cross-rack power tax
    ax_r = axes[1]
    ax_r.plot(
        WAFER_TOTAL,
        endpoints,
        marker="o",
        linewidth=2.4,
        markersize=5,
        color=color_net,
        label="Cross-rack optical endpoints",
    )
    ax_r.set_xlabel("Total wafers in deployment (4x4 wafer)")
    ax_r.set_ylabel("# cross-rack optical endpoints")
    ax_r.set_title("(b) Cross-rack power tax", fontsize=11)
    ax_r.grid(True, alpha=0.25, linestyle="--", linewidth=0.7)
    ax_r.axvline(w_single, linestyle=":", linewidth=2.0, color="#666666")
    ax_r.annotate(
        "Single-rack boundary\n(4 wafers)",
        xy=(w_single, 0),
        xytext=(w_single + 1.2, max(endpoints) * 0.20),
        arrowprops=dict(arrowstyle="->", lw=1.2, color="#333333"),
        fontsize=9,
    )

    ax_r2 = ax_r.twinx()
    ax_r2.plot(
        WAFER_TOTAL,
        p_extra_kw,
        marker="s",
        linewidth=2.2,
        markersize=4.8,
        color=color_power,
        label="Extra optical power (17W endpoint)",
    )
    ax_r2.plot(
        WAFER_TOTAL,
        p_extra_opt_kw,
        linestyle="--",
        linewidth=2.2,
        color=color_power_opt,
        label="Optimistic optics (8W endpoint)",
    )
    ax_r2.set_ylabel("Extra optical power (kW)")

    lines1, labels1 = ax_r.get_legend_handles_labels()
    lines2, labels2 = ax_r2.get_legend_handles_labels()
    ax_r.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left", frameon=True)

    fig.suptitle(
        "Scalability Limits and Cross-rack Power Overhead of Layered Supernodes",
        y=1.02,
        fontsize=12.5,
    )
    fig.tight_layout()

    fig_path = out_dir / "fig_sec84_scalability.png"
    fig.savefig(fig_path, bbox_inches="tight")

    # ------------------------------
    # Short markdown note for paper text
    # ------------------------------
    md_path = out_dir / "sec84_figure_writeup_short.md"
    md_content = f"""# Section 8.4 Figure Write-up (Short)

## Figure Caption Suggestion
**Figure X. Scalability limits and cross-rack power overhead of layered supernodes.**
(a) Single-rack scalability is bounded by both packaging and network constraints, with the practical limit given by their minimum.
(b) Beyond the single-rack boundary (4 wafers for 4x4), cross-rack optical endpoints and power overhead grow approximately linearly with deployment size.

## How to describe this in the paper
This lightweight scalability sweep shows that cabinet-scale expansion is constrained primarily by interconnect limits rather than raw physical tray capacity.
Under the Section 8.4 settings (C_rack={C_RACK}, S_rack={S_RACK}, R={R}), the 4x4 wafer design remains feasible and reaches a practical single-rack sweet spot, while 5x5 becomes infeasible due to switch-tray demand exceeding the rack budget.
Once scaling crosses the single-rack boundary (W_single={w_single}), the number of cross-rack optical endpoints increases linearly, which translates into a near-linear extra optical power tax.
This supports a design strategy that first maximizes single-rack integration and only then scales out across racks when capacity requirements demand it.

## Files generated
- `fig_sec84_scalability.png`
- `sec84_scalability_left_panel.csv`
- `sec84_scalability_right_panel.csv`
- `sec84_scalability_full_for_ppt.csv`
"""
    md_path.write_text(md_content, encoding="utf-8")

    print(f"Saved figure: {fig_path}")
    print(f"Saved CSVs: {csv_left}, {csv_right}, {csv_full}")
    print(f"Saved note: {md_path}")


if __name__ == "__main__":
    main()
