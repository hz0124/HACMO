#!/usr/bin/env python3
"""Generate the simplified Section 8.4 hardware-analysis figure.

The figure has two panels:
  (a) cluster optical power tax after a four-wafer cabinet boundary;
  (b) on-wafer area budget using FRED's original published area numbers.

All output stays under hacmo/hardware_analysis/sec84_hardware_summary.
"""

from __future__ import annotations

import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.text as mtext
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "sec84_hardware_summary"
CJK_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
CJK_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
CJK_CACHE_DIR = Path("/tmp/hacmo_sec84_fonts")
CJK_SC_REGULAR = CJK_CACHE_DIR / "NotoSansCJKsc-Regular.otf"
CJK_SC_BOLD = CJK_CACHE_DIR / "NotoSansCJKsc-Bold.otf"


@dataclass(frozen=True)
class SourceDatum:
    key: str
    value: float
    unit: str
    source: str
    url: str
    note: str


SOURCE_DATA = [
    SourceDatum(
        "fred_wafer_area",
        70000.0,
        "mm^2",
        "FRED arXiv:2406.19580v2, Sec. 6.2.2, lines 990-991",
        "https://arxiv.org/pdf/2406.19580",
        "FRED assumes a standard 300 mm wafer and reports 70,000 mm^2 wafer area.",
    ),
    SourceDatum(
        "fred_npu_io_area",
        26640.0,
        "mm^2",
        "FRED arXiv:2406.19580v2, Sec. 6.2.2, lines 1077-1080",
        "https://arxiv.org/pdf/2406.19580",
        "20 NPU chiplets plus 18 I/O controllers.",
    ),
    SourceDatum(
        "fred3_12_l1_switch_area",
        685.0,
        "mm^2",
        "FRED arXiv:2406.19580v2, Table 4, lines 1139-1142",
        "https://arxiv.org/pdf/2406.19580",
        "Used as the conservative area proxy for one HACMO side-local crossbar.",
    ),
    SourceDatum(
        "fred_total_fabric_area",
        25195.0,
        "mm^2",
        "FRED arXiv:2406.19580v2, Table 4, lines 1141-1148",
        "https://arxiv.org/pdf/2406.19580",
        "Reference only: full FRED switch-fabric area overhead.",
    ),
    SourceDatum(
        "single_cabinet_wafer_limit",
        4.0,
        "wafers/cabinet",
        "Corrected Section 8.4 hardware-analysis assumption",
        "paper/micro2026-paper928.pdf",
        "The cabinet radix constraint limits one cabinet to four wafers.",
    ),
    SourceDatum(
        "cisco_osfp_800g_low_power",
        14.0,
        "W/endpoint",
        "Cisco OSFP 800G Transceiver Modules Data Sheet, Table 2",
        "https://www.cisco.com/c/en/us/products/collateral/interfaces-modules/transceiver-modules/osfp-800g-transceiver-modules-ds.html",
        "Lower end of the 14-17 W optical endpoint range used for the power band.",
    ),
    SourceDatum(
        "cisco_osfp_800g_dr8_max_power",
        17.0,
        "W/endpoint",
        "Cisco OSFP 800G Transceiver Modules Data Sheet, Table 2",
        "https://www.cisco.com/c/en/us/products/collateral/interfaces-modules/transceiver-modules/osfp-800g-transceiver-modules-ds.html",
        "OSFP-800G-DR8/DR8P max power is listed as 16/17 W.",
    ),
    SourceDatum(
        "nvidia_passive_dac_osfp_end_power",
        0.1,
        "W/endpoint",
        "NVIDIA MCP7Y00-Nxxx 800Gb/s passive DAC product specifications",
        "https://docs.nvidia.com/networking/display/mcp7y00nxxx800pub",
        "NVIDIA lists OSFP ends at 0.1 W for passive DAC.",
    ),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cluster_rows = build_cluster_power_rows()
    area_rows = build_area_rows()
    write_csv(OUT_DIR / "sec84_cluster_power_data.csv", cluster_rows)
    write_csv(OUT_DIR / "sec84_wafer_area_data.csv", area_rows)
    write_source_data(OUT_DIR / "sec84_source_data.csv")
    for lang in ["en", "zh"]:
        plot_figure(lang, cluster_rows, area_rows)
    write_ppt_guide(OUT_DIR / "sec84_ppt_reproduction_guide.md", cluster_rows, area_rows)
    for lang in ["en", "zh"]:
        shutil.copy2(
            OUT_DIR / f"fig_sec84_hardware_analysis_{lang}.pdf",
            HERE / f"fig_sec84_hardware_analysis_{lang}.pdf",
        )
    print(f"wrote {OUT_DIR}")


def build_cluster_power_rows() -> list[dict[str, float | int | str]]:
    total_wafers = [4, 8, 16, 24, 32, 36, 64]
    single_cabinet_wafers = 4
    ports_per_4x4_wafer = 16
    optical_low_w = 14.0
    optical_high_w = 17.0
    passive_dac_w = 0.1
    rows = []
    for wafers in total_wafers:
        extra_wafers = max(wafers - single_cabinet_wafers, 0)
        links = extra_wafers * ports_per_4x4_wafer
        endpoints = 2 * links
        rows.append(
            {
                "total_wafers": wafers,
                "single_cabinet_wafers": single_cabinet_wafers,
                "total_cabinets": math.ceil(wafers / single_cabinet_wafers),
                "extra_cabinets_cross_rack": max(math.ceil(wafers / single_cabinet_wafers) - 1, 0),
                "extra_wafers_cross_rack": extra_wafers,
                "ports_per_4x4_wafer": ports_per_4x4_wafer,
                "cross_rack_links": links,
                "optical_endpoints": endpoints,
                "optical_power_low_w_per_endpoint": optical_low_w,
                "optical_power_high_w_per_endpoint": optical_high_w,
                "passive_dac_power_w_per_endpoint": passive_dac_w,
                "optical_power_kw_14w": endpoints * optical_low_w / 1000.0,
                "optical_power_kw_17w": endpoints * optical_high_w / 1000.0,
                "passive_dac_power_kw": endpoints * passive_dac_w / 1000.0,
            }
        )
    return rows


def build_area_rows() -> list[dict[str, float | str]]:
    wafer = 70000.0
    npu_io = 26640.0
    crossbar_single = 685.0
    hacmo_crossbars = 4 * crossbar_single
    available = wafer - npu_io - hacmo_crossbars
    fred_fabric = 25195.0
    return [
        {
            "component": "NPU + HBM + I/O controllers",
            "component_zh": "NPU + HBM + I/O 控制器",
            "area_mm2": npu_io,
            "share_of_wafer_pct": 100.0 * npu_io / wafer,
            "source": "FRED Sec. 6.2.2",
        },
        {
            "component": "HACMO side-local crossbars",
            "component_zh": "HACMO 侧边本地 crossbar",
            "area_mm2": hacmo_crossbars,
            "share_of_wafer_pct": 100.0 * hacmo_crossbars / wafer,
            "source": "4 x FRED3(12) L1 switch area from FRED Table 4",
        },
        {
            "component": "Remaining unclaimed area",
            "component_zh": "剩余未占用面积",
            "area_mm2": available,
            "share_of_wafer_pct": 100.0 * available / wafer,
            "source": "Derived from FRED wafer area",
        },
        {
            "component": "Reference: full FRED fabric",
            "component_zh": "参考：完整 FRED fabric",
            "area_mm2": fred_fabric,
            "share_of_wafer_pct": 100.0 * fred_fabric / wafer,
            "source": "FRED Table 4",
        },
    ]


def plot_figure(lang: str, cluster_rows: list[dict], area_rows: list[dict]) -> None:
    setup_fonts(lang)
    is_zh = lang == "zh"
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.6,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.unicode_minus": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.78), dpi=300)
    fig.subplots_adjust(wspace=0.36)
    plot_power_panel(axes[0], cluster_rows, is_zh)
    plot_area_panel(axes[1], area_rows, is_zh)
    if is_zh:
        apply_cjk_font(fig)
    fig.tight_layout(pad=0.5)
    fig.savefig(OUT_DIR / f"fig_sec84_hardware_analysis_{lang}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_power_panel(ax: plt.Axes, rows: list[dict], is_zh: bool) -> None:
    wafers = np.array([int(r["total_wafers"]) for r in rows])
    optical_low = np.array([float(r["optical_power_kw_14w"]) for r in rows])
    optical_high = np.array([float(r["optical_power_kw_17w"]) for r in rows])
    dac = np.array([float(r["passive_dac_power_kw"]) for r in rows])
    endpoints = np.array([int(r["optical_endpoints"]) for r in rows])
    x = np.arange(len(wafers))
    bar_color = "#D95F02"
    low_color = "#8C2D04"
    ref_color = "#4C78A8"
    optical_bar = ax.bar(
        x,
        optical_high,
        width=0.58,
        color=bar_color,
        alpha=0.88,
        label="Optical @17 W/endpoint" if not is_zh else "光模块 @17 W/端口",
    )
    lower_bound = ax.scatter(
        x,
        optical_low,
        marker="_",
        s=90,
        linewidths=1.8,
        color=low_color,
        label="14 W lower bound" if not is_zh else "14 W 下界",
        zorder=3,
    )
    (dac_line,) = ax.plot(
        x,
        dac,
        marker="o",
        linewidth=1.8,
        markersize=3.8,
        color=ref_color,
        label="Passive DAC ref." if not is_zh else "被动铜缆参考",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in wafers])
    ax.set_xlabel("Total wafers in deployment" if not is_zh else "部署中的总晶圆数")
    ax.set_ylabel("Inter-rack endpoint power (kW)" if not is_zh else "跨机柜端端口功耗 (kW)")
    ax.set_title("(a) Cluster power, 4 wafers/cabinet" if not is_zh else "(a) 集群功耗：4片晶圆/机柜")
    ax.set_ylim(0.0, max(optical_high) * 1.18)
    ax.grid(axis="y", alpha=0.22, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend([optical_bar, lower_bound, dac_line], [h.get_label() for h in [optical_bar, lower_bound, dac_line]], loc="upper left", frameon=False)
    summary = (
        "4->8: 128 endpoints, 1.8-2.2 kW\n36 wafers: 1024 endpoints, 14.3-17.4 kW"
        if not is_zh
        else "4->8片：128个端口，1.8-2.2 kW\n36片：1024个端口，14.3-17.4 kW"
    )
    ax.text(0.05, 0.61, summary, transform=ax.transAxes, ha="left", va="top", fontsize=6.8, bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.8))
    ax2 = ax.twinx()
    ax2.plot(x, endpoints, linestyle="--", linewidth=1.2, color="#555555", alpha=0.65)
    ax2.set_ylabel("# optical endpoints" if not is_zh else "光互联端口数")
    ax2.tick_params(axis="y", labelsize=7.4)
    ax2.spines["top"].set_visible(False)


def plot_area_panel(ax: plt.Axes, rows: list[dict], is_zh: bool) -> None:
    base_rows = rows[:3]
    labels = [r["component_zh"] if is_zh else r["component"] for r in base_rows]
    values = [float(r["area_mm2"]) for r in base_rows]
    colors = ["#8DA0CB", "#D62728", "#BDBDBD"]
    explode = [0.0, 0.085, 0.0]
    wedges, _ = ax.pie(
        values,
        colors=colors,
        explode=explode,
        startangle=100,
        counterclock=False,
        wedgeprops=dict(linewidth=0.7, edgecolor="white"),
    )
    ax.set_title("(b) On-wafer area budget" if not is_zh else "(b) 晶圆上面积预算")
    center_text = "HACMO crossbars\n2,740 mm²\n3.9% wafer" if not is_zh else "HACMO crossbar\n2,740 mm²\n占晶圆 3.9%"
    ax.text(0.0, 0.0, center_text, ha="center", va="center", fontsize=7.4, weight="bold", color="#A50F15")
    legend_labels = [
        f"{lab}: {val:,.0f} mm² ({100.0 * val / 70000.0:.1f}%)"
        for lab, val in zip(labels, values)
    ]
    ax.legend(wedges, legend_labels, loc="lower center", bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=6.45)
    ref = rows[3]
    note = (
        f"FRED full fabric ref.: {float(ref['area_mm2']):,.0f} mm² ({float(ref['share_of_wafer_pct']):.1f}%)"
        if not is_zh
        else f"参考：完整 FRED fabric 为 {float(ref['area_mm2']):,.0f} mm²（{float(ref['share_of_wafer_pct']):.1f}%）"
    )
    ax.text(0.5, -0.27, note, transform=ax.transAxes, ha="center", va="top", fontsize=6.55, color="#4D4D4D")


def setup_fonts(lang: str) -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    if lang != "zh":
        return
    for p in get_cjk_font_paths():
        if p.exists():
            fm.fontManager.addfont(str(p))


def apply_cjk_font(fig: plt.Figure) -> None:
    regular_path, bold_path = get_cjk_font_paths()
    if not regular_path.exists():
        return
    regular = fm.FontProperties(fname=str(regular_path))
    bold = fm.FontProperties(fname=str(bold_path if bold_path.exists() else regular_path))
    for text in fig.findobj(match=mtext.Text):
        size = text.get_fontsize()
        weight = text.get_fontweight()
        style = text.get_fontstyle()
        is_bold = str(weight).lower() in {"bold", "heavy", "semibold"} or (
            isinstance(weight, (int, float)) and weight >= 600
        )
        text.set_fontproperties(bold if is_bold else regular)
        text.set_fontsize(size)
        text.set_fontweight(weight)
        text.set_fontstyle(style)


def get_cjk_font_paths() -> tuple[Path, Path]:
    regular = extract_ttc_face(CJK_REGULAR, "NotoSansCJKsc-Regular", CJK_SC_REGULAR)
    bold = extract_ttc_face(CJK_BOLD, "NotoSansCJKsc-Bold", CJK_SC_BOLD)
    return regular, bold


def extract_ttc_face(ttc_path: Path, postscript_name: str, out_path: Path) -> Path:
    if out_path.exists():
        return out_path
    if not ttc_path.exists():
        return ttc_path
    try:
        from fontTools.ttLib import TTCollection
    except ImportError:
        return ttc_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    collection = TTCollection(str(ttc_path))
    for font in collection.fonts:
        if font["name"].getDebugName(6) == postscript_name:
            font.save(str(out_path))
            return out_path
    return ttc_path


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_source_data(path: Path) -> None:
    rows = [
        {
            "key": d.key,
            "value": d.value,
            "unit": d.unit,
            "source": d.source,
            "url": d.url,
            "note": d.note,
        }
        for d in SOURCE_DATA
    ]
    write_csv(path, rows)


def write_ppt_guide(path: Path, cluster_rows: list[dict], area_rows: list[dict]) -> None:
    row8 = next(r for r in cluster_rows if int(r["total_wafers"]) == 8)
    row36 = next(r for r in cluster_rows if int(r["total_wafers"]) == 36)
    hacmo = area_rows[1]
    guide = f"""# Section 8.4 Hardware Figure Reproduction Guide

## Output Files

- English PDF: `fig_sec84_hardware_analysis_en.pdf`
- Chinese PDF: `fig_sec84_hardware_analysis_zh.pdf`
- Cluster-power data: `sec84_cluster_power_data.csv`
- Wafer-area data: `sec84_wafer_area_data.csv`
- Source data table: `sec84_source_data.csv`

## Panel (a): Cluster Interconnect Power

Corrected formula with a four-wafer single-cabinet limit:

```text
single_cabinet_wafers = 4
ports_per_wafer = 4 * sqrt(16) = 16
cross_rack_links = max(total_wafers - 4, 0) * ports_per_wafer
optical_endpoints = 2 * cross_rack_links
optical_power_range_kW = optical_endpoints * [14 W, 17 W] / 1000
passive_DAC_ref_kW = optical_endpoints * 0.1 W / 1000
```

Adding one extra four-wafer cabinet, from 4 to 8 wafers, creates {int(row8['cross_rack_links'])} cross-rack links and {int(row8['optical_endpoints'])} optical endpoints, i.e. {float(row8['optical_power_kw_14w']):.2f}-{float(row8['optical_power_kw_17w']):.2f} kW optical endpoint power.

The 36-wafer point is now modeled as nine four-wafer cabinets.  It has {int(row36['optical_endpoints'])} optical endpoints and {float(row36['optical_power_kw_14w']):.2f}-{float(row36['optical_power_kw_17w']):.2f} kW optical endpoint power.  This replaces the inconsistent older 18-to-36-wafer estimate.

Power constants:

- Optical endpoint: 14-17 W range; the plotted bar uses 17 W and the short horizontal marker shows the 14 W lower bound.
- Passive DAC endpoint reference: 0.1 W, from NVIDIA MCP7Y00-Nxxx OSFP passive DAC specs.

## Panel (b): On-Wafer Area Budget

FRED source numbers:

```text
wafer_area = 70,000 mm^2
NPU + I/O controller area = 26,640 mm^2
FRED3(12) L1 switch area = 685 mm^2
full FRED fabric area = 25,195 mm^2
```

HACMO crossbar proxy:

```text
HACMO_crossbar_area = 4 * 685 = {float(hacmo['area_mm2']):.0f} mm^2
share_of_wafer = {float(hacmo['share_of_wafer_pct']):.2f}%
remaining_area = 70,000 - 26,640 - 2,740 = 40,620 mm^2
```

The right panel intentionally shows HACMO's side-local crossbars against the full FRED wafer area and the FRED NPU/I/O baseline, so the crossbar overhead is visually small and numerically traceable.

## PPT Recreation

1. Use `sec84_cluster_power_data.csv` for the left panel. Plot `optical_power_kw_17w` as orange bars, `optical_power_kw_14w` as the lower-bound tick marks, and `passive_dac_power_kw` as a blue line.
2. Use `sec84_wafer_area_data.csv` rows 1-3 for the pie chart. Highlight `HACMO side-local crossbars` in red.
3. Keep the source note from `sec84_source_data.csv` in speaker notes or backup slides.

## Source Links

- FRED paper: https://arxiv.org/pdf/2406.19580
- Cisco OSFP 800G data sheet: https://www.cisco.com/c/en/us/products/collateral/interfaces-modules/transceiver-modules/osfp-800g-transceiver-modules-ds.html
- NVIDIA passive DAC product spec: https://docs.nvidia.com/networking/display/mcp7y00nxxx800pub
"""
    path.write_text(guide, encoding="utf-8")


if __name__ == "__main__":
    main()
