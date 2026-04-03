#!/usr/bin/env python3
"""
plot_benchmarks.py — Generate benchmark charts from llamacpp_baselines.csv.

Reads:  benchmarks/results/llamacpp_baselines.csv
Writes: docs/benchmarks/throughput.png
        docs/benchmarks/latency.png
        docs/benchmarks/wall_clock.png
        docs/benchmarks/summary.md

Usage:
    python scripts/plot_benchmarks.py
    python scripts/plot_benchmarks.py --csv benchmarks/results/llamacpp_baselines.csv \
        --out-dir docs/benchmarks

Notes:
  * Requires Python 3 + matplotlib. No seaborn or other charting lib needed.
  * Uses the 'Agg' backend so it runs headless (no display required).
  * All figures include a disclaimer note: results are CPU-only / illustrative.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Matplotlib headless setup — MUST happen before any other matplotlib import
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 (intentional late import)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISCLAIMER = (
    "CPU-only \u2022 illustrative / experimental \u2014 not a reproducible performance claim"
)

# ---------------------------------------------------------------------------
# Label derivation helpers
# ---------------------------------------------------------------------------

def _run_label(row):
    """Derive a short human-readable label from a CSV row."""
    model = row.get("model_basename", "")
    quant = row.get("quant_hint", "")
    ctx   = row.get("context_tokens", "")
    gen   = row.get("requested_max_new_tokens", "")

    # Shorten the model name: qwen2.5-0.5b-instruct -> qwen25_05b
    name = model.lower()
    for sub in [".gguf", "-instruct", "-chat", "-base", "-" + quant.lower()]:
        name = name.replace(sub, "")
    name = name.replace(".", "").replace("-", "_")

    parts = [p for p in [name, quant, "ctx" + ctx, "gen" + gen] if p]
    return "_".join(parts)


def _unique_labels(rows):
    """Return per-row labels, appending (#2), (#3) ... for duplicates."""
    raw   = [_run_label(r) for r in rows]
    count = defaultdict(int)
    seen  = defaultdict(int)
    for lbl in raw:
        count[lbl] += 1

    result = []
    for lbl in raw:
        if count[lbl] > 1:
            seen[lbl] += 1
            result.append("{} (#{})".format(lbl, seen[lbl]))
        else:
            result.append(lbl)
    return result


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError("No data rows found in {!r}".format(path))
    return rows


def _f(row, key):
    """Return float value for key, or None if missing / blank / non-numeric."""
    val = row.get(key, "").strip()
    if not val or val.lower() == "null":
        return None
    try:
        return float(val)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _bar_chart(labels, values, title, ylabel, out_path, color="#4C72B0"):
    """Render a simple horizontal bar chart with value labels."""
    valid = [(lbl, v) for lbl, v in zip(labels, values) if v is not None]
    if not valid:
        print("  [skip] {} -- no data to plot".format(out_path.name), file=sys.stderr)
        return

    lbls, vals = zip(*valid)

    fig, ax = plt.subplots(figsize=(9, max(3, 0.9 * len(lbls) + 1.5)))
    bars = ax.barh(range(len(lbls)), vals, color=color, edgecolor="white")

    # Value labels at end of each bar
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_width() + max(vals) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            "{:.3g}".format(v),
            va="center",
            fontsize=9,
        )

    ax.set_yticks(range(len(lbls)))
    ax.set_yticklabels(lbls, fontsize=9)
    ax.set_xlabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlim(0, max(vals) * 1.18)
    ax.invert_yaxis()

    fig.text(
        0.5, -0.03, DISCLAIMER,
        ha="center", fontsize=7, color="grey",
        transform=ax.transAxes,
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  wrote {}".format(out_path))


def _latency_chart(labels, means, mins, maxs, out_path):
    """Horizontal bar chart for mean latency with min/max error bars."""
    tuples = [
        (lbl, mn, mi, mx)
        for lbl, mn, mi, mx in zip(labels, means, mins, maxs)
        if mn is not None
    ]
    if not tuples:
        print("  [skip] {} -- no data to plot".format(out_path.name), file=sys.stderr)
        return

    lbls = [t[0] for t in tuples]
    mn_v = [t[1] for t in tuples]
    lo_e = [t[1] - (t[2] if t[2] is not None else t[1]) for t in tuples]
    hi_e = [(t[3] if t[3] is not None else t[1]) - t[1] for t in tuples]

    fig, ax = plt.subplots(figsize=(9, max(3, 0.9 * len(lbls) + 1.5)))
    ax.barh(
        range(len(lbls)), mn_v,
        xerr=[lo_e, hi_e],
        color="#55A868", edgecolor="white",
        error_kw={"ecolor": "#2D6A4F", "capsize": 4, "linewidth": 1.2},
    )

    for i, v in enumerate(mn_v):
        ax.text(
            v + max(mn_v) * 0.01, i,
            "{:.0f} ms".format(v),
            va="center", fontsize=9,
        )

    ax.set_yticks(range(len(lbls)))
    ax.set_yticklabels(lbls, fontsize=9)
    ax.set_xlabel("Total inference latency -- mean (ms)  [error bars = min/max]", fontsize=10)
    ax.set_title("Inference Latency per Run", fontsize=11, pad=10)
    ax.set_xlim(0, max(mn_v) * 1.18)
    ax.invert_yaxis()

    fig.text(
        0.5, -0.03, DISCLAIMER,
        ha="center", fontsize=7, color="grey",
        transform=ax.transAxes,
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  wrote {}".format(out_path))


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def _write_summary(rows, labels, out_path):
    lines = [
        "# Benchmark Summary",
        "",
        "> **Experimental / illustrative** -- all runs are CPU-only on a single machine.",
        "> Numbers are included to confirm the end-to-end path works, not as",
        "> reproducible performance claims.",
        "",
        "| Run | Backend | Model | Quant | ctx | gen | tok/s (mean) | lat mean (ms) | lat min (ms) | lat p50 (ms) | lat p99 (ms) | wall clock (ms) |",
        "|-----|---------|-------|-------|-----|-----|--------------|---------------|--------------|--------------|--------------|-----------------|",
    ]

    for row, lbl in zip(rows, labels):
        def col(k):
            v = row.get(k, "").strip()
            return v if v else "--"

        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                lbl,
                col("backend_name"),
                col("model_basename"),
                col("quant_hint"),
                col("context_tokens"),
                col("requested_max_new_tokens"),
                col("generated_tokens_per_second_mean"),
                col("latency_mean_ms"),
                col("latency_min_ms"),
                col("latency_p50_ms"),
                col("latency_p99_ms"),
                col("wall_clock_ms"),
            )
        )

    lines += [
        "",
        "---",
        "",
        "## Notes",
        "",
        "- All runs use `llama.cpp` backend, CPU-only (no GPU offload).",
        "- Model: `qwen2.5-0.5b-instruct-q4_0.gguf` (Qwen2.5 0.5B Instruct, Q4_0 quantisation).",
        "- `tok/s` is the mean of per-iteration generated-tokens/latency values.",
        "- `wall_clock_ms` includes model load + all warmup and timed iterations.",
        "- Charts: [throughput.png](throughput.png) · [latency.png](latency.png) · [wall_clock.png](wall_clock.png)",
        "",
        "_Generated by `scripts/plot_benchmarks.py` from `benchmarks/results/llamacpp_baselines.csv`._",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  wrote {}".format(out_path))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark charts from llamacpp_baselines.csv"
    )
    parser.add_argument(
        "--csv",
        default="benchmarks/results/llamacpp_baselines.csv",
        help="Path to the CSV file (default: benchmarks/results/llamacpp_baselines.csv)",
    )
    parser.add_argument(
        "--out-dir",
        default="docs/benchmarks",
        help="Output directory for PNG and summary.md (default: docs/benchmarks)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir  = Path(args.out_dir)

    if not csv_path.exists():
        print("Error: CSV not found: {}".format(csv_path), file=sys.stderr)
        sys.exit(1)

    print("Reading {} ...".format(csv_path))
    rows   = load_csv(str(csv_path))
    labels = _unique_labels(rows)
    print("  {} run(s) found".format(len(rows)))
    for lbl in labels:
        print("    * {}".format(lbl))

    # ---- throughput chart ------------------------------------------------
    print("Generating throughput.png ...")
    _bar_chart(
        labels,
        [_f(r, "generated_tokens_per_second_mean") for r in rows],
        title="Throughput per Run",
        ylabel="Generated tokens / second (mean of per-iteration values)",
        out_path=out_dir / "throughput.png",
        color="#4C72B0",
    )

    # ---- latency chart ---------------------------------------------------
    print("Generating latency.png ...")
    _latency_chart(
        labels,
        means=[_f(r, "latency_mean_ms") for r in rows],
        mins =[_f(r, "latency_min_ms")  for r in rows],
        maxs =[_f(r, "latency_max_ms")  for r in rows],
        out_path=out_dir / "latency.png",
    )

    # ---- wall-clock chart -----------------------------------------------
    print("Generating wall_clock.png ...")
    _bar_chart(
        labels,
        [_f(r, "wall_clock_ms") for r in rows],
        title="Wall-Clock Time per Run  (model load + warmup + timed iters)",
        ylabel="Wall-clock time (ms)",
        out_path=out_dir / "wall_clock.png",
        color="#C44E52",
    )

    # ---- markdown summary -----------------------------------------------
    print("Generating summary.md ...")
    _write_summary(rows, labels, out_dir / "summary.md")

    print("Done.")


if __name__ == "__main__":
    main()
