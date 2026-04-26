#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import shlex
import struct
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_OFFICIAL = ROOT.parent / "song"

COLORS = {
    "repro_cpu_raw": "#4E79A7",
    "official_cpu_raw": "#9C755F",
    "repro_gpu_raw": "#59A14F",
    "official_gpu_raw": "#2F6B2F",
    "repro_gpu_hash": "#E15759",
}

LABELS = {
    "repro_cpu_raw": "Repro CPU Raw",
    "official_cpu_raw": "Official CPU Raw (wall)",
    "repro_gpu_raw": "Repro GPU Raw",
    "official_gpu_raw": "Official GPU Raw (wall)",
    "repro_gpu_hash": "Repro GPU Hash",
}

DEFAULT_METHOD_ORDER = [
    "repro_cpu_raw",
    "repro_gpu_raw",
    "repro_gpu_hash",
    "official_cpu_raw",
    "official_gpu_raw",
]

METRICS = [
    {"key": "build_ms", "title": "Build Time (ms)", "higher_is_better": False, "log_scale": True},
    {"key": "search_ms", "title": "Query Time (ms)", "higher_is_better": False, "log_scale": True},
    {"key": "qps", "title": "QPS", "higher_is_better": True, "log_scale": True},
    {"key": "recall", "title": "Recall", "higher_is_better": True, "log_scale": False},
]


def _tail_file(path, n=40):
    path = Path(path)
    if not path.exists():
        return ""
    try:
        lines = path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception as exc:
        return f"<failed to read {path}: {exc}>"


def run(cmd, cwd=None, stdout_path=None, stderr_path=None, env=None):
    cwd = str(cwd) if cwd else None
    cmd_str = " ".join(map(shlex.quote, cmd))
    loc = cwd or os.getcwd()
    print(f"[cmd] cwd={loc}", flush=True)
    print(f"[cmd] run: {cmd_str}", flush=True)
    if stdout_path:
        print(f"[cmd] stdout -> {stdout_path}", flush=True)
    if stderr_path:
        print(f"[cmd] stderr -> {stderr_path}", flush=True)

    stdout_f = open(stdout_path, "w") if stdout_path else subprocess.PIPE
    stderr_f = open(stderr_path, "w") if stderr_path else subprocess.PIPE
    try:
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=stdout_f, stderr=stderr_f, text=True)
        dt_ms = (time.perf_counter() - t0) * 1000.0
    finally:
        if stdout_path:
            stdout_f.close()
        if stderr_path:
            stderr_f.close()

    print(f"[cmd] done: rc={proc.returncode} time_ms={dt_ms:.1f}", flush=True)
    if proc.returncode != 0:
        if stdout_path:
            tail = _tail_file(stdout_path)
            if tail:
                print(f"[cmd] stdout tail ({stdout_path}):\n{tail}", flush=True)
        elif proc.stdout:
            print(f"[cmd] stdout:\n{proc.stdout}", flush=True)
        if stderr_path:
            tail = _tail_file(stderr_path)
            if tail:
                print(f"[cmd] stderr tail ({stderr_path}):\n{tail}", flush=True)
        elif proc.stderr:
            print(f"[cmd] stderr:\n{proc.stderr}", flush=True)
        raise RuntimeError(f"command failed ({proc.returncode}): {cmd_str}")
    return dt_ms


def read_fvecs_subset(path: Path, limit: int):
    vecs = []
    with path.open("rb") as f:
        while len(vecs) < limit:
            head = f.read(4)
            if not head:
                break
            dim = struct.unpack("<i", head)[0]
            raw = f.read(4 * dim)
            if len(raw) != 4 * dim:
                break
            vec = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)
            vecs.append(vec)
    if not vecs:
        raise RuntimeError(f"empty fvecs: {path}")
    return np.stack(vecs, axis=0)


def compute_gt(base: np.ndarray, query: np.ndarray, k: int, block: int = 64):
    base = np.asarray(base, dtype=np.float32)
    query = np.asarray(query, dtype=np.float32)
    base_sq = np.sum(base * base, axis=1, dtype=np.float32)
    gt = np.empty((query.shape[0], k), dtype=np.int32)
    base_t = base.T.copy()
    for start in range(0, query.shape[0], block):
        qb = query[start : start + block]
        qb_sq = np.sum(qb * qb, axis=1, dtype=np.float32)
        dists = qb_sq[:, None] + base_sq[None, :] - 2.0 * (qb @ base_t)
        idx = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
        part = np.take_along_axis(dists, idx, axis=1)
        order = np.argsort(part, axis=1)
        gt[start : start + qb.shape[0]] = np.take_along_axis(idx, order, axis=1)
    return gt


def write_libsvm(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in arr:
            parts = ["0"]
            for j, val in enumerate(row, start=1):
                if val != 0.0:
                    parts.append(f"{j}:{float(val):.9g}")
            f.write(" ".join(parts) + "\n")


def read_ids(path: Path, nq: int, k: int):
    out = np.full((nq, k), -1, dtype=np.int32)
    row = 0
    with path.open() as f:
        for line in f:
            if row >= nq:
                break
            vals = []
            for tok in line.strip().split():
                try:
                    vals.append(int(tok))
                except ValueError:
                    break
                if len(vals) >= k:
                    break
            if not vals:
                continue
            out[row, : len(vals)] = vals
            row += 1
    return out


def recall_at_k(ids: np.ndarray, gt: np.ndarray, k: int):
    total = 0.0
    for i in range(min(len(ids), len(gt))):
        gt_set = set(int(x) for x in gt[i, :k])
        hit = sum(1 for x in ids[i, :k] if int(x) in gt_set)
        total += hit / k
    return total / len(gt)


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def load_rows_from_csv(path: Path):
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            item = {
                "method": row["method"],
                "build_ms": float(row["build_ms"]),
                "search_ms": float(row["search_ms"]),
                "qps": float(row["qps"]),
                "recall": float(row["recall"]),
                "k": int(float(row["k"])),
            }
            rows.append(item)
    if not rows:
        raise RuntimeError(f"empty metrics csv: {path}")
    return rows


def configure_matplotlib():
    plt.rcParams.update({
        "figure.dpi": 160,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.25,
    })


def metric_spec(key: str):
    for spec in METRICS:
        if spec["key"] == key:
            return spec
    raise KeyError(key)


def method_label(method: str):
    return LABELS.get(method, method)


def method_color(method: str):
    return COLORS.get(method, "#777777")


def sort_rows_default(rows):
    order = {name: idx for idx, name in enumerate(DEFAULT_METHOD_ORDER)}
    return sorted(rows, key=lambda r: (order.get(r["method"], len(order)), r["method"]))


def rank_map(rows, key: str, higher_is_better: bool):
    ordered = sorted(rows, key=lambda r: r[key], reverse=higher_is_better)
    return {row["method"]: idx + 1 for idx, row in enumerate(ordered)}


def best_row(rows, key: str, higher_is_better: bool):
    return sorted(rows, key=lambda r: r[key], reverse=higher_is_better)[0]


def enrich_rows(rows):
    rows = [dict(r) for r in sort_rows_default(rows)]
    baseline = next((r for r in rows if r["method"] == "official_gpu_raw"), rows[0])
    ranks = {}
    bests = {}
    for spec in METRICS:
        ranks[spec["key"]] = rank_map(rows, spec["key"], spec["higher_is_better"])
        bests[spec["key"]] = best_row(rows, spec["key"], spec["higher_is_better"])

    for row in rows:
        total_rank = 0.0
        for spec in METRICS:
            key = spec["key"]
            row[f"{key}_rank"] = ranks[key][row["method"]]
            total_rank += row[f"{key}_rank"]
        row["mean_rank"] = total_rank / len(METRICS)

        row["vs_official_gpu_build_speedup"] = baseline["build_ms"] / row["build_ms"]
        row["vs_official_gpu_search_speedup"] = baseline["search_ms"] / row["search_ms"]
        row["vs_official_gpu_qps_ratio"] = row["qps"] / baseline["qps"]
        row["vs_official_gpu_recall_gap"] = row["recall"] - baseline["recall"]
        row["vs_best_recall_gap"] = row["recall"] - bests["recall"]["recall"]
        row["vs_best_qps_ratio"] = row["qps"] / bests["qps"]["qps"]

    return rows, {"baseline": baseline, "bests": bests}


def format_metric(key: str, value: float):
    if key == "recall":
        return f"{value:.4f}"
    if key == "qps":
        return f"{value:,.2f}"
    return f"{value:,.2f}"


def format_signed(value: float, digits: int = 4):
    return f"{value:+.{digits}f}"


def metric_detail_rows(rows, key: str, higher_is_better: bool):
    return sorted(rows, key=lambda r: r[key], reverse=higher_is_better)


def plot_dashboard(rows, out_png: Path, out_pdf: Path, title: str):
    methods = [r["method"] for r in rows]
    labels = [method_label(m) for m in methods]
    colors = [method_color(m) for m in methods]
    build_ms = [r.get("build_ms", math.nan) for r in rows]
    search_ms = [r.get("search_ms", math.nan) for r in rows]
    qps = [r.get("qps", math.nan) for r in rows]
    recall = [r.get("recall", math.nan) for r in rows]

    configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=18, fontweight="bold")

    def bar(ax, values, name, logy=False):
        x = np.arange(len(labels))
        ax.bar(x, values, color=colors, edgecolor="#222", linewidth=0.6)
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_title(name)
        ax.grid(axis="y")
        if logy:
            ax.set_yscale("log")
        for xi, yi in zip(x, values):
            if math.isnan(yi):
                continue
            text = f"{yi:.4f}" if name.startswith("Recall") else (f"{yi:.2f}" if yi < 100 else f"{yi:.0f}")
            ax.text(xi, yi, text, ha="center", va="bottom", fontsize=9)

    bar(axes[0, 0], build_ms, "Build Time (ms)", logy=True)
    bar(axes[0, 1], search_ms, "Query Time (ms)", logy=True)
    bar(axes[1, 0], qps, "QPS", logy=True)
    bar(axes[1, 1], recall, f"Recall@{rows[0]['k']}")
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(rows, out_png: Path, out_pdf: Path, title: str):
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(9, 7))
    for row in rows:
        method = row["method"]
        marker = "s" if "gpu" in method else "o"
        ax.scatter(
            row["recall"],
            row["qps"],
            s=140,
            color=method_color(method),
            marker=marker,
            edgecolor="#222",
            linewidth=0.8,
        )
        ax.annotate(method_label(method), (row["recall"], row["qps"]), xytext=(8, 8), textcoords="offset points", fontsize=10)
    ax.set_xlabel(f"Recall@{rows[0]['k']}")
    ax.set_ylabel("QPS")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.set_title(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_metric_detail(rows, key: str, out_png: Path, out_pdf: Path):
    spec = metric_spec(key)
    ordered = metric_detail_rows(rows, key, spec["higher_is_better"])
    labels = [method_label(r["method"]) for r in ordered]
    values = [r[key] for r in ordered]
    colors = [method_color(r["method"]) for r in ordered]

    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, edgecolor="#222", linewidth=0.6)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(spec["title"])
    ax.set_title(f"Detailed {spec['title']} Comparison")
    ax.grid(axis="x")
    if spec["log_scale"]:
        ax.set_xscale("log")
    max_value = max(values)
    for yi, value in enumerate(values):
        label = format_metric(key, value)
        x_text = value * 1.05 if value > 0 else max_value * 0.02
        ax.text(x_text, yi, label, va="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_rank_heatmap(rows, out_png: Path, out_pdf: Path):
    ordered = sorted(rows, key=lambda r: (r["mean_rank"], r["method"]))
    rank_matrix = np.array([
        [r["build_ms_rank"], r["search_ms_rank"], r["qps_rank"], r["recall_rank"]]
        for r in ordered
    ], dtype=float)

    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    im = ax.imshow(rank_matrix, cmap="YlGn_r", aspect="auto")
    ax.set_xticks(range(4), ["Build", "Query", "QPS", "Recall"])
    ax.set_yticks(range(len(ordered)), [method_label(r["method"]) for r in ordered])
    ax.set_title("Metric Ranking Heatmap (1 = best)")
    for i in range(rank_matrix.shape[0]):
        for j in range(rank_matrix.shape[1]):
            ax.text(j, i, int(rank_matrix[i, j]), ha="center", va="center", fontsize=10, color="#111")
    fig.colorbar(im, ax=ax, shrink=0.9, label="Rank")
    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_baseline_comparison(rows, out_png: Path, out_pdf: Path, baseline_method: str):
    baseline = next((r for r in rows if r["method"] == baseline_method), None)
    if baseline is None:
        return

    methods = [r["method"] for r in rows]
    labels = [method_label(m) for m in methods]
    colors = [method_color(m) for m in methods]

    build_speedup = [baseline["build_ms"] / r["build_ms"] for r in rows]
    search_speedup = [baseline["search_ms"] / r["search_ms"] for r in rows]
    qps_ratio = [r["qps"] / baseline["qps"] for r in rows]
    recall_gap = [r["recall"] - baseline["recall"] for r in rows]

    configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Relative to {method_label(baseline_method)}", fontsize=18, fontweight="bold")

    def bar(ax, values, title, ref_line=1.0, signed=False):
        x = np.arange(len(labels))
        ax.bar(x, values, color=colors, edgecolor="#222", linewidth=0.6)
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_title(title)
        ax.grid(axis="y")
        ax.axhline(ref_line, color="#444", linewidth=1.0, linestyle="--")
        for xi, yi in zip(x, values):
            label = format_signed(yi, 3) if signed else f"{yi:.2f}x"
            ax.text(xi, yi, label, ha="center", va="bottom" if yi >= ref_line else "top", fontsize=9)

    bar(axes[0, 0], build_speedup, "Build Speedup", ref_line=1.0)
    bar(axes[0, 1], search_speedup, "Query Speedup", ref_line=1.0)
    bar(axes[1, 0], qps_ratio, "QPS Ratio", ref_line=1.0)
    bar(axes[1, 1], recall_gap, f"Recall@{rows[0]['k']} Gap", ref_line=0.0, signed=True)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_latency_recall_panels(rows, out_png: Path, out_pdf: Path):
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    pairs = [
        ("search_ms", "Query Time (ms) vs Recall"),
        ("build_ms", "Build Time (ms) vs Recall"),
    ]

    for ax, (metric, title) in zip(axes, pairs):
        for row in rows:
            method = row["method"]
            marker = "s" if "gpu" in method else "o"
            ax.scatter(
                row[metric],
                row["recall"],
                s=140,
                color=method_color(method),
                marker=marker,
                edgecolor="#222",
                linewidth=0.8,
            )
            ax.annotate(method_label(method), (row[metric], row["recall"]), xytext=(8, 8), textcoords="offset points", fontsize=9)
        ax.set_xscale("log")
        ax.set_xlabel(metric_spec(metric)["title"])
        ax.set_ylabel(f"Recall@{rows[0]['k']}")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def write_rankings_csv(path: Path, rows):
    flat = []
    for row in sorted(rows, key=lambda r: (r["mean_rank"], r["method"])):
        flat.append({
            "method": row["method"],
            "label": method_label(row["method"]),
            "build_ms": row["build_ms"],
            "build_ms_rank": row["build_ms_rank"],
            "search_ms": row["search_ms"],
            "search_ms_rank": row["search_ms_rank"],
            "qps": row["qps"],
            "qps_rank": row["qps_rank"],
            "recall": row["recall"],
            "recall_rank": row["recall_rank"],
            "mean_rank": row["mean_rank"],
        })
    write_csv(path, flat)


def write_baseline_csv(path: Path, rows):
    flat = []
    for row in rows:
        flat.append({
            "method": row["method"],
            "label": method_label(row["method"]),
            "build_speedup_vs_official_gpu": row["vs_official_gpu_build_speedup"],
            "search_speedup_vs_official_gpu": row["vs_official_gpu_search_speedup"],
            "qps_ratio_vs_official_gpu": row["vs_official_gpu_qps_ratio"],
            "recall_gap_vs_official_gpu": row["vs_official_gpu_recall_gap"],
            "recall_gap_vs_best": row["vs_best_recall_gap"],
        })
    write_csv(path, flat)


def summary_takeaways(rows, context):
    best_recall = context["bests"]["recall"]
    best_qps = context["bests"]["qps"]
    best_build = context["bests"]["build_ms"]
    best_search = context["bests"]["search_ms"]
    baseline = context["baseline"]

    best_repro = max(
        (r for r in rows if r["method"].startswith("repro_")),
        key=lambda r: r["recall"],
    )
    fastest_repro_query = min(
        (r for r in rows if r["method"].startswith("repro_")),
        key=lambda r: r["search_ms"],
    )

    return [
        f"Highest recall@{best_recall['k']}: {method_label(best_recall['method'])} at {best_recall['recall']:.4f}.",
        f"Highest QPS: {method_label(best_qps['method'])} at {best_qps['qps']:,.2f}, which is {best_qps['vs_official_gpu_qps_ratio']:.2f}x {method_label(baseline['method'])}.",
        f"Fastest build: {method_label(best_build['method'])} at {best_build['build_ms']:,.2f} ms.",
        f"Fastest query latency: {method_label(best_search['method'])} at {best_search['search_ms']:,.2f} ms.",
        f"Best reproduction recall: {method_label(best_repro['method'])} at {best_repro['recall']:.4f}, gap {abs(best_repro['vs_official_gpu_recall_gap']):.4f} from {method_label(baseline['method'])}.",
        f"Fastest reproduction query path: {method_label(fastest_repro_query['method'])} at {fastest_repro_query['search_ms']:,.2f} ms, {fastest_repro_query['vs_official_gpu_search_speedup']:.2f}x faster than {method_label(baseline['method'])}.",
    ]


def write_summary(path: Path, rows, args, notes, context):
    takeaways = summary_takeaways(rows, context)
    baseline = context["baseline"]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# SONG comparison report\n\n")
        f.write("## Experiment setup\n\n")
        f.write(f"- subset: n={args.n}, nq={args.nq}, dim={args.dim}, k={args.k}\n")
        f.write(f"- pq_size: {args.pq_size}\n")
        f.write(f"- official root: `{args.official_root}`\n")
        if getattr(args, "from_metrics", ""):
            f.write(f"- regenerated from existing metrics: `{args.from_metrics}`\n")
        f.write("\n## Main takeaways\n\n")
        for line in takeaways:
            f.write(f"- {line}\n")

        f.write("\n## Detailed metrics\n\n")
        f.write("| Method | Build ms | Build rank | Search ms | Search rank | QPS | QPS rank | Recall | Recall rank | Mean rank |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in sorted(rows, key=lambda r: (r["mean_rank"], r["method"])):
            f.write(
                "| "
                f"{method_label(row['method'])} | "
                f"{row['build_ms']:.2f} | {row['build_ms_rank']} | "
                f"{row['search_ms']:.2f} | {row['search_ms_rank']} | "
                f"{row['qps']:.2f} | {row['qps_rank']} | "
                f"{row['recall']:.4f} | {row['recall_rank']} | "
                f"{row['mean_rank']:.2f} |\n"
            )

        f.write(f"\n## Relative to {method_label(baseline['method'])}\n\n")
        f.write("| Method | Build speedup | Query speedup | QPS ratio | Recall gap |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                "| "
                f"{method_label(row['method'])} | "
                f"{row['vs_official_gpu_build_speedup']:.2f}x | "
                f"{row['vs_official_gpu_search_speedup']:.2f}x | "
                f"{row['vs_official_gpu_qps_ratio']:.2f}x | "
                f"{format_signed(row['vs_official_gpu_recall_gap'], 4)} |\n"
            )

        f.write("\n## Method-by-method notes\n\n")
        for row in rows:
            f.write(
                f"- {method_label(row['method'])}: build {row['build_ms']:.2f} ms, "
                f"query {row['search_ms']:.2f} ms, QPS {row['qps']:.2f}, recall {row['recall']:.4f}; "
                f"relative to {method_label(baseline['method'])}, build {row['vs_official_gpu_build_speedup']:.2f}x, "
                f"query {row['vs_official_gpu_search_speedup']:.2f}x, recall gap {format_signed(row['vs_official_gpu_recall_gap'], 4)}.\n"
            )

        f.write("\n## Useful output files\n\n")
        f.write("- `metrics.csv`: raw metric table\n")
        f.write("- `leaderboard.csv`: ranked table for report screenshots\n")
        f.write("- `vs_official_gpu.csv`: direct comparison against the official GPU baseline\n")
        f.write("- `presentation_notes.md`: short talking points for an oral report\n")
        f.write("- `figures/`: detailed figures, including per-metric plots and ranking heatmap\n")

        f.write("\n## Notes\n\n")
        for note in notes:
            f.write(f"- {note}\n")


def write_presentation_notes(path: Path, rows, args, context):
    takeaways = summary_takeaways(rows, context)
    best_recall = context["bests"]["recall"]
    best_qps = context["bests"]["qps"]
    best_repro = max((r for r in rows if r["method"].startswith("repro_")), key=lambda r: r["recall"])

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# Presentation notes\n\n")
        f.write("## One-paragraph summary\n\n")
        f.write(
            f"On the SIFT subset with n={args.n}, nq={args.nq}, dim={args.dim}, and Recall@{args.k}, "
            f"the strongest accuracy came from {method_label(best_recall['method'])} "
            f"with recall {best_recall['recall']:.4f}. The highest throughput came from "
            f"{method_label(best_qps['method'])} with {best_qps['qps']:.2f} QPS, but its recall was "
            f"{best_qps['recall']:.4f}. Among the reproduction variants, the best recall was "
            f"{best_repro['recall']:.4f}, which is {abs(best_repro['vs_official_gpu_recall_gap']):.4f} "
            f"below the official GPU baseline.\n"
        )

        f.write("\n## Suggested speaking points\n\n")
        for line in takeaways:
            f.write(f"- {line}\n")

        f.write("\n## Suggested report framing\n\n")
        f.write("- Start with recall first, because the reproduction/official gap is the main story in these numbers.\n")
        f.write("- Then explain the speed story separately: some reproduction paths are much faster, but they trade away substantial recall.\n")
        f.write("- Use `leaderboard.csv` and `figures/rank_heatmap.png` as compact summary slides.\n")


def build_manifest(out_dir: Path, figs: Path):
    return {
        "metrics_csv": str(out_dir / "metrics.csv"),
        "leaderboard_csv": str(out_dir / "leaderboard.csv"),
        "vs_official_gpu_csv": str(out_dir / "vs_official_gpu.csv"),
        "report_md": str(out_dir / "report.md"),
        "presentation_notes_md": str(out_dir / "presentation_notes.md"),
        "dashboard_png": str(figs / "dashboard.png"),
        "tradeoff_png": str(figs / "recall_vs_qps.png"),
        "build_detail_png": str(figs / "build_ms_detail.png"),
        "query_detail_png": str(figs / "search_ms_detail.png"),
        "qps_detail_png": str(figs / "qps_detail.png"),
        "recall_detail_png": str(figs / "recall_detail.png"),
        "rank_heatmap_png": str(figs / "rank_heatmap.png"),
        "baseline_compare_png": str(figs / "vs_official_gpu_raw.png"),
        "latency_recall_png": str(figs / "latency_vs_recall.png"),
    }


def write_metrics_csv(path: Path, rows):
    flat = []
    for row in sort_rows_default(rows):
        flat.append({
            "method": row["method"],
            "build_ms": row["build_ms"],
            "search_ms": row["search_ms"],
            "qps": row["qps"],
            "recall": row["recall"],
            "k": row["k"],
        })
    write_csv(path, flat)

def parse_int_list(text: str):
    values = []
    for tok in text.split(","):
        tok = tok.strip()
        if tok:
            values.append(int(tok))
    if not values:
        raise RuntimeError("empty pq list")
    return sorted(set(values))


def pq_tag(pq: int):
    return f"pq{pq:04d}"


def load_sweep_rows(path: Path, k: int):
    rows = []
    with path.open(newline="") as f:
        for raw in csv.DictReader(f):
            row = {
                "method": raw["method"],
                "pq": int(raw["pq"]),
                "build_ms": float(raw["build_ms"]),
                "search_ms": float(raw["search_ms_mean"]),
                "search_ms_std": float(raw["search_ms_std"]),
                "qps": float(raw["qps_mean"]),
                "qps_std": float(raw["qps_std"]),
                "recall": float(raw["recall"]),
                "k": k,
                "upload_ms": float(raw.get("upload_ms", 0) or 0),
                "query_projection_ms": float(raw.get("query_projection_ms", 0) or 0),
                "alloc_ms": float(raw.get("alloc_ms", 0) or 0),
                "h2d_ms": float(raw.get("h2d_ms", 0) or 0),
                "memset_ms": float(raw.get("memset_ms", 0) or 0),
                "kernel_ms": float(raw.get("kernel_ms", 0) or 0),
                "d2h_ms": float(raw.get("d2h_ms", 0) or 0),
                "visited_mean": float(raw.get("visited_mean", 0) or 0),
                "visited_p95": float(raw.get("visited_p95", 0) or 0),
                "expanded_mean": float(raw.get("expanded_mean", 0) or 0),
                "expanded_p95": float(raw.get("expanded_p95", 0) or 0),
                "neighbor_slots_mean": float(raw.get("neighbor_slots_mean", 0) or 0),
                "distance_evals_mean": float(raw.get("distance_evals_mean", 0) or 0),
                "distance_evals_p95": float(raw.get("distance_evals_p95", 0) or 0),
                "frontier_pushes_mean": float(raw.get("frontier_pushes_mean", 0) or 0),
                "result_updates_mean": float(raw.get("result_updates_mean", 0) or 0),
                "peak_frontier_mean": float(raw.get("peak_frontier_mean", 0) or 0),
                "peak_results_mean": float(raw.get("peak_results_mean", 0) or 0),
                "early_stop_rate": float(raw.get("early_stop_rate", 0) or 0),
                "ids_path": raw.get("ids_path", ""),
                "query_stats_path": raw.get("query_stats_path", ""),
                "batch_stats_path": raw.get("batch_stats_path", ""),
            }
            rows.append(row)
    if not rows:
        raise RuntimeError(f"empty sweep summary: {path}")
    return rows


def per_query_recalls(ids: np.ndarray, gt: np.ndarray, k: int):
    recalls = []
    for i in range(min(len(ids), len(gt))):
        gt_set = set(int(x) for x in gt[i, :k])
        hit = sum(1 for x in ids[i, :k] if int(x) in gt_set)
        recalls.append(hit / k)
    return recalls


def summarize_values(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
    }


def write_official_query_stats(path: Path, method: str, pq: int, ids: np.ndarray, gt: np.ndarray, k: int):
    recalls = per_query_recalls(ids, gt, k)
    rows = []
    for i, rec in enumerate(recalls):
        rows.append({"query_id": i, "method": method, "pq": pq, "recall": rec})
    write_csv(path, rows)
    return recalls


def capture_text(cmd, cwd=None):
    try:
        proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except FileNotFoundError as exc:
        return {
            "cmd": cmd,
            "returncode": 127,
            "stdout": "",
            "stderr": str(exc),
        }


def parse_config_snapshot(path: Path):
    snapshot = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("constexpr int"):
            lhs, rhs = line.split("=")
            name = lhs.split()[-1].strip()
            value = rhs.strip().rstrip(";")
            try:
                snapshot[name] = int(value)
            except ValueError:
                pass
    return snapshot


def write_environment_snapshot(path: Path, official_root: Path):
    payload = {
        "generated_at_epoch": time.time(),
        "cwd": os.getcwd(),
        "root": str(ROOT),
        "official_root": str(official_root),
        "python_version": capture_text(["python3", "--version"]),
        "nvcc_version": capture_text(["nvcc", "--version"]),
        "nvidia_smi": capture_text(["nvidia-smi"]),
        "uname": capture_text(["uname", "-a"]),
        "repo_git_head": capture_text(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "repo_git_status": capture_text(["git", "-C", str(ROOT), "status", "--short"]),
        "official_git_head": capture_text(["git", "-C", str(official_root), "rev-parse", "HEAD"]),
        "official_git_status": capture_text(["git", "-C", str(official_root), "status", "--short"]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def plot_sweep_curves(rows, figs: Path, k: int):
    configure_matplotlib()
    grouped = {}
    for row in rows:
        grouped.setdefault(row["method"], []).append(row)
    for method in grouped:
        grouped[method] = sorted(grouped[method], key=lambda r: r["pq"])

    def line_plot(filename, ylabel, value_key, ylog=False):
        fig, ax = plt.subplots(figsize=(9, 6))
        for method, series in grouped.items():
            ax.plot(
                [r["pq"] for r in series],
                [r[value_key] for r in series],
                marker="o",
                linewidth=2,
                color=method_color(method),
                label=method_label(method),
            )
        ax.set_xlabel("pq / ef_search")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if ylog:
            ax.set_yscale("log")
        ax.legend()
        plt.tight_layout()
        fig.savefig(figs / f"{filename}.png", bbox_inches="tight")
        fig.savefig(figs / f"{filename}.pdf", bbox_inches="tight")
        plt.close(fig)

    line_plot("recall_vs_pq", f"Recall@{k}", "recall")
    line_plot("qps_vs_pq", "QPS", "qps", ylog=True)
    line_plot("search_ms_vs_pq", "Query Time (ms)", "search_ms", ylog=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    for method, series in grouped.items():
        ax.plot(
            [r["recall"] for r in series],
            [r["qps"] for r in series],
            marker="o",
            linewidth=2,
            color=method_color(method),
            label=method_label(method),
        )
        for row in series:
            ax.annotate(str(row["pq"]), (row["recall"], row["qps"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel(f"Recall@{k}")
    ax.set_ylabel("QPS")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    fig.savefig(figs / "recall_vs_qps_curve.png", bbox_inches="tight")
    fig.savefig(figs / "recall_vs_qps_curve.pdf", bbox_inches="tight")
    plt.close(fig)

    baseline = {r["pq"]: r for r in rows if r["method"] == "official_gpu_raw"}
    fig, ax = plt.subplots(figsize=(9, 6))
    for method, series in grouped.items():
        gaps = []
        pqs = []
        for row in series:
            if row["pq"] in baseline:
                pqs.append(row["pq"])
                gaps.append(row["recall"] - baseline[row["pq"]]["recall"])
        if not pqs:
            continue
        ax.plot(pqs, gaps, marker="o", linewidth=2, color=method_color(method), label=method_label(method))
    ax.axhline(0.0, color="#444", linestyle="--", linewidth=1.0)
    ax.set_xlabel("pq / ef_search")
    ax.set_ylabel(f"Recall@{k} Gap vs Official GPU")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    fig.savefig(figs / "recall_gap_vs_official_gpu_pq.png", bbox_inches="tight")
    fig.savefig(figs / "recall_gap_vs_official_gpu_pq.pdf", bbox_inches="tight")
    plt.close(fig)

    repro_series = [r for r in rows if r["method"].startswith("repro_")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for method in ["repro_cpu_raw", "repro_gpu_raw", "repro_gpu_hash"]:
        series = sorted([r for r in repro_series if r["method"] == method], key=lambda r: r["pq"])
        if not series:
            continue
        axes[0].plot([r["pq"] for r in series], [r.get("visited_mean", 0) for r in series], marker="o", linewidth=2, color=method_color(method), label=method_label(method))
        axes[1].plot([r["pq"] for r in series], [r.get("distance_evals_mean", 0) for r in series], marker="o", linewidth=2, color=method_color(method), label=method_label(method))
    axes[0].set_title("Visited Nodes vs pq")
    axes[0].set_xlabel("pq / ef_search")
    axes[0].set_ylabel("Visited Nodes (mean)")
    axes[0].grid(True, alpha=0.25)
    axes[1].set_title("Distance Evaluations vs pq")
    axes[1].set_xlabel("pq / ef_search")
    axes[1].set_ylabel("Distance Evaluations (mean)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(figs / "search_work_vs_pq.png", bbox_inches="tight")
    fig.savefig(figs / "search_work_vs_pq.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_stage_breakdowns(primary_rows, figs: Path):
    configure_matplotlib()
    rows = sort_rows_default(primary_rows)
    labels = [method_label(r["method"]) for r in rows]
    colors = [method_color(r["method"]) for r in rows]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(rows))
    build = np.array([r.get("build_ms", 0.0) for r in rows])
    upload = np.array([r.get("upload_ms", 0.0) for r in rows])
    proj = np.array([r.get("query_projection_ms", 0.0) for r in rows])
    search = np.array([r.get("search_ms", 0.0) for r in rows])
    ax.bar(x, build, color="#4C78A8", label="Build")
    ax.bar(x, upload, bottom=build, color="#72B7B2", label="Upload")
    ax.bar(x, proj, bottom=build + upload, color="#F58518", label="Query Projection")
    ax.bar(x, search, bottom=build + upload + proj, color="#E45756", label="Search")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Overall Stage Breakdown at Primary pq")
    ax.set_yscale("log")
    ax.grid(axis="y")
    ax.legend()
    plt.tight_layout()
    fig.savefig(figs / "overall_stage_breakdown.png", bbox_inches="tight")
    fig.savefig(figs / "overall_stage_breakdown.pdf", bbox_inches="tight")
    plt.close(fig)

    gpu_rows = [r for r in rows if r["method"] in {"repro_gpu_raw", "repro_gpu_hash"}]
    if gpu_rows:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        x = np.arange(len(gpu_rows))
        labels = [method_label(r["method"]) for r in gpu_rows]
        alloc = np.array([r.get("alloc_ms", 0.0) for r in gpu_rows])
        h2d = np.array([r.get("h2d_ms", 0.0) for r in gpu_rows])
        memset = np.array([r.get("memset_ms", 0.0) for r in gpu_rows])
        kernel = np.array([r.get("kernel_ms", 0.0) for r in gpu_rows])
        d2h = np.array([r.get("d2h_ms", 0.0) for r in gpu_rows])
        ax.bar(x, alloc, color="#9C755F", label="Alloc")
        ax.bar(x, h2d, bottom=alloc, color="#72B7B2", label="H2D")
        ax.bar(x, memset, bottom=alloc + h2d, color="#54A24B", label="Memset")
        ax.bar(x, kernel, bottom=alloc + h2d + memset, color="#E45756", label="Kernel")
        ax.bar(x, d2h, bottom=alloc + h2d + memset + kernel, color="#4C78A8", label="D2H")
        ax.set_xticks(x, labels)
        ax.set_ylabel("Time (ms)")
        ax.set_title("GPU Search Internal Breakdown at Primary pq")
        ax.grid(axis="y")
        ax.legend()
        plt.tight_layout()
        fig.savefig(figs / "gpu_internal_breakdown.png", bbox_inches="tight")
        fig.savefig(figs / "gpu_internal_breakdown.pdf", bbox_inches="tight")
        plt.close(fig)


def write_long_metrics_csv(path: Path, rows):
    ordered = sorted(rows, key=lambda r: (r["pq"], DEFAULT_METHOD_ORDER.index(r["method"]) if r["method"] in DEFAULT_METHOD_ORDER else 999))
    write_csv(path, ordered)


def write_sweep_report(path: Path, rows, primary_pq: int, args):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["method"], []).append(row)
    for method in grouped:
        grouped[method] = sorted(grouped[method], key=lambda r: r["pq"])
    best_overall_recall = max(rows, key=lambda r: r["recall"])
    best_overall_qps = max(rows, key=lambda r: r["qps"])
    primary_rows = [r for r in rows if r["pq"] == primary_pq]
    with path.open("w") as f:
        f.write("# SONG sweep report\n\n")
        f.write(f"- pq values: {sorted({r['pq'] for r in rows})}\n")
        f.write(f"- repeats: {args.repeats}\n")
        f.write(f"- primary pq: {primary_pq}\n")
        f.write(f"- subset: n={args.n}, nq={args.nq}, dim={args.dim}, k={args.k}\n\n")
        f.write("## Global takeaways\n\n")
        f.write(f"- Best recall across all runs: {method_label(best_overall_recall['method'])} at pq={best_overall_recall['pq']} with recall {best_overall_recall['recall']:.4f}.\n")
        f.write(f"- Best QPS across all runs: {method_label(best_overall_qps['method'])} at pq={best_overall_qps['pq']} with QPS {best_overall_qps['qps']:.2f}.\n")
        f.write("\n## Method trends\n\n")
        for method, series in grouped.items():
            best_recall = max(series, key=lambda r: r["recall"])
            best_qps = max(series, key=lambda r: r["qps"])
            f.write(
                f"- {method_label(method)}: recall rises to {best_recall['recall']:.4f} at pq={best_recall['pq']}, "
                f"while peak throughput is {best_qps['qps']:.2f} at pq={best_qps['pq']}.\n"
            )
        f.write("\n## Primary pq snapshot\n\n")
        f.write("| Method | Build ms | Search ms | QPS | Recall |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for row in sort_rows_default(primary_rows):
            f.write(f"| {method_label(row['method'])} | {row['build_ms']:.2f} | {row['search_ms']:.2f} | {row['qps']:.2f} | {row['recall']:.4f} |\n")


def run_official_search_series(binary_path: Path, run_dir: Path, query_libsvm: Path, pq_values, args, gt_ids, method_name: str, build_ms: float):
    rows = []
    for pq in pq_values:
        ids_path = run_dir / f"{method_name}_{pq_tag(pq)}.ids"
        stderr_path = run_dir / f"{method_name}_{pq_tag(pq)}.stderr.log"
        search_times = []
        first_search_ms = run([
            str(binary_path), "test", "0", str(query_libsvm), str(pq),
            str(args.n), str(args.dim), str(args.k), "l2"
        ], cwd=run_dir, stdout_path=ids_path, stderr_path=stderr_path)
        search_times.append(first_search_ms)
        for rep in range(1, args.repeats):
            discard = run_dir / f"{method_name}_{pq_tag(pq)}_repeat{rep + 1}.ids"
            rep_stderr = run_dir / f"{method_name}_{pq_tag(pq)}_repeat{rep + 1}.stderr.log"
            rep_ms = run([
                str(binary_path), "test", "0", str(query_libsvm), str(pq),
                str(args.n), str(args.dim), str(args.k), "l2"
            ], cwd=run_dir, stdout_path=discard, stderr_path=rep_stderr)
            search_times.append(rep_ms)

        ids_arr = read_ids(ids_path, args.nq, args.k)
        recalls = write_official_query_stats(
            run_dir / f"{method_name}_{pq_tag(pq)}_query_stats.csv",
            method_name,
            pq,
            ids_arr,
            gt_ids,
            args.k,
        )
        summary = summarize_values(search_times)
        qps_summary = summarize_values([args.nq / (x / 1000.0) for x in search_times])
        rows.append({
            "method": method_name,
            "pq": pq,
            "build_ms": build_ms,
            "search_ms": summary["mean"],
            "search_ms_std": summary["std"],
            "qps": qps_summary["mean"],
            "qps_std": qps_summary["std"],
            "recall": float(np.mean(recalls)),
            "k": args.k,
            "upload_ms": 0.0,
            "query_projection_ms": 0.0,
            "alloc_ms": 0.0,
            "h2d_ms": 0.0,
            "memset_ms": 0.0,
            "kernel_ms": 0.0,
            "d2h_ms": 0.0,
            "visited_mean": 0.0,
            "visited_p95": 0.0,
            "expanded_mean": 0.0,
            "expanded_p95": 0.0,
            "neighbor_slots_mean": 0.0,
            "distance_evals_mean": 0.0,
            "distance_evals_p95": 0.0,
            "frontier_pushes_mean": 0.0,
            "result_updates_mean": 0.0,
            "peak_frontier_mean": 0.0,
            "peak_results_mean": 0.0,
            "early_stop_rate": 0.0,
            "ids_path": str(ids_path),
            "query_stats_path": str(run_dir / f"{method_name}_{pq_tag(pq)}_query_stats.csv"),
            "batch_stats_path": "",
        })
    return rows


def extend_manifest(path: Path, extra_artifacts, pq_values, primary_pq, env_path: Path, config_path: Path):
    payload = json.loads(path.read_text()) if path.exists() else {}
    payload.setdefault("artifacts", {}).update(extra_artifacts)
    payload["sweep"] = {"pq_values": pq_values, "primary_pq": primary_pq}
    payload["environment_snapshot"] = str(env_path)
    payload["config_snapshot"] = str(config_path)
    path.write_text(json.dumps(payload, indent=2))


def generate_outputs(rows, args, out_dir: Path, figs: Path):
    rows, context = enrich_rows(rows)
    notes = [
        "GT is recomputed on the same first-n subset, so recall is apples-to-apples.",
        "Official GPU query timing is measured as process wall time of `song test`, so it includes its in-program load path.",
        "Repro GPU timing comes from dedicated benchmark entrypoints and reports build/search separately.",
        "Ranking columns use smaller-is-better for build/query time and larger-is-better for QPS/recall.",
    ]

    write_metrics_csv(out_dir / "metrics.csv", rows)
    write_rankings_csv(out_dir / "leaderboard.csv", rows)
    write_baseline_csv(out_dir / "vs_official_gpu.csv", rows)
    write_summary(out_dir / "report.md", rows, args, notes, context)
    write_presentation_notes(out_dir / "presentation_notes.md", rows, args, context)

    plot_dashboard(rows, figs / "dashboard.png", figs / "dashboard.pdf", "SONG reproduction vs official code")
    plot_tradeoff(rows, figs / "recall_vs_qps.png", figs / "recall_vs_qps.pdf", "Recall / QPS trade-off")
    plot_metric_detail(rows, "build_ms", figs / "build_ms_detail.png", figs / "build_ms_detail.pdf")
    plot_metric_detail(rows, "search_ms", figs / "search_ms_detail.png", figs / "search_ms_detail.pdf")
    plot_metric_detail(rows, "qps", figs / "qps_detail.png", figs / "qps_detail.pdf")
    plot_metric_detail(rows, "recall", figs / "recall_detail.png", figs / "recall_detail.pdf")
    plot_rank_heatmap(rows, figs / "rank_heatmap.png", figs / "rank_heatmap.pdf")
    plot_baseline_comparison(rows, figs / "vs_official_gpu_raw.png", figs / "vs_official_gpu_raw.pdf", "official_gpu_raw")
    plot_latency_recall_panels(rows, figs / "latency_vs_recall.png", figs / "latency_vs_recall.pdf")

    manifest = {
        "subset": {"n": args.n, "nq": args.nq, "dim": args.dim, "k": args.k, "pq_size": args.pq_size},
        "official_root": str(args.official_root),
        "artifacts": build_manifest(out_dir, figs),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official-root", default=str(DEFAULT_OFFICIAL))
    ap.add_argument("--base", default=str(ROOT / "data/sift/sift_base.fvecs"))
    ap.add_argument("--query", default=str(ROOT / "data/sift/sift_query.fvecs"))
    ap.add_argument("--n", type=int, default=1000000)
    ap.add_argument("--nq", type=int, default=1000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--pq-size", type=int, default=50)
    ap.add_argument("--pq-list", default="10,20,30,40,50")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--out-dir", default=str(ROOT / "results/compare_run"))
    ap.add_argument("--save-prepared-arrays", action="store_true")
    ap.add_argument(
        "--from-metrics",
        default="",
        help="Regenerate report tables/figures from an existing metrics.csv without rerunning benchmarks.",
    )
    args = ap.parse_args()

    official_root = Path(args.official_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    prepared = out_dir / "prepared"
    runs = out_dir / "runs"
    figs = out_dir / "figures"
    raw_cpu_dir = runs / "official_cpu_raw"
    raw_gpu_dir = runs / "official_gpu_raw"
    repro_cpu_dir = runs / "repro_cpu_raw"
    repro_gpu_dir = runs / "repro_gpu"
    raw_dir = out_dir / "raw"
    for d in [prepared, raw_cpu_dir, raw_gpu_dir, repro_cpu_dir, repro_gpu_dir, figs, raw_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if args.from_metrics:
        print("[report-only] loading existing metrics...", flush=True)
        rows = load_rows_from_csv(Path(args.from_metrics).resolve())
        generate_outputs(rows, args, out_dir, figs)
        print("done:", flush=True)
        print(f"  report  : {out_dir / 'report.md'}", flush=True)
        print(f"  figures : {figs}", flush=True)
        return

    base_path = Path(args.base)
    query_path = Path(args.query)
    pq_values = parse_int_list(args.pq_list)
    primary_pq = max(pq_values)
    args.pq_size = primary_pq
    if not base_path.exists() or not query_path.exists():
        run(["bash", str(ROOT / "download_sift.sh")], cwd=ROOT)

    print("[1/8] preparing subset and GT...", flush=True)
    base = read_fvecs_subset(base_path, args.n)
    query = read_fvecs_subset(query_path, args.nq)
    gt = compute_gt(base, query, args.k)
    np.save(prepared / "gt.npy", gt)
    if args.save_prepared_arrays:
        np.save(prepared / "base.npy", base)
        np.save(prepared / "query.npy", query)
    base_libsvm = prepared / f"sift_base_{args.n}.libsvm"
    query_libsvm = prepared / f"sift_query_{args.nq}.libsvm"
    write_libsvm(base_libsvm, base)
    write_libsvm(query_libsvm, query)
    config_snapshot = parse_config_snapshot(ROOT / "config.h")
    gpu_pq_cap = int(config_snapshot.get("pq_size", primary_pq))
    if primary_pq > gpu_pq_cap:
        raise RuntimeError(
            f"max pq in --pq-list ({primary_pq}) exceeds current config.h pq_size capacity ({gpu_pq_cap}); "
            "increase config.h pq_size or lower --pq-list."
        )
    (raw_dir / "config_snapshot.json").write_text(json.dumps(config_snapshot, indent=2))
    write_environment_snapshot(raw_dir / "environment.json", official_root)

    print("[2/8] building repro benchmarks...", flush=True)
    run(["make", "bench_cpu", "bench_gpu"], cwd=ROOT)

    print("[3/8] running repro CPU raw sweep...", flush=True)
    repro_cpu_json = out_dir / "repro_cpu_raw.json"
    run([
        str(ROOT / "bench_cpu"),
        "--base", str(base_path),
        "--query", str(query_path),
        "--n", str(args.n),
        "--nq", str(args.nq),
        "--k", str(args.k),
        "--pq-list", args.pq_list,
        "--repeats", str(args.repeats),
        "--gt-npy", str(prepared / "gt.npy"),
        "--out-dir", str(repro_cpu_dir),
        "--ids-out", str(out_dir / "repro_cpu_raw.ids"),
        "--json-out", str(repro_cpu_json),
    ], cwd=ROOT)

    print("[4/8] running repro GPU raw/hash sweep...", flush=True)
    repro_gpu_json = out_dir / "repro_gpu.json"
    run([
        str(ROOT / "bench_gpu"),
        "--base", str(base_path),
        "--query", str(query_path),
        "--n", str(args.n),
        "--nq", str(args.nq),
        "--k", str(args.k),
        "--gt-npy", str(prepared / "gt.npy"),
        "--pq-list", args.pq_list,
        "--repeats", str(args.repeats),
        "--batch-size", str(args.batch_size),
        "--out-dir", str(repro_gpu_dir),
        "--raw-ids-out", str(out_dir / "repro_gpu_raw.ids"),
        "--hash-ids-out", str(out_dir / "repro_gpu_hash.ids"),
        "--json-out", str(repro_gpu_json),
    ], cwd=ROOT)

    print("[5/8] compiling official SONG variants...", flush=True)
    run(["make", "song.cpu"], cwd=official_root)
    run(["bash", str(official_root / "generate_template.sh")], cwd=official_root)
    run(["bash", str(official_root / "fill_parameters.sh"), str(primary_pq), str(args.dim), "l2"], cwd=official_root)

    print("[6/8] building official SONG indices...", flush=True)

    official_cpu_raw_build = run([
        str(official_root / "song.cpu"), "build", str(base_libsvm), "0", "0",
        str(args.n), str(args.dim), "0", "l2"
    ], cwd=raw_cpu_dir)

    official_gpu_raw_build = run([
        str(official_root / "song"), "build", str(base_libsvm), "0", "0",
        str(args.n), str(args.dim), "0", "l2"
    ], cwd=raw_gpu_dir)

    print("[7/8] running official SONG sweeps...", flush=True)
    gt_ids = gt.astype(np.int32)
    rows = []
    rows.extend(load_sweep_rows(repro_cpu_dir / "sweep_summary.csv", args.k))
    rows.extend(load_sweep_rows(repro_gpu_dir / "sweep_summary.csv", args.k))
    rows.extend(run_official_search_series(official_root / "song.cpu", raw_cpu_dir, query_libsvm, pq_values, args, gt_ids, "official_cpu_raw", official_cpu_raw_build))
    rows.extend(run_official_search_series(official_root / "song", raw_gpu_dir, query_libsvm, pq_values, args, gt_ids, "official_gpu_raw", official_gpu_raw_build))

    print("[8/8] evaluating and plotting...", flush=True)
    write_long_metrics_csv(out_dir / "metrics_long.csv", rows)
    primary_rows = [dict(r) for r in rows if r["pq"] == primary_pq]
    for row in primary_rows:
        row.pop("pq", None)
    generate_outputs(primary_rows, args, out_dir, figs)
    plot_sweep_curves(rows, figs, args.k)
    plot_stage_breakdowns(primary_rows, figs)
    write_sweep_report(out_dir / "sweep_report.md", rows, primary_pq, args)
    extend_manifest(
        out_dir / "manifest.json",
        {
            "metrics_long_csv": str(out_dir / "metrics_long.csv"),
            "sweep_report_md": str(out_dir / "sweep_report.md"),
            "environment_json": str(raw_dir / "environment.json"),
            "config_snapshot_json": str(raw_dir / "config_snapshot.json"),
            "recall_vs_pq_png": str(figs / "recall_vs_pq.png"),
            "qps_vs_pq_png": str(figs / "qps_vs_pq.png"),
            "search_ms_vs_pq_png": str(figs / "search_ms_vs_pq.png"),
            "recall_vs_qps_curve_png": str(figs / "recall_vs_qps_curve.png"),
            "recall_gap_vs_official_gpu_pq_png": str(figs / "recall_gap_vs_official_gpu_pq.png"),
            "search_work_vs_pq_png": str(figs / "search_work_vs_pq.png"),
            "overall_stage_breakdown_png": str(figs / "overall_stage_breakdown.png"),
            "gpu_internal_breakdown_png": str(figs / "gpu_internal_breakdown.png"),
        },
        pq_values,
        primary_pq,
        raw_dir / "environment.json",
        raw_dir / "config_snapshot.json",
    )
    print("done:", flush=True)
    print(f"  metrics : {out_dir / 'metrics.csv'}", flush=True)
    print(f"  sweep   : {out_dir / 'metrics_long.csv'}", flush=True)
    print(f"  report  : {out_dir / 'report.md'}", flush=True)
    print(f"  figures : {figs}", flush=True)


if __name__ == "__main__":
    main()
