#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import shlex
import struct
import subprocess
import sys
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
    "official_cpu_raw": "Official CPU Raw",
    "repro_gpu_raw": "Repro GPU Raw",
    "official_gpu_raw": "Official GPU Raw",
    "repro_gpu_hash": "Repro GPU Hash",
}


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
    cmd_str = ' '.join(map(shlex.quote, cmd))
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
    with path.open() as f:
        for i, line in enumerate(f):
            if i >= nq:
                break
            vals = [int(x) for x in line.strip().split()[:k]]
            out[i, : len(vals)] = vals
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


def plot_dashboard(rows, out_png: Path, out_pdf: Path, title: str):
    methods = [r["method"] for r in rows]
    labels = [LABELS[m] for m in methods]
    colors = [COLORS[m] for m in methods]
    build_ms = [r.get("build_ms", math.nan) for r in rows]
    search_ms = [r.get("search_ms", math.nan) for r in rows]
    qps = [r.get("qps", math.nan) for r in rows]
    recall = [r.get("recall", math.nan) for r in rows]

    plt.rcParams.update({
        "figure.dpi": 160,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.25,
    })

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
            ax.text(xi, yi, f"{yi:.2f}" if yi < 100 else f"{yi:.0f}", ha="center", va="bottom", fontsize=9)

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
    plt.rcParams.update({"figure.dpi": 160, "font.size": 11})
    fig, ax = plt.subplots(figsize=(9, 7))
    for r in rows:
        m = r["method"]
        marker = "s" if "gpu" in m else "o"
        ax.scatter(r["recall"], r["qps"], s=140, color=COLORS[m], marker=marker, edgecolor="#222", linewidth=0.8)
        ax.annotate(LABELS[m], (r["recall"], r["qps"]), xytext=(8, 8), textcoords="offset points", fontsize=10)
    ax.set_xlabel(f"Recall@{rows[0]['k']}")
    ax.set_ylabel("QPS")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.set_title(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def write_summary(path: Path, rows, args, notes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# SONG comparison report\n\n")
        f.write(f"- subset: n={args.n}, nq={args.nq}, dim={args.dim}, k={args.k}\n")
        f.write(f"- pq_size: {args.pq_size}\n")
        f.write(f"- official root: `{args.official_root}`\n\n")
        f.write("## Metrics\n\n")
        f.write("| Method | Build ms | Search ms | QPS | Recall |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {LABELS[r['method']]} | {r['build_ms']:.2f} | {r['search_ms']:.2f} | {r['qps']:.2f} | {r['recall']:.4f} |\n")
        f.write("\n## Notes\n\n")
        for note in notes:
            f.write(f"- {note}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official-root", default=str(DEFAULT_OFFICIAL))
    ap.add_argument("--base", default=str(ROOT / "data/sift/sift_base.fvecs"))
    ap.add_argument("--query", default=str(ROOT / "data/sift/sift_query.fvecs"))
    ap.add_argument("--n", type=int, default=1000000)
    ap.add_argument("--nq", type=int, default=1000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--pq-size", type=int, default=50)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--out-dir", default=str(ROOT / "results/compare_run"))
    args = ap.parse_args()

    official_root = Path(args.official_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    prepared = out_dir / "prepared"
    runs = out_dir / "runs"
    figs = out_dir / "figures"
    raw_cpu_dir = runs / "official_cpu_raw"
    raw_gpu_dir = runs / "official_gpu_raw"
    for d in [prepared, raw_cpu_dir, raw_gpu_dir, figs]:
        d.mkdir(parents=True, exist_ok=True)

    base_path = Path(args.base)
    query_path = Path(args.query)
    if not base_path.exists() or not query_path.exists():
        run(["bash", str(ROOT / "download_sift.sh")], cwd=ROOT)

    print("[1/7] preparing subset and GT...", flush=True)
    base = read_fvecs_subset(base_path, args.n)
    query = read_fvecs_subset(query_path, args.nq)
    gt = compute_gt(base, query, args.k)
    np.save(prepared / "gt.npy", gt)
    np.save(prepared / "base.npy", base)
    np.save(prepared / "query.npy", query)
    base_libsvm = prepared / f"sift_base_{args.n}.libsvm"
    query_libsvm = prepared / f"sift_query_{args.nq}.libsvm"
    write_libsvm(base_libsvm, base)
    write_libsvm(query_libsvm, query)

    print("[2/7] building repro benchmarks...", flush=True)
    run(["make", "bench_cpu", "bench_gpu"], cwd=ROOT)

    print("[3/7] running repro CPU raw...", flush=True)
    repro_cpu_json = out_dir / "repro_cpu_raw.json"
    repro_cpu_ids = out_dir / "repro_cpu_raw.ids"
    run([
        str(ROOT / "bench_cpu"),
        "--base", str(base_path),
        "--query", str(query_path),
        "--n", str(args.n),
        "--nq", str(args.nq),
        "--k", str(args.k),
        "--pq-size", str(args.pq_size),
        "--ids-out", str(repro_cpu_ids),
        "--json-out", str(repro_cpu_json),
    ], cwd=ROOT)

    print("[4/7] running repro GPU raw/hash...", flush=True)
    repro_gpu_json = out_dir / "repro_gpu.json"
    repro_gpu_raw_ids = out_dir / "repro_gpu_raw.ids"
    repro_gpu_hash_ids = out_dir / "repro_gpu_hash.ids"
    run([
        str(ROOT / "bench_gpu"),
        "--base", str(base_path),
        "--query", str(query_path),
        "--n", str(args.n),
        "--nq", str(args.nq),
        "--k", str(args.k),
        "--raw-ids-out", str(repro_gpu_raw_ids),
        "--hash-ids-out", str(repro_gpu_hash_ids),
        "--json-out", str(repro_gpu_json),
    ], cwd=ROOT)

    print("[5/7] compiling official SONG variants...", flush=True)
    run(["make", "song.cpu"], cwd=official_root)
    run(["bash", str(official_root / "generate_template.sh")], cwd=official_root)
    run(["bash", str(official_root / "fill_parameters.sh"), str(args.pq_size), str(args.dim), "l2"], cwd=official_root)

    print("[6/7] running official SONG variants...", flush=True)
    official_cpu_raw_ids = out_dir / "official_cpu_raw.ids"
    official_gpu_raw_ids = out_dir / "official_gpu_raw.ids"

    official_cpu_raw_build = run([
        str(official_root / "song.cpu"), "build", str(base_libsvm), "0", "0",
        str(args.n), str(args.dim), "0", "l2"
    ], cwd=raw_cpu_dir)
    official_cpu_raw_search = run([
        str(official_root / "song.cpu"), "test", "0", str(query_libsvm), str(args.pq_size),
        str(args.n), str(args.dim), str(args.k), "l2"
    ], cwd=raw_cpu_dir, stdout_path=official_cpu_raw_ids, stderr_path=raw_cpu_dir / "test.stderr.log")

    official_gpu_raw_build = run([
        str(official_root / "song"), "build", str(base_libsvm), "0", "0",
        str(args.n), str(args.dim), "0", "l2"
    ], cwd=raw_gpu_dir)
    official_gpu_raw_search = run([
        str(official_root / "song"), "test", "0", str(query_libsvm), str(args.pq_size),
        str(args.n), str(args.dim), str(args.k), "l2"
    ], cwd=raw_gpu_dir, stdout_path=official_gpu_raw_ids, stderr_path=raw_gpu_dir / "test.stderr.log")

    print("[7/7] evaluating and plotting...", flush=True)
    gt_ids = gt.astype(np.int32)
    rows = []

    repro_cpu_meta = json.loads((repro_cpu_json).read_text())
    repro_cpu_ids_arr = read_ids(repro_cpu_ids, args.nq, args.k)
    rows.append({
        "method": "repro_cpu_raw",
        "build_ms": float(repro_cpu_meta["build_ms"]),
        "search_ms": float(repro_cpu_meta["search_ms"]),
        "qps": float(repro_cpu_meta["qps"]),
        "recall": float(recall_at_k(repro_cpu_ids_arr, gt_ids, args.k)),
        "k": args.k,
    })

    repro_gpu_meta = json.loads((repro_gpu_json).read_text())
    repro_gpu_raw_arr = read_ids(repro_gpu_raw_ids, args.nq, args.k)
    repro_gpu_hash_arr = read_ids(repro_gpu_hash_ids, args.nq, args.k)
    rows.append({
        "method": "repro_gpu_raw",
        "build_ms": float(repro_gpu_meta["raw_build_ms"]),
        "search_ms": float(repro_gpu_meta["raw_search_ms"]),
        "qps": float(repro_gpu_meta["raw_qps"]),
        "recall": float(recall_at_k(repro_gpu_raw_arr, gt_ids, args.k)),
        "k": args.k,
    })
    rows.append({
        "method": "repro_gpu_hash",
        "build_ms": float(repro_gpu_meta["hash_build_ms"]),
        "search_ms": float(repro_gpu_meta["hash_search_ms"]),
        "qps": float(repro_gpu_meta["hash_qps"]),
        "recall": float(recall_at_k(repro_gpu_hash_arr, gt_ids, args.k)),
        "k": args.k,
    })

    official_cpu_raw_arr = read_ids(official_cpu_raw_ids, args.nq, args.k)
    rows.append({
        "method": "official_cpu_raw",
        "build_ms": official_cpu_raw_build,
        "search_ms": official_cpu_raw_search,
        "qps": args.nq / (official_cpu_raw_search / 1000.0),
        "recall": float(recall_at_k(official_cpu_raw_arr, gt_ids, args.k)),
        "k": args.k,
    })

    official_gpu_raw_arr = read_ids(official_gpu_raw_ids, args.nq, args.k)
    rows.append({
        "method": "official_gpu_raw",
        "build_ms": official_gpu_raw_build,
        "search_ms": official_gpu_raw_search,
        "qps": args.nq / (official_gpu_raw_search / 1000.0),
        "recall": float(recall_at_k(official_gpu_raw_arr, gt_ids, args.k)),
        "k": args.k,
    })

    rows = sorted(rows, key=lambda r: ["official" in r["method"], "hash" in r["method"], "gpu" in r["method"]])
    write_csv(out_dir / "metrics.csv", rows)

    notes = [
        "GT is recomputed on the same first-n subset, so recall is apples-to-apples.",
        "Official GPU query timing is measured as process wall time of `song test`, so it includes its in-program load path.",
        "Repro GPU timing comes from dedicated benchmark entrypoints and reports build/search separately.",
    ]
    write_summary(out_dir / "report.md", rows, args, notes)
    plot_dashboard(rows, figs / "dashboard.png", figs / "dashboard.pdf", "SONG reproduction vs official code")
    plot_tradeoff(rows, figs / "recall_vs_qps.png", figs / "recall_vs_qps.pdf", "Recall / QPS trade-off")

    manifest = {
        "subset": {"n": args.n, "nq": args.nq, "dim": args.dim, "k": args.k, "pq_size": args.pq_size},
        "official_root": str(official_root),
        "artifacts": {
            "metrics_csv": str(out_dir / "metrics.csv"),
            "report_md": str(out_dir / "report.md"),
            "dashboard_png": str(figs / "dashboard.png"),
            "tradeoff_png": str(figs / "recall_vs_qps.png"),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("done:", flush=True)
    print(f"  metrics : {out_dir / 'metrics.csv'}", flush=True)
    print(f"  report  : {out_dir / 'report.md'}", flush=True)
    print(f"  figures : {figs}", flush=True)


if __name__ == "__main__":
    main()
