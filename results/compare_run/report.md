# SONG comparison report

## Experiment setup

- subset: n=1000000, nq=1000, dim=128, k=10
- pq_size: 50
- official root: `/root/song`

## Main takeaways

- Highest recall@10: Official GPU Raw (wall) at 0.8863.
- Highest QPS: Repro GPU Raw at 112,736.00, which is 99.41x Official GPU Raw (wall).
- Fastest build: Repro GPU Hash at 501,106.00 ms.
- Fastest query latency: Repro GPU Raw at 8.87 ms.
- Best reproduction recall: Repro CPU Raw at 0.8848, gap 0.0015 from Official GPU Raw (wall).
- Fastest reproduction query path: Repro GPU Raw at 8.87 ms, 99.41x faster than Official GPU Raw (wall).

## Detailed metrics

| Method | Build ms | Build rank | Search ms | Search rank | QPS | QPS rank | Recall | Recall rank | Mean rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Repro GPU Hash | 501106.00 | 1 | 10.87 | 2 | 92023.70 | 2 | 0.1963 | 5 | 2.50 |
| Repro GPU Raw | 785934.00 | 4 | 8.87 | 1 | 112736.00 | 1 | 0.8838 | 4 | 2.50 |
| Repro CPU Raw | 775734.00 | 3 | 312.08 | 3 | 3204.71 | 3 | 0.8848 | 3 | 3.00 |
| Official CPU Raw (wall) | 723827.79 | 2 | 999.44 | 5 | 1004.45 | 5 | 0.8861 | 2 | 3.50 |
| Official GPU Raw (wall) | 842451.98 | 5 | 881.82 | 4 | 1134.08 | 4 | 0.8863 | 1 | 3.50 |

## Relative to Official GPU Raw (wall)

| Method | Build speedup | Query speedup | QPS ratio | Recall gap |
|---|---:|---:|---:|---:|
| Repro CPU Raw | 1.09x | 2.83x | 2.83x | -0.0015 |
| Repro GPU Raw | 1.07x | 99.41x | 99.41x | -0.0025 |
| Repro GPU Hash | 1.68x | 81.11x | 81.14x | -0.6900 |
| Official CPU Raw (wall) | 1.16x | 0.88x | 0.89x | -0.0002 |
| Official GPU Raw (wall) | 1.00x | 1.00x | 1.00x | +0.0000 |

## Method-by-method notes

- Repro CPU Raw: build 775734.00 ms, query 312.08 ms, QPS 3204.71, recall 0.8848; relative to Official GPU Raw (wall), build 1.09x, query 2.83x, recall gap -0.0015.
- Repro GPU Raw: build 785934.00 ms, query 8.87 ms, QPS 112736.00, recall 0.8838; relative to Official GPU Raw (wall), build 1.07x, query 99.41x, recall gap -0.0025.
- Repro GPU Hash: build 501106.00 ms, query 10.87 ms, QPS 92023.70, recall 0.1963; relative to Official GPU Raw (wall), build 1.68x, query 81.11x, recall gap -0.6900.
- Official CPU Raw (wall): build 723827.79 ms, query 999.44 ms, QPS 1004.45, recall 0.8861; relative to Official GPU Raw (wall), build 1.16x, query 0.88x, recall gap -0.0002.
- Official GPU Raw (wall): build 842451.98 ms, query 881.82 ms, QPS 1134.08, recall 0.8863; relative to Official GPU Raw (wall), build 1.00x, query 1.00x, recall gap +0.0000.

## Useful output files

- `metrics.csv`: raw metric table
- `leaderboard.csv`: ranked table for report screenshots
- `vs_official_gpu.csv`: direct comparison against the official GPU baseline
- `presentation_notes.md`: short talking points for an oral report
- `figures/`: detailed figures, including per-metric plots and ranking heatmap

## Notes

- GT is recomputed on the same first-n subset, so recall is apples-to-apples.
- Official GPU query timing is measured as process wall time of `song test`, so it includes its in-program load path.
- Repro GPU timing comes from dedicated benchmark entrypoints and reports build/search separately.
- Ranking columns use smaller-is-better for build/query time and larger-is-better for QPS/recall.
