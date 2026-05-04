# Presentation notes

## One-paragraph summary

On the SIFT subset with n=1000000, nq=1000, dim=128, and Recall@10, the strongest accuracy came from Official GPU Raw (wall) with recall 0.8863. The highest throughput came from Repro GPU Raw with 112736.00 QPS, but its recall was 0.8838. Among the reproduction variants, the best recall was 0.8848, which is 0.0015 below the official GPU baseline.

## Suggested speaking points

- Highest recall@10: Official GPU Raw (wall) at 0.8863.
- Highest QPS: Repro GPU Raw at 112,736.00, which is 99.41x Official GPU Raw (wall).
- Fastest build: Repro GPU Hash at 501,106.00 ms.
- Fastest query latency: Repro GPU Raw at 8.87 ms.
- Best reproduction recall: Repro CPU Raw at 0.8848, gap 0.0015 from Official GPU Raw (wall).
- Fastest reproduction query path: Repro GPU Raw at 8.87 ms, 99.41x faster than Official GPU Raw (wall).

## Suggested report framing

- Start with recall first, because the reproduction/official gap is the main story in these numbers.
- Then explain the speed story separately: some reproduction paths are much faster, but they trade away substantial recall.
- Use `leaderboard.csv` and `figures/rank_heatmap.png` as compact summary slides.
