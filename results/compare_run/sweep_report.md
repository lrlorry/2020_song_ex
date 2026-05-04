# SONG sweep report

- pq values: [10, 20, 30, 40, 50]
- repeats: 3
- primary pq: 50
- subset: n=1000000, nq=1000, dim=128, k=10

## Global takeaways

- Best recall across all runs: Official GPU Raw (wall) at pq=10 with recall 0.8863.
- Best QPS across all runs: Repro GPU Hash at pq=10 with QPS 486597.00.

## Method trends

- Repro CPU Raw: recall rises to 0.8848 at pq=50, while peak throughput is 9074.53 at pq=10.
- Repro GPU Raw: recall rises to 0.8838 at pq=50, while peak throughput is 332754.00 at pq=10.
- Repro GPU Hash: recall rises to 0.1963 at pq=50, while peak throughput is 486597.00 at pq=10.
- Official CPU Raw (wall): recall rises to 0.8861 at pq=50, while peak throughput is 1366.90 at pq=10.
- Official GPU Raw (wall): recall rises to 0.8863 at pq=10, while peak throughput is 1134.08 at pq=50.

## Primary pq snapshot

| Method | Build ms | Search ms | QPS | Recall |
|---|---:|---:|---:|---:|
| Repro CPU Raw | 775734.00 | 312.08 | 3204.71 | 0.8848 |
| Repro GPU Raw | 785934.00 | 8.87 | 112736.00 | 0.8838 |
| Repro GPU Hash | 501106.00 | 10.87 | 92023.70 | 0.1963 |
| Official CPU Raw (wall) | 723827.79 | 999.44 | 1004.45 | 0.8861 |
| Official GPU Raw (wall) | 842451.98 | 881.82 | 1134.08 | 0.8863 |
