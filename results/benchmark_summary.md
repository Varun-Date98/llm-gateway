# Benchmark Summary

| Sweep | RPS | Requests | Successes | P99 latency (s) | P99 TTFT (s) | Cost / 1M tokens |
|---|---:|---:|---:|---:|---:|---:|
| short_streaming | 1.0000 | 5 | 5 | 0.0647 | 0.0646 | 0.0722 |
| short_streaming | 2.0000 | 10 | 10 | 0.0643 | 0.0642 | 0.0722 |
| short_streaming | 4.0000 | 20 | 20 | 0.0505 | 0.0503 | 0.0722 |
| short_streaming | 8.0000 | 40 | 40 | 0.0631 | 0.0630 | 0.0722 |
| short_streaming | 12.0000 | 60 | 60 | 0.0707 | 0.0706 | 0.0722 |
| shared_prefix_streaming | 1.0000 | 5 | 5 | 0.0659 | 0.0657 | 0.0554 |
| shared_prefix_streaming | 2.0000 | 10 | 10 | 0.0673 | 0.0669 | 0.0554 |
| shared_prefix_streaming | 4.0000 | 20 | 20 | 0.0840 | 0.0837 | 0.0554 |
| shared_prefix_streaming | 8.0000 | 40 | 40 | 0.2626 | 0.2622 | 0.0554 |
| shared_prefix_streaming | 12.0000 | 60 | 60 | 0.0729 | 0.0728 | 0.0554 |
| mixed_streaming | 1.0000 | 5 | 5 | 0.1874 | 0.1872 | 0.2443 |
| mixed_streaming | 2.0000 | 10 | 10 | 0.1886 | 0.1884 | 0.2443 |
| mixed_streaming | 4.0000 | 20 | 20 | 0.1909 | 0.1908 | 0.2443 |
| mixed_streaming | 8.0000 | 40 | 40 | 0.1906 | 0.1905 | 0.2443 |
| mixed_streaming | 12.0000 | 60 | 60 | 0.1793 | 0.1792 | 0.2443 |
