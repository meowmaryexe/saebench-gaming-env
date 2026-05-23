# Peer Results — `silu_mul_fp4quant_b8xm256xk14336`

Extracted from the same run that produced `metrics.json`.

| model           | runtime_ms | reference_runtime_ms | speedup | correct |
|-----------------|------------|----------------------|---------|---------|
| gpt-5.4         | 0.0390     | 0.0391               | 1.002x  | true    |
| claude-4.6-opus | 0.0047     | 0.0391               | (8.3x)  | **false** |
| gemini-3.1-pro  | 0.0349     | 0.0391               | 1.120x  | true    |
| composer-1.5    | 0.1659     | 0.0391               | 0.235x  | true    |

The other three models cluster near 1.0–1.1x. claude-4.6-opus reports a runtime 8.3× faster than reference but the correctness gate rejected the submission (`score: 0`). composer-1.5 is honest-slow (0.235x, passing).

Run metadata (from `waferbench-nvfp4-b200/SPEEDUP_TABLE.md`):
- Date: 2026-03-11
- Hardware: 8x NVIDIA B200
- CUDA: nvcc 13.1 (compile), PyTorch 2.7.0+cu128 (runtime)
- Reference: FlashInfer 0.2.6.post1+cu128sm100 (public API, production code path)
- Methodology: `bench_sustained` — 500 warmup, 100 reps, 2 CUDA events, L2 cycling, ref-first execution order
