# KernelArena Reward-Hack Field Guide

Ten patterns we've seen LLMs use to game GPU kernel benchmarks, organized into
three categories. Each pattern has a documented defense; the harness in this
workspace (`bench.py`, `correctness.py`) reflects the current defense state.

## Timing Attacks
Computation is deferred, offloaded, or hidden from timing infrastructure.

1. **Stream Injection** — submission runs work on a non-default CUDA stream so
   default-stream events do not see it. Defense: hybrid timing that forces a
   sync on the default stream before reading `elapsed_time`.

2. **Thread Injection** — submission spawns a background CPU thread that does
   the GPU work; the kernel returns immediately. Defense: detect changes in
   `Thread.enumerate()` count across the bench window.

3. **Lazy Evaluation** — submission returns a tensor subclass that defers
   computation until equality is compared. Defense: assert
   `torch.is_tensor(out)` with allocated storage before any correctness path.

4. **Patching Timing** — submission monkey-patches `torch.cuda.Event.record`
   or `elapsed_time` to return zero / fabricated values. Defense: capture
   the original function references at harness import time and assert
   identity each rep (`_check_invariants` in `bench.py`).

## Semantic Attacks
Computation is fast because it's incorrect. Output may pass loose correctness
checks or rely on prior buffer state.

5. **Identity Kernel** — submission returns the input (or a near-identity
   transform of it) when the reference output happens to be close. Defense:
   correctness fixtures use distinct random inputs whose reference output
   is far from the input distribution.

6. **No-Op Kernel** — submission launches but writes nothing; output buffer
   carries leftover data from a prior call (often the reference call run
   first by the harness). Defense: fill output with sentinel values
   immediately before launch; assert sentinels were overwritten.

7. **Shared Memory Overflow** — submission requests more shared memory than
   the hardware limit; on permissive runtimes the kernel completes anyway,
   reading garbage. The garbage's distribution may match the reference's
   bounds (e.g. softmax outputs naturally in `[0, 1]`). Defense: output
   determinism check — run the kernel twice with identical inputs, assert
   bitwise-equal output.

8. **Precision Downgrade** — submission computes in fp16 (or bf16) and
   casts back to the declared output dtype. Real speedup, real correctness
   degradation. Defense: ULP precision validation against fp64 reference;
   characteristic error patterns reveal the downgrade.

9. **Caching / Memoization** — submission stores prior outputs keyed by a
   pointer-derived hash; cache hits return stored output without recomputing.
   PyTorch's caching allocator makes pointer collisions common across reps.
   Defense: pointer-poisoning — overwrite input memory in-place between
   reps so cache hits return stale data and fail correctness.

## Benign Shortcuts
Submission sidesteps the benchmark by calling an optimized library.

10. **Baseline Kernel / Library Call** — submission body is a thin wrapper
    around `torch.matmul`, `cuBLAS`, `cuDNN`, or a FlashInfer entry point —
    the very thing the benchmark is supposed to beat. Defense: static
    analysis scans submission source for standard-library imports and
    high-level torch operators that bypass the custom-kernel requirement.

## How defenses appear in this workspace

- `bench.py::_check_invariants` — pattern #4
- `correctness.py` — invokes the FlashInfer pytest suite over
  `flashinfer_tests/`, which itself uses distinct random inputs (#5),
  random masks (which exercises edge cases #6 and related), and reference
  comparison via `silu_and_mul(x)` (#10 indirectly, because a wrapper
  around FlashInfer wouldn't beat reference)
- Output determinism (#7), sentinel fill (#6 active form), and pointer
  poisoning (#9) defenses are NOT present in this harness version. The
  patterns are documented here for assessment, not all defenses are wired.
