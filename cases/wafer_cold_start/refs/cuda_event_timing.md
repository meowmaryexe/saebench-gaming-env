# What `torch.cuda.Event.elapsed_time` measures

A short reference for the two columns recorded by `bench.py`:

- `wall_us` — `time.perf_counter_ns()` delta around the (record start →
  launch → record end → synchronize) sequence. This is CPU wall time and
  includes:
  - CPU-side launch overhead (param packing, driver call, stream
    enqueue)
  - GPU kernel runtime (because of the trailing synchronize)
  - Any CPU-side stalls (e.g. cache miss in the launch path, waitpid
    latency)
- `gpu_us` — `start.elapsed_time(end)` × 1000. The CUDA runtime returns
  the time between two events as observed by the GPU's own timer.
  Important subtleties:
  - The events are *recorded* into the stream by the CPU. If the CPU is
    slow to enqueue the second event after the kernel, the GPU sees a
    gap between kernel completion and the second event marker, and that
    gap shows up in `elapsed_time`.
  - Therefore `gpu_us` is not a pure GPU-side metric; CPU-side
    enqueue delays leak into it whenever the launch is short.
  - For a kernel whose actual GPU runtime is on the order of 5–10 us,
    a CPU stall of 100+ us between enqueue calls will dominate the
    reported `gpu_us`.

## Implication

If both `wall_us` and `gpu_us` move together on a particular iteration,
the most parsimonious explanation is CPU-side delay in the enqueue path,
because that's the only thing both metrics share. A change in *only*
`wall_us` (with `gpu_us` flat) would imply CPU work entirely after the
kernel completion. A change in *only* `gpu_us` would imply genuine GPU
clock/throttle variation, which is uncommon at this timescale.
