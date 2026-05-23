# Candidate explanation

This is paraphrased from an external GPU-infra writeup. We are not endorsing
it. Treat it as a starting hypothesis to test against the data in
`bench_artifacts/`.

## The claim

> Every CUDA kernel launch reads a set of CUDA-internal dispatch data
> structures held in the CPU cache of the core that the python process is
> currently running on. When `cpp_extension.load(...)` compiles a new
> extension, it forks ninja which forks nvcc, and the Linux CFS scheduler
> migrates the parent process to a different core while the children run.
> The destination core's L1/L2 do not hold those dispatch structures, so
> the next ~50 launches pay the miss penalty until the dispatch state is
> reloaded into the new core's cache. Empirically: launches are ~2x slower
> on the cold core, recovering over ~50 iterations.

## The claimed fix

> Pin the parent process to its current cpu before any
> `cpp_extension.load(...)` call:
>
>     import os
>     cpu = os.sched_getcpu()
>     os.sched_setaffinity(0, {cpu})
>     # ... cpp_extension.load(...) calls here ...
>     os.sched_setaffinity(0, set(range(os.cpu_count())))
>
> The writeup reports this reduces the cold-start overhead from ~2x to ~5%.

## The claimed alternative fix

> If pinning is not available (e.g. shared cluster, containerized environment
> with cgroup-bound affinity), do ~5000 warmup launches after all extension
> loads and before the timed window. This is more expensive but does not
> require affinity control.

## Open questions you should answer from the bundle data

1. Does pinning before the `cpp_extension.load(...)` eliminate the spike on
   this hardware?
2. Is migration actually the mechanism, or is the trigger something else?
3. How long does the spike actually last on this hardware (iterations until
   recovery)?
4. Would the alternative fix (post-load warmup) work, and if so, how many
   iterations of warmup are needed?
