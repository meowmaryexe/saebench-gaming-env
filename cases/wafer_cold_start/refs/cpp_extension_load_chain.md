# What `torch.utils.cpp_extension.load(...)` actually does

A condensed reference for the call chain triggered by `load(name, sources, ...)`.
This is descriptive — read the torch source for ground truth.

1. **Source staging.** Sources are copied (or symlinked) into
   `build_directory`. A `build.ninja` file is generated describing the
   compile and link steps.

2. **Ninja invocation.** `subprocess.run(["ninja", ...])` is exec'd. This
   forks the python process; the child execs into `ninja`, which itself
   forks once per build step.

3. **Per-source compile.** Ninja launches `nvcc` (or `g++` for non-cuda
   sources) for each translation unit. Each `nvcc` invocation itself forks
   `cicc`, `ptxas`, `fatbinary`, and `cudafe++` to do the actual work. So
   one `load(...)` call typically generates 5–15 fork+exec events.

4. **Link.** Ninja links the resulting object files into a shared object
   (`.so`) using the host compiler.

5. **`dlopen`.** Back in the parent python process, the resulting `.so` is
   loaded via `importlib.machinery` (which calls `dlopen` on Linux). The
   pybind11 registration runs.

6. **Return.** A python module object is returned.

## Side effects on the parent process

- Each `subprocess.run(...)` with default kwargs forks the parent. Linux
  uses `vfork`-like semantics for the parent's address space until the
  child execs.
- The parent waits on the child via `waitpid`/`wait4`. While waiting, the
  scheduler is free to migrate the parent to another core if there is
  contention. (Whether it actually does so is hardware- and
  configuration-dependent.)
- After the child execs, COW page handling can invalidate parts of the
  parent's TLB and L1.
- The new `.so` is mmap'd into the parent's address space. Mapping a fresh
  file allocates VMA slots and may flush portions of the TLB.

## Caches that could be cold after a `load(...)` call

- L1d / L2 of the current cpu, if any process state was evicted by the
  child running on the same core.
- TLB entries for previously-mapped pages, if the new VMA setup invalidated
  them.
- Branch predictor state for code paths not exercised since the load.
- CUDA driver's per-stream / per-context launch-path data structures,
  which are owned by the driver and may have been touched by the dlopen of
  a new pybind11-registered module.

## Caches that are NOT cold

- The GPU itself — the GPU has its own caches and clocks, unrelated to the
  CPU's scheduler decisions.
