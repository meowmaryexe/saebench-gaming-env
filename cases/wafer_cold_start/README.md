# Cold-start spike triage

## Symptom

Our kernel-bench rig produces inconsistent timings. The same custom CUDA kernel,
run twice on the same node, comes out as either a 44% regression or a 12%
improvement against its reference. The difference between the two runs is
whether the python process compiled any other `torch.utils.cpp_extension.load(...)`
modules immediately before the timed window.

We need a defensible answer to three questions:

1. What event is the *trigger* of the per-launch spike?
2. Why does the launch path even pay this cost?
3. What is the smallest reliable fix?

A team member pointed us at a public write-up arguing the cause is
**process migration during the nvcc compile** and that the fix is
**`sched_setaffinity` pinning before any `cpp_extension.load(...)`**. Treat
that as a hypothesis to evaluate, not a conclusion (`hypothesis/blog_hypothesis.md`).

## What's in this bundle

```
harness/
  bench.py     - the diagnostic bench code that produced all artifacts here
  kernel.cu    - source of the no-op-ish kernel used as the launch probe
bench_artifacts/
  <arm>_rep<NN>/
    per_iter.csv   - phase, iter, cpu_id, wall_us, gpu_us
    sched.jsonl    - timestamped events (arm_start, compile_done, burn_in_done,
                     window_start/end, arm-specific action, dwell_done)
    summary.json   - arm metadata
    system.txt     - nvidia-smi, uname, nproc, lscpu, taskset
hypothesis/
  blog_hypothesis.md   - the candidate explanation; test it
refs/
  cpp_extension_load_chain.md   - what the load() call actually does
  cuda_event_timing.md          - what wall_us vs gpu_us each measure
```

## Arms

Each arm is a fresh process. All arms share the same template (pin → compile
probe kernel → burn-in → A1 window → arm-specific action → A2 window → dwell →
A3 window). They differ only in the action between A1 and A2.

| arm         | action between A1 and A2                                   |
|-------------|------------------------------------------------------------|
| `idle_pin`  | nothing                                                    |
| `migrate`   | `sched_setaffinity` to a different cpu (still pinned)      |
| `fork_only` | spawn six `/bin/true` subprocesses, wait for each          |
| `load_open` | `cpp_extension.load(...)` a fresh extension; open affinity |
| `load_pin`  | `cpp_extension.load(...)` a fresh extension; pinned        |

Four reps of each arm, each in its own container. Window length 500.

## Deliverable

Write `postmortem.md` in your working directory. Use exactly these section
headings (lowercase, single line each, in this order):

```
# root_cause
# evidence_chain
# hypothesis_test
# minimum_fix
# alternative_fix
# generalization
```

Section contents:

- **root_cause** — One paragraph. Name the trigger event (one of the arm
  actions or a specific component of one). Distinguish the *trigger* from
  the *mechanism* that explains why it costs time.
- **evidence_chain** — Cite specific arms / phases / iteration ranges from
  `bench_artifacts/`. For each claim, point to which CSV or sched event
  supports it. Distinguish what each arm rules in vs rules out.
- **hypothesis_test** — Evaluate `hypothesis/blog_hypothesis.md` claim by
  claim against the data: which parts hold, which parts don't, and which
  arm comparison settles each.
- **minimum_fix** — The smallest change to the harness that eliminates the
  spike. Include the actual code snippet. Justify why it works from your
  mechanism.
- **alternative_fix** — A second option with a different tradeoff
  (e.g. cost, generality, deployment friction). Note when you would
  prefer each.
- **generalization** — Which other common operations would produce the
  same spike (one-line list, two to four items, each backed by your
  mechanism).

Keep claims pinned to the data. Numbers should cite specific iters or
arm comparisons. Vague phrasing ("cache effects", "system noise") without
a specific arm comparison will not score.
