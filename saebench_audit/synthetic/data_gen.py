"""Synthetic-task generators for the validity panel (Section 4).

This module draws a fresh activation pool from a SynthSAEBench model for a
given ``(base_model_repo, base_model_revision, seed)`` triple and builds
hierarchy-aware sparse-probing, TPP, and SCR tasks on top of it. Outputs are
written to disk so they can be reused across SAEs in the panel.

Per ``(base_model, seed)`` the output directory contains:

* ``hidden.pt``, ``features.pt``, ``feature_binary.pt``
* ``sp_tasks.json``, ``sp_labels.pt``
* ``tpp_tasks.json``, ``tpp_sibling_indices.pt``
* ``scr_tasks.json``, ``scr_labels.pt``

Reseeding redraws both the activation pool and the per-task feature/atom
selections; the hierarchy itself is regenerated deterministically from the
model config (seed-independent).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from sae_lens.synthetic import SyntheticModel
from sae_lens.synthetic.hierarchy.hierarchy import generate_hierarchy
from sae_lens.synthetic.hierarchy.node import HierarchyNode

V1_REPO = "decoderesearch/synth-sae-bench-16k-v1"
NUM_SAMPLES = 60_000
D_SAE = 4096
DEVICE_DEFAULT = "cuda"

# Frozen task feature picks used by the paper's Section 4 figures (Figure 1,
# Table 2). The current generators in this file pick different feature
# indices: an earlier version of ``data_gen`` seeded its task RNGs as
# ``RandomState(seed)`` / ``RandomState(seed + 1)`` etc.; the multi-seed
# rewrite shifted those to ``seed + 100`` / ``seed + 1000`` /
# ``seed + 9000`` / ``seed + next_id + 5000`` so per-task RNG streams differ
# across reseeds. Loading the fixture preserves the paper's exact picks for
# users who want bit-equivalent Section-4 inputs.
PAPER_FIXTURES_DIR = Path(__file__).resolve().parent / "paper_fixtures"


@dataclass
class TPPTask:
    """A TPP sibling-group task. Four siblings under one depth-3 parent."""

    name: str
    parent_idx: int
    sibling_feats: list[int]
    category: str  # "all_in" or "all_out"


@dataclass
class SCRTask:
    """A spurious-correlation-removal task on a (T, S) feature pair."""

    name: str
    t_op: str
    t_feats: list[int]
    t_cat: str
    s_op: str
    s_feats: list[int]
    s_cat: str
    root_t: int
    root_s: int
    cell_counts: dict[str, int]


@dataclass
class SPTask:
    """A sparse-probing task: classify whether a feature pattern is firing."""

    id: int
    op: str
    feats: list[int]
    type: str
    pos_rate: float

    def name(self) -> str:
        return f"{self.type}__{self.op}__{'_'.join(str(f) for f in self.feats)}"


def model_tag_for_repo(repo: str, revision: str | None = None) -> str:
    """Return a filesystem-safe tag for ``(repo, revision)``."""
    rev = revision if revision is not None else "main"
    return f"{repo.replace('/', '__').replace(':', '__')}__rev_{rev.replace('/', '__')}"


def _eval_op(feature_binary: torch.Tensor, op: str, feats: list[int]) -> torch.Tensor:
    cols = [feature_binary[:, f] for f in feats]
    if op == "single":
        return cols[0]
    if op == "and":
        out = cols[0].clone()
        for c in cols[1:]:
            out = out * c
        return out
    if op == "or":
        out = cols[0].clone()
        for c in cols[1:]:
            out = ((out + c) > 0).float()
        return out
    raise ValueError(f"Unknown op {op}")


def _construct_label_np(
    fb_np: npt.NDArray[np.bool_], op: str, feats: list[int]
) -> npt.NDArray[np.bool_]:
    cols = [fb_np[:, f] for f in feats]
    if op == "single":
        return cols[0]
    if op == "and":
        out = cols[0].copy()
        for c in cols[1:]:
            out &= c
        return out
    if op == "or":
        out = cols[0].copy()
        for c in cols[1:]:
            out |= c
        return out
    raise ValueError(op)


def _collect_depth3_sibling_groups(
    roots: list[HierarchyNode],
) -> list[tuple[int, int, list[int]]]:
    groups: list[tuple[int, int, list[int]]] = []
    for root_i, root in enumerate(roots):
        stack: list[tuple[HierarchyNode, int]] = [(root, 0)]
        while stack:
            node, d = stack.pop(0)
            if d == 2:
                child_feats = [
                    c.feature_index
                    for c in node.children
                    if c.feature_index is not None
                ]
                if len(child_feats) == 4 and getattr(
                    node, "mutually_exclusive_children", False
                ):
                    parent_idx = node.feature_index
                    if parent_idx is None:
                        continue
                    groups.append((int(parent_idx), root_i, child_feats))
                continue
            for c in node.children:
                stack.append((c, d + 1))
    return groups


def _feature_to_root(roots: list[HierarchyNode]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for r_i, root in enumerate(roots):
        for f in root.get_all_feature_indices():
            mapping[int(f)] = r_i
    return mapping


def _sample_pool(
    model: SyntheticModel, num_samples: int, seed: int, batch: int = 4096
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw a synthetic ``(hidden, features)`` pool of size ``num_samples``."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    hidden_chunks: list[torch.Tensor] = []
    feat_chunks: list[torch.Tensor] = []
    remaining = num_samples
    with torch.no_grad():
        while remaining > 0:
            n = min(batch, remaining)
            h, f = model.sample_with_features(n)
            hidden_chunks.append(h.cpu().float())
            feat_chunks.append(f.cpu().float())
            remaining -= n
    hidden = torch.cat(hidden_chunks, dim=0)
    features = torch.cat(feat_chunks, dim=0)
    return hidden, features


def _build_tpp_tasks(
    fire_rates: torch.Tensor,
    groups: list[tuple[int, int, list[int]]],
    seed: int,
    tpp_per_cat: int = 30,
    min_sib_rate: float = 1e-4,
) -> list[TPPTask]:
    all_in: list[tuple[int, int, list[int]]] = []
    all_out: list[tuple[int, int, list[int]]] = []
    for parent_idx, root_i, feats in groups:
        in_count = sum(1 for f in feats if f < D_SAE)
        if in_count == 4:
            all_in.append((parent_idx, root_i, feats))
        elif in_count == 0:
            all_out.append((parent_idx, root_i, feats))

    def usable(feats: list[int]) -> bool:
        return all(fire_rates[f].item() >= min_sib_rate for f in feats)

    rng = np.random.RandomState(seed + 100)
    out: list[TPPTask] = []
    for cat, gs in [("all_in", all_in), ("all_out", all_out)]:
        eligible = [g for g in gs if usable(g[2])]
        rng.shuffle(eligible)
        for parent_idx, _, feats in eligible[:tpp_per_cat]:
            out.append(
                TPPTask(
                    name=f"tpp_{cat}_parent{parent_idx}",
                    parent_idx=int(parent_idx),
                    sibling_feats=[int(x) for x in feats],
                    category=cat,
                )
            )
    return out


def _build_scr_tasks(  # noqa: ARG001
    fb_np: npt.NDArray[np.bool_],
    fire_rates: torch.Tensor,
    feat_to_root: dict[int, int],
    seed: int,
    scr_attempts: int = 15,
    n_per_combo: int = 3,
    min_per_cell_default: int = 100,
    min_per_cell_out_out: int = 40,
) -> list[SCRTask]:
    in_features = [
        int(f)
        for f in torch.where(
            (torch.arange(fb_np.shape[1]) < D_SAE)
            & (fire_rates >= 0.005)
            & (fire_rates <= 0.25)
        )[0].tolist()
    ]
    out_features = [
        int(f)
        for f in torch.where(
            (torch.arange(fb_np.shape[1]) >= D_SAE)
            & (fire_rates >= 0.0005)
            & (fire_rates <= 0.03)
        )[0].tolist()
    ]

    def root_of(f: int) -> int | None:
        return feat_to_root.get(f)

    def group_by_root(pool: list[int]) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for f in pool:
            r = root_of(f)
            if r is None:
                continue
            out.setdefault(r, []).append(f)
        return out

    in_pool_by_root = group_by_root(in_features)
    out_pool_by_root = group_by_root(out_features)

    def try_build_concept(
        cat: str, op: str, scr_rng: np.random.RandomState, max_tries: int = 100
    ) -> tuple[list[int], int, torch.Tensor, float] | None:
        pool = in_features if cat == "in" else out_features
        pool_by_root = in_pool_by_root if cat == "in" else out_pool_by_root
        if cat == "in":
            rate_lo, rate_hi = 0.03, 0.30
            or_size = 2
        else:
            rate_lo, rate_hi = 0.001, 0.10
            or_size = 10
        for _ in range(max_tries):
            if op == "single":
                if not pool:
                    return None
                f = int(scr_rng.choice(pool))
                root = root_of(f)
                if root is None:
                    continue
                lab = _construct_label_np(fb_np, "single", [f])
                rate = float(lab.mean())
                if rate_lo <= rate <= rate_hi:
                    return [f], root, torch.from_numpy(lab).float(), rate
            else:
                size = 2 if op == "and" else or_size
                eligible = [r for r, fs in pool_by_root.items() if len(fs) >= size]
                if not eligible:
                    return None
                root = int(scr_rng.choice(eligible))
                chosen = list(
                    scr_rng.choice(pool_by_root[root], size=size, replace=False)
                )
                chosen = [int(x) for x in chosen]
                lab = _construct_label_np(fb_np, op, chosen)
                rate = float(lab.mean())
                if op == "and" and rate < rate_lo * 0.1:
                    continue
                if op == "or" and rate > rate_hi * 2:
                    continue
                if rate < rate_lo * 0.1:
                    continue
                return chosen, root, torch.from_numpy(lab).float(), rate
        return None

    scr_tasks: list[SCRTask] = []
    combos = [("in", "single"), ("in", "or"), ("out", "or")]
    scr_rng = np.random.RandomState(seed + 1000)
    for t_cat, t_op in combos:
        for s_cat, s_op in combos:
            kept = 0
            for _ in range(scr_attempts):
                t_res = try_build_concept(t_cat, t_op, scr_rng)
                if t_res is None:
                    break
                t_feats, t_root, t_lab, _ = t_res
                s_res = None
                for _ in range(20):
                    cand = try_build_concept(s_cat, s_op, scr_rng)
                    if cand is None:
                        continue
                    if cand[1] != t_root:
                        s_res = cand
                        break
                if s_res is None:
                    continue
                s_feats, s_root, s_lab, _ = s_res
                cells = {
                    "00": int(((t_lab == 0) & (s_lab == 0)).sum()),
                    "01": int(((t_lab == 0) & (s_lab == 1)).sum()),
                    "10": int(((t_lab == 1) & (s_lab == 0)).sum()),
                    "11": int(((t_lab == 1) & (s_lab == 1)).sum()),
                }
                min_cell = (
                    min_per_cell_out_out
                    if (t_cat == "out" and s_cat == "out")
                    else min_per_cell_default
                )
                if min(cells.values()) < min_cell:
                    continue
                scr_tasks.append(
                    SCRTask(
                        name=f"scr_T{t_cat[0]}_{t_op}_S{s_cat[0]}_{s_op}_{kept}",
                        t_op=t_op,
                        t_feats=t_feats,
                        t_cat=t_cat,
                        s_op=s_op,
                        s_feats=s_feats,
                        s_cat=s_cat,
                        root_t=t_root,
                        root_s=s_root,
                        cell_counts=cells,
                    )
                )
                kept += 1
                if kept >= n_per_combo:
                    break
    return scr_tasks


def _build_sp_tasks(
    fb_np: npt.NDArray[np.bool_],
    fire_rates: torch.Tensor,
    seed: int,
    per_cat: int = 24,
) -> list[SPTask]:
    in_dict_mask = torch.arange(fb_np.shape[1]) < D_SAE
    out_dict_mask = ~in_dict_mask

    in_lo, in_hi = 0.005, 0.20
    out_lo, out_hi = 0.0007, 0.005
    in_singles = [
        int(f)
        for f in torch.where(
            in_dict_mask & (fire_rates >= in_lo) & (fire_rates <= in_hi)
        )[0].tolist()
    ]
    out_singles = [
        int(f)
        for f in torch.where(
            out_dict_mask & (fire_rates >= out_lo) & (fire_rates <= out_hi)
        )[0].tolist()
    ]
    rng_singles = np.random.RandomState(seed + 9000)
    rng_singles.shuffle(in_singles)
    rng_singles.shuffle(out_singles)

    sp_tasks: list[SPTask] = []
    next_id = 0
    for f in in_singles[:per_cat]:
        lab = _eval_op(torch.from_numpy(fb_np).float(), "single", [f])
        sp_tasks.append(
            SPTask(
                id=next_id,
                op="single",
                feats=[int(f)],
                type="single_in",
                pos_rate=float(lab.mean()),
            )
        )
        next_id += 1
    for f in out_singles[:per_cat]:
        lab = _eval_op(torch.from_numpy(fb_np).float(), "single", [f])
        sp_tasks.append(
            SPTask(
                id=next_id,
                op="single",
                feats=[int(f)],
                type="single_out",
                pos_rate=float(lab.mean()),
            )
        )
        next_id += 1

    in_bool = [
        int(f)
        for f in torch.where(
            in_dict_mask & (fire_rates >= 0.02) & (fire_rates <= 0.30)
        )[0].tolist()
    ]
    out_bool = [
        int(f)
        for f in torch.where(
            out_dict_mask & (fire_rates >= 0.0005) & (fire_rates <= 0.003)
        )[0].tolist()
    ]
    next_id = _sample_bool_sp(
        sp_tasks,
        fb_np,
        "and",
        in_bool,
        in_bool,
        [(2, 0), (3, 0)],
        0.001,
        0.20,
        "bool_in",
        next_id,
        seed,
        4,
    )
    next_id = _sample_bool_sp(
        sp_tasks,
        fb_np,
        "or",
        in_bool,
        in_bool,
        [(2, 0), (3, 0)],
        0.005,
        0.25,
        "bool_in",
        next_id,
        seed,
        4,
    )
    next_id = _sample_bool_sp(
        sp_tasks,
        fb_np,
        "or",
        out_bool,
        out_bool,
        [(2, 0), (3, 0), (4, 0)],
        0.001,
        0.05,
        "bool_out",
        next_id,
        seed,
        4,
    )
    min_and_mixed = 30.0 / NUM_SAMPLES
    next_id = _sample_bool_sp(
        sp_tasks,
        fb_np,
        "and",
        in_bool,
        out_bool,
        [(1, 1), (2, 1), (1, 2)],
        min_and_mixed,
        0.10,
        "bool_mixed",
        next_id,
        seed,
        4,
    )
    next_id = _sample_bool_sp(
        sp_tasks,
        fb_np,
        "or",
        in_bool,
        out_bool,
        [(1, 1), (1, 2), (2, 2)],
        0.005,
        0.25,
        "bool_mixed",
        next_id,
        seed,
        4,
    )
    return sp_tasks


def _sample_bool_sp(  # noqa: PLR0913
    sp_tasks: list[SPTask],
    fb_np: npt.NDArray[np.bool_],
    op: str,
    cand_a: list[int],
    cand_b: list[int],
    sizes: list[tuple[int, int]],
    rate_lo: float,
    rate_hi: float,
    task_type: str,
    id_start: int,
    seed: int,
    n_tasks: int,
    max_tries: int = 200,
) -> int:
    next_id = id_start
    for size_a, size_b in sizes:
        if len(cand_a) < size_a or (size_b > 0 and len(cand_b) < size_b):
            continue
        rng_local = np.random.RandomState(seed + next_id + 5000)
        kept = 0
        for _ in range(n_tasks * 5):
            if kept >= n_tasks:
                break
            feats: list[int] | None = None
            for _ in range(max_tries):
                a_idx = rng_local.choice(len(cand_a), size=size_a, replace=False)
                cand = [cand_a[int(i)] for i in a_idx]
                if size_b > 0:
                    b_idx = rng_local.choice(len(cand_b), size=size_b, replace=False)
                    cand = cand + [cand_b[int(i)] for i in b_idx]
                lab_np = _construct_label_np(fb_np, op, cand)
                rate = float(lab_np.mean())
                if rate_lo <= rate <= rate_hi:
                    feats = cand
                    break
            if feats is None:
                continue
            lab_np = _construct_label_np(fb_np, op, feats)
            sp_tasks.append(
                SPTask(
                    id=next_id,
                    op=op,
                    feats=[int(x) for x in feats],
                    type=task_type,
                    pos_rate=float(lab_np.mean()),
                )
            )
            next_id += 1
            kept += 1
    return next_id


def _save(
    out_dir: Path,
    *,
    hidden: torch.Tensor,
    features: torch.Tensor,
    feature_binary: torch.Tensor,
    sp_tasks: list[SPTask],
    tpp_tasks: list[TPPTask],
    scr_tasks: list[SCRTask],
) -> None:
    torch.save(hidden, out_dir / "hidden.pt")
    torch.save(features, out_dir / "features.pt")
    torch.save(feature_binary, out_dir / "feature_binary.pt")

    sp_labels = {
        t.id: _eval_op(feature_binary, t.op, t.feats).to(torch.int64) for t in sp_tasks
    }
    torch.save(sp_labels, out_dir / "sp_labels.pt")
    with open(out_dir / "sp_tasks.json", "w") as f:
        json.dump(
            [
                {
                    "id": t.id,
                    "op": t.op,
                    "feats": t.feats,
                    "type": t.type,
                    "name": t.name(),
                    "pos_rate": t.pos_rate,
                }
                for t in sp_tasks
            ],
            f,
            indent=2,
        )

    tpp_sibling_indices: dict[str, dict[str, torch.Tensor]] = {}
    for t in tpp_tasks:
        per_class: dict[str, torch.Tensor] = {}
        for f in t.sibling_feats:
            idx = torch.nonzero(feature_binary[:, f].bool(), as_tuple=True)[0]
            per_class[f"class_{f}"] = idx
        tpp_sibling_indices[t.name] = per_class
    torch.save(tpp_sibling_indices, out_dir / "tpp_sibling_indices.pt")
    with open(out_dir / "tpp_tasks.json", "w") as f:
        json.dump(
            [
                {
                    "name": t.name,
                    "parent_idx": t.parent_idx,
                    "sibling_feats": t.sibling_feats,
                    "category": t.category,
                }
                for t in tpp_tasks
            ],
            f,
            indent=2,
        )

    scr_labels: dict[str, dict[str, torch.Tensor]] = {}
    payload: list[dict[str, object]] = []
    for t in scr_tasks:
        t_lab = _eval_op(feature_binary, t.t_op, t.t_feats)
        s_lab = _eval_op(feature_binary, t.s_op, t.s_feats)
        scr_labels[t.name] = {
            "t": t_lab.to(torch.bool),
            "s": s_lab.to(torch.bool),
        }
        payload.append(
            {
                "name": t.name,
                "t_op": t.t_op,
                "t_feats": t.t_feats,
                "t_cat": t.t_cat,
                "s_op": t.s_op,
                "s_feats": t.s_feats,
                "s_cat": t.s_cat,
                "root_t": t.root_t,
                "root_s": t.root_s,
                "cell_counts": t.cell_counts,
            }
        )
    torch.save(scr_labels, out_dir / "scr_labels.pt")
    with open(out_dir / "scr_tasks.json", "w") as f:
        json.dump(payload, f, indent=2)


def _load_tasks_from_fixture(
    fixture_dir: Path,
) -> tuple[list[SPTask], list[TPPTask], list[SCRTask]]:
    """Load (sp_tasks, tpp_tasks, scr_tasks) from JSON files in ``fixture_dir``.

    Each fixture file is the same JSON layout that ``_save`` writes.
    """
    with open(fixture_dir / "sp_tasks.json") as f:
        sp_payload = json.load(f)
    sp_tasks = [
        SPTask(
            id=int(t["id"]),
            op=t["op"],
            feats=[int(x) for x in t["feats"]],
            type=t["type"],
            pos_rate=float(t["pos_rate"]),
        )
        for t in sp_payload
    ]
    with open(fixture_dir / "tpp_tasks.json") as f:
        tpp_payload = json.load(f)
    tpp_tasks = [
        TPPTask(
            name=t["name"],
            parent_idx=int(t["parent_idx"]),
            sibling_feats=[int(x) for x in t["sibling_feats"]],
            category=t["category"],
        )
        for t in tpp_payload
    ]
    with open(fixture_dir / "scr_tasks.json") as f:
        scr_payload = json.load(f)
    scr_tasks = [
        SCRTask(
            name=t["name"],
            t_op=t["t_op"],
            t_feats=[int(x) for x in t["t_feats"]],
            t_cat=t["t_cat"],
            s_op=t["s_op"],
            s_feats=[int(x) for x in t["s_feats"]],
            s_cat=t["s_cat"],
            root_t=int(t["root_t"]),
            root_s=int(t["root_s"]),
            cell_counts={k: int(v) for k, v in t["cell_counts"].items()},
        )
        for t in scr_payload
    ]
    return sp_tasks, tpp_tasks, scr_tasks


def generate(
    base_model_repo: str,
    base_model_revision: str | None,
    seed: int,
    out_dir: Path,
    *,
    device: str = DEVICE_DEFAULT,
    num_samples: int = NUM_SAMPLES,
    paper_fixture: str | Path | None = None,
) -> None:
    """Generate (or reuse) the synthetic-task data for one ``(model, seed)``.

    If ``out_dir / "scr_tasks.json"`` already exists, returns immediately
    without recomputing anything. Otherwise the activation pool, hierarchy,
    and per-task feature/atom selections are rebuilt.

    If ``paper_fixture`` is given, the SP / TPP / SCR task feature picks are
    loaded verbatim from JSON files in that directory instead of being
    sampled fresh. The activation pool, ``feature_binary``, and the
    per-task labels (``sp_labels``, ``tpp_sibling_indices``, ``scr_labels``)
    are still recomputed deterministically from ``base_model_repo`` /
    ``seed``; only the *task feature selections* are taken from the
    fixture. ``paper_fixture`` may be:

        * a string like ``"v1_seed_1234"`` → resolved under
          ``saebench_audit/synthetic/paper_fixtures/<name>``.
        * an absolute or relative ``Path`` to a directory containing
          ``sp_tasks.json``, ``tpp_tasks.json``, ``scr_tasks.json``.

    The shipped ``v1_seed_1234`` fixture matches the feature picks used to
    generate the paper's Figure 1 / Table 2. See ``PAPER_FIXTURES_DIR`` for
    the search root.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sentinel = out_dir / "scr_tasks.json"
    if sentinel.exists():
        return

    if base_model_revision is None:
        model = SyntheticModel.from_pretrained(base_model_repo, device=device)
    else:
        model = SyntheticModel.from_pretrained(
            base_model_repo, model_path=base_model_revision, device=device
        )

    hierarchy_cfg = model.cfg.hierarchy
    if hierarchy_cfg is None:
        raise ValueError(
            f"Synthetic model {base_model_repo!r} has no hierarchy config; "
            "cannot build TPP / SCR sibling tasks without it."
        )
    hierarchy = generate_hierarchy(model.cfg.num_features, hierarchy_cfg, seed=None)
    feat_to_root = _feature_to_root(hierarchy.roots)

    hidden, features = _sample_pool(model, num_samples, seed=seed)
    feature_binary = (features > 0).float()
    fire_rates = feature_binary.mean(dim=0)
    fb_np = feature_binary.numpy().astype(bool)

    if paper_fixture is not None:
        fixture_dir = (
            paper_fixture
            if isinstance(paper_fixture, Path)
            else PAPER_FIXTURES_DIR / paper_fixture
        )
        if not fixture_dir.is_dir():
            raise FileNotFoundError(f"paper_fixture directory not found: {fixture_dir}")
        sp_tasks, tpp_tasks, scr_tasks = _load_tasks_from_fixture(fixture_dir)
        print(
            f"[data_gen] loaded paper fixture from {fixture_dir}: "
            f"sp_tasks={len(sp_tasks)} tpp_tasks={len(tpp_tasks)} "
            f"scr_tasks={len(scr_tasks)}"
        )
    else:
        groups = _collect_depth3_sibling_groups(hierarchy.roots)
        tpp_tasks = _build_tpp_tasks(fire_rates, groups, seed=seed)
        scr_tasks = _build_scr_tasks(
            fb_np, fire_rates, feat_to_root=feat_to_root, seed=seed
        )
        sp_tasks = _build_sp_tasks(fb_np, fire_rates, seed=seed)

        type_counts: Counter[str] = Counter(t.type for t in sp_tasks)
        print(
            f"[data_gen] sp_tasks={len(sp_tasks)} (by type: {dict(type_counts)}); "
            f"tpp_tasks={len(tpp_tasks)}; scr_tasks={len(scr_tasks)}"
        )

    _save(
        out_dir,
        hidden=hidden,
        features=features,
        feature_binary=feature_binary,
        sp_tasks=sp_tasks,
        tpp_tasks=tpp_tasks,
        scr_tasks=scr_tasks,
    )
