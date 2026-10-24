"""Reproduce every reported experiment.

    python experiments/run_benchmark.py                 # full protocol (TEST seeds), trains the model if missing
    python experiments/run_benchmark.py --quick         # 1 seed, 3 families: smoke test (~2 min)

Writes (under --out, default results/):
    baseline.json  static.json  remem.json  ablation.json     aggregated metrics (mean, 95% CI over seeds)
    sensitivity.json  curves.json  growth.json  calibration.json       robustness and diagnostics
    failures.jsonl  failure_analysis.md                                 automatic failure analysis
    RESULTS.md                                                          tables generated from the JSONs
    plots/*.png   runs/<family>_seed0_<system>/ledger.json              inspect with remem.viz.ledger_cli
    records/*.jsonl.gz                                                  per-task records of the main systems

Nothing is hard-coded: every number in RESULTS.md and the plots is computed from the runs above.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from remem.config import load_systems  # noqa: E402
from remem.eval import failures as failmod  # noqa: E402
from remem.eval.agent import AgentConfig  # noqa: E402
from remem.eval.metrics import aggregate, mean_ci, paired_delta, summarize_run  # noqa: E402
from remem.eval.pools import ensure_pools, load_pools, make_embedder  # noqa: E402
from remem.eval.protocol import EXTRA_FAMILIES, HEADLINE_FAMILIES, READER_MODES, TEST_SEEDS  # noqa: E402
from remem.eval.runner import apply_world_selection, run_stream  # noqa: E402
from remem.eval.world import make_world  # noqa: E402

log = logging.getLogger("benchmark")

BASELINES = ["A_vector"]
STATIC = ["B_static"]
REMEM = ["C_remem"]
ABLATION = [
    "abl0_base", "abl1_temporal", "abl2_utility_heuristic", "abl3_learned_model", "abl4_outcome_feedback",
    "abl5_retention", "abl6_contradiction", "abl7_cost_aware",
    "loo_no_feedback", "loo_no_contradiction", "loo_no_retention", "loo_no_cost_aware", "loo_heuristic_utility",
    "loo_no_exclusive_schema", "attr_presence", "attr_signals_loo", "attr_usage_only", "ret_frequency_prune",
]
NEW = ["loo_no_guard", "new_thompson", "schema_learned", "attr_signals_loo_ambiguous"]
CORE = ["A_vector", "B_static", "C_remem"]
MAIN_SYSTEMS = BASELINES + STATIC + REMEM + ABLATION + NEW
FAILURE_SYSTEMS = {"A_vector", "B_static", "C_remem"}
RECORD_SYSTEMS = {"A_vector", "B_static", "C_remem"}
CURVE_BINS = 10


# ── worker ─────────────────────────────────────────────────────────────────────

def run_job(job: dict[str, Any]) -> list[dict[str, Any]]:
    """Run all requested systems on one (family, seed, reader mode, perturbation) world."""
    logging.basicConfig(level=logging.WARNING)
    out_dir = Path(job["out_dir"])
    cache = Path(job["cache"])
    systems = load_systems(ROOT / "configs" / "systems.yaml")
    emb = make_embedder()
    world = make_world(job["family"], job["seed"], job["world_overrides"])
    agent_cfg = AgentConfig(mode=job["mode"], **job["agent_overrides"])
    pools_by_backend: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for name in job["systems"]:
        cfg = apply_world_selection(systems[name], world)
        # model paths are relative to the repo root
        if cfg.utility.model_path:
            cfg = replace(cfg, utility=replace(cfg.utility, model_path=str(ROOT / cfg.utility.model_path)))
        be = cfg.backend
        if be not in pools_by_backend:
            pools_by_backend[be] = load_pools(cache, job["family"], job["seed"], be, cfg.pool_size, job["world_overrides"])
        run = run_stream(world, pools_by_backend[be], cfg, emb, agent_cfg, seed=job["seed"],
                         feedback_noise=job["feedback_noise"])
        metrics = summarize_run(run)
        metrics['utility_demoted_gold_rate'] = sum('utility_demoted_gold' in failmod.classify(r, cfg.selection.top_k) for r in run.records) / max(1, len(run.records))
        recs = run.records
        n = len(recs)
        row: dict[str, Any] = {
            "tag": job["tag"], "mode": job["mode"], "family": job["family"], "seed": job["seed"], "system": name,
            "metrics": metrics,
            "success_seq": [int(r.success) for r in recs],
            "trap_seq": [r.n_trap_selected + r.n_hub_selected for r in recs],
            "gold_seq": [r.gold_in_context / max(1, r.n_gold) for r in recs],
            "dormant_seq": [(r.kind, int(r.success)) for r in recs if r.kind.startswith("critical")],
            "snapshots": run.snapshots,
        }
        # calibration bins (probability vs empirical gold rate) over considered candidates
        ps = np.asarray([t[0] for r in recs for t in r.scored_probs])
        ys = np.asarray([t[2] for r in recs for t in r.scored_probs])
        if len(ps) and name in CORE:
            edges = np.linspace(0, 1, 11)
            bins = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (ps >= lo) & ((ps < hi) if hi < 1 else (ps <= hi))
                if m.any():
                    bins.append([float(ps[m].mean()), float(ys[m].mean()), int(m.sum())])
            row["calibration"] = bins
        if name in FAILURE_SYSTEMS and job["tag"] == "main":
            row["failures"] = failmod.analyse(run, world, cfg.selection.top_k)
        if job["tag"] == "main" and job["mode"] == "first":
            if name in RECORD_SYSTEMS:
                p = out_dir / "records" / f"{job['family']}_{job['seed']}_{name}.jsonl.gz"
                p.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(p, "wt", encoding="utf-8") as fh:
                    for r in recs:
                        d = asdict(r)
                        d.pop("scored_probs")
                        fh.write(json.dumps(d) + "\n")
            if name in ("C_remem", "ret_frequency_prune") and job["seed"] == TEST_SEEDS[0]:
                d = out_dir / "runs" / f"{job['family']}_seed{job['seed']}_{name}"
                run.ledger.save(d / "ledger.json")
        rows.append(row)
    return rows


# ── orchestration ──────────────────────────────────────────────────────────────

def build_jobs(args, out_dir: Path, cache: Path) -> tuple[list[dict], list[tuple]]:
    families = list(HEADLINE_FAMILIES) + list(EXTRA_FAMILIES)
    seeds = TEST_SEEDS
    if args.quick:
        families = ["distractor", "contradiction", "budget_tight"]
        seeds = TEST_SEEDS[:1]
    base = {"out_dir": str(out_dir), "cache": str(cache), "feedback_noise": 0.0, "agent_overrides": {},
            "world_overrides": None}
    jobs: list[dict] = []
    for fam in families:
        for seed in seeds:
            for mode in READER_MODES:
                jobs.append({**base, "tag": "main", "family": fam, "seed": seed, "mode": mode, "systems": MAIN_SYSTEMS})
    if not args.skip_sensitivity:
        sens_fams = [f for f in HEADLINE_FAMILIES if f in families]
        for fam in sens_fams:
            for seed in seeds:
                for fn in (0.1, 0.2):
                    jobs.append({**base, "tag": f"feedback_noise_{fn}", "family": fam, "seed": seed, "mode": "first",
                                 "systems": CORE, "feedback_noise": fn})
                for irr in (0.0, 0.10):
                    jobs.append({**base, "tag": f"distraction_{irr}", "family": fam, "seed": seed, "mode": "first",
                                 "systems": CORE, "agent_overrides": {"per_irrelevant": irr}})
        for fam in [f for f in ("contradiction", "state_change") if f in families]:
            for seed in seeds:
                jobs.append({**base, "tag": "no_update_cue", "family": fam, "seed": seed, "mode": "first",
                             "systems": CORE + ["loo_no_exclusive_schema", "schema_learned"],
                             "world_overrides": {"update_cue_rate": 0.0}})
                jobs.append({**base, "tag": "extraction_noise_0.2", "family": fam, "seed": seed, "mode": "first",
                             "systems": CORE, "world_overrides": {"extraction_noise": 0.2}})
    pool_jobs = {(j["family"], j["seed"], be, 20, json.dumps(j["world_overrides"], sort_keys=True))
                 for j in jobs for be in ("vector",)}
    pool_jobs_t = [(f, s, b, k, json.loads(o)) for f, s, b, k, o in sorted(pool_jobs)]
    return jobs, pool_jobs_t


def group(rows: list[dict]) -> dict[tuple, list[dict]]:
    g: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        g[(r["tag"], r["mode"], r["family"], r["system"])].append(r)
    return g


def headline(rows_by_seed: dict[int, list[dict]], metric: str) -> list[float]:
    """Per-seed mean of `metric` over the headline families."""
    return [float(np.nanmean([r["metrics"].get(metric, np.nan) for r in rs])) for _, rs in sorted(rows_by_seed.items())]


def aggregate_all(rows: list[dict]) -> dict[str, Any]:
    """{tag: {mode: {system: {family|'HEADLINE': {metric: {mean, ci95, n}}}}}}"""
    out: dict[str, Any] = {}
    g = group(rows)
    for (tag, mode, fam, system), rs in g.items():
        node = out.setdefault(tag, {}).setdefault(mode, {}).setdefault(system, {})
        node[fam] = aggregate([r["metrics"] for r in sorted(rs, key=lambda r: r["seed"])])
    # headline across HEADLINE_FAMILIES: average families within seed, CI over seeds
    for tag, modes in out.items():
        for mode, systems in modes.items():
            for system, fams in systems.items():
                per_seed: dict[int, list[dict]] = defaultdict(list)
                for fam in HEADLINE_FAMILIES:
                    for r in g.get((tag, mode, fam, system), []):
                        per_seed[r["seed"]].append(r)
                if not per_seed:
                    continue
                keys = sorted({k for rs in per_seed.values() for r in rs for k in r["metrics"]})
                fams["HEADLINE"] = {}
                for k in keys:
                    m, ci = mean_ci(headline(per_seed, k))
                    fams["HEADLINE"][k] = {"mean": m, "ci95": ci, "n": len(per_seed)}
    return out


def paired_vs(rows: list[dict], a: str, b: str, tag: str, mode: str, metric: str, fams) -> tuple[float, float]:
    g = group(rows)
    da: dict[int, list[float]] = defaultdict(list)
    db: dict[int, list[float]] = defaultdict(list)
    for fam in fams:
        for r in g.get((tag, mode, fam, a), []):
            da[r["seed"]].append(r["metrics"][metric])
        for r in g.get((tag, mode, fam, b), []):
            db[r["seed"]].append(r["metrics"][metric])
    seeds = sorted(set(da) & set(db))
    return paired_delta([float(np.mean(da[s])) for s in seeds], [float(np.mean(db[s])) for s in seeds])


def curves(rows: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    g = group(rows)
    for (tag, mode, fam, system), rs in g.items():
        if tag != "main" or mode != "first" or system not in CORE + ["ret_frequency_prune"]:
            continue
        def binned(key):
            acc = []
            for r in rs:
                seq = np.asarray(r[key], dtype=float)
                chunks = np.array_split(seq, CURVE_BINS)
                acc.append([float(c.mean()) for c in chunks])
            return np.mean(acc, axis=0).tolist()
        out.setdefault(fam, {})[system] = {k: binned(k) for k in ("success_seq", "trap_seq", "gold_seq")}
        dorm = [x for r in rs for x in r["dormant_seq"]]
        if dorm:
            out[fam][system]["critical_success"] = {
                k: float(np.mean([s for kk, s in dorm if kk == k])) for k in {kk for kk, _ in dorm}}
    return out


def growth(rows: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for r in rows:
        if r["tag"] != "main" or r["mode"] != "first" or r["system"] not in ("C_remem", "A_vector", "ret_frequency_prune"):
            continue
        if r["seed"] != TEST_SEEDS[0]:
            continue
        out.setdefault(r["family"], {})[r["system"]] = r["snapshots"]
    return out


def calibration(rows: list[dict]) -> dict[str, Any]:
    acc: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0]))
    for r in rows:
        if r["tag"] != "main" or r["mode"] != "first" or "calibration" not in r:
            continue
        for b, (p, y, n) in enumerate(r["calibration"]):
            cell = acc[r["system"]][int(round(p * 10))]
            cell[0] += p * n
            cell[1] += y * n
            cell[2] += n
    return {s: [[c[0] / c[2], c[1] / c[2], c[2]] for _, c in sorted(bins.items())] for s, bins in acc.items()}


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")


def merge_failures(rows: list[dict], out: Path) -> dict[str, Any]:
    counts: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    n_tasks: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    examples: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if "failures" not in r or r["mode"] != "first":
            continue
        f = r["failures"]
        n_tasks[r["system"]][r["family"]] += f["n_tasks"]
        for cat, c in f["counts"].items():
            counts[r["system"]][r["family"]][cat] += c
        for cat, exs in f["examples"].items():
            for ex in exs:
                if len(examples[(r["system"])][cat]) < 12:
                    examples[r["system"]][cat].append(ex)
    with open(out / "failures.jsonl", "w", encoding="utf-8") as fh:
        for system, cats in examples.items():
            for cat, exs in cats.items():
                for ex in exs:
                    fh.write(json.dumps({"category": cat, **ex}) + "\n")
    return {"counts": {s: {f: dict(c) for f, c in d.items()} for s, d in counts.items()},
            "n_tasks": {s: dict(d) for s, d in n_tasks.items()}, "examples": {s: dict(d) for s, d in examples.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--quick", action="store_true", help="1 seed, 3 families, no sensitivity")
    ap.add_argument("--skip-sensitivity", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.skip_sensitivity = True
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    out = Path(args.out)
    cache = out / "cache" / "pools"
    t0 = time.time()

    models = ROOT / "results" / "models"
    if args.retrain or not (models / "utility_full.json").exists():
        import subprocess
        log.info("training utility models ...")
        subprocess.check_call([sys.executable, str(ROOT / "experiments" / "train_utility.py"), "--workers", str(args.workers)])

    jobs, pool_jobs = build_jobs(args, out, cache)
    log.info("%d jobs; building %d candidate pools (cached ones are skipped)", len(jobs), len(pool_jobs))
    ensure_pools(pool_jobs, cache, args.workers)

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, res in enumerate(ex.map(run_job, jobs), 1):
            rows.extend(res)
            if i % 10 == 0 or i == len(jobs):
                log.info("%d/%d jobs done (%.0fs)", i, len(jobs), time.time() - t0)

    agg = aggregate_all(rows)
    main_agg = agg["main"]

    def pick(names):
        return {m: {s: main_agg[m][s] for s in names if s in main_agg[m]} for m in main_agg}

    save_json(out / "baseline.json", pick(BASELINES))
    save_json(out / "static.json", pick(STATIC))
    save_json(out / "remem.json", pick(REMEM))
    save_json(out / "ablation.json", pick(ABLATION))
    save_json(out / "new_features.json", pick(NEW))
    save_json(out / "sensitivity.json", {t: v for t, v in agg.items() if t != "main"})
    save_json(out / "curves.json", curves(rows))
    save_json(out / "growth.json", growth(rows))
    save_json(out / "calibration.json", calibration(rows))
    fa = merge_failures(rows, out)
    save_json(out / "failure_summary.json", {"counts": fa["counts"], "n_tasks": fa["n_tasks"]})

    # paired deltas for the headline claim
    paired = {}
    for mode in READER_MODES:
        for metric in ("task_success", "tokens_per_task", "context_precision", "gold_recall"):
            for a, b in (("C_remem", "A_vector"), ("C_remem", "B_static"), ("B_static", "A_vector")):
                fams = [f for f in HEADLINE_FAMILIES if f in {r["family"] for r in rows}]
                d, ci = paired_vs(rows, a, b, "main", mode, metric, fams)
                paired[f"{mode}|{metric}|{a} - {b}"] = {"delta": d, "ci95": ci}
    save_json(out / "paired_deltas.json", paired)

    from remem.eval import report
    report.write_all(out, agg, curves(rows), growth(rows), calibration(rows), fa, paired,
                     [f for f in list(HEADLINE_FAMILIES) + list(EXTRA_FAMILIES) if f in {r["family"] for r in rows}])
    log.info("done in %.0fs; results in %s", time.time() - t0, out)


if __name__ == "__main__":
    main()
