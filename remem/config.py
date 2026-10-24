"""Typed configuration for ReMem systems and experiments (YAML-loadable)."""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_type_hints

import yaml

FEATURE_GROUPS = ("static", "history", "outcome", "contradiction")


@dataclass
class UtilityConfig:
    kind: str = "heuristic"              # heuristic | learned
    model_path: str | None = None        # JSON written by experiments/train_utility.py
    feature_groups: tuple[str, ...] = ("static", "history", "outcome", "contradiction")
    explore_c: float = 0.0               # exploration strength
    explore_mode: str = "ucb"            # ucb: + c*posterior_std (deterministic) | thompson: + c*(Beta sample - mean)
    prior_a: float = 1.0                 # Beta prior for contribution rate
    prior_b: float = 3.0                 # prior mean 0.25: unseen memories are mildly skeptical
    class_shrink: float = 4.0            # pseudo-count pulling per-class rate toward the global rate
    recency_tau_days: float = 30.0
    rank_weight: float = 0.0             # extra weight of base-rank in heuristic


@dataclass
class AttributionConfig:
    method: str = "signals"              # presence | signals | signals_loo
    credit_threshold: float = 0.5        # credit >= threshold counts as "contributed"
    w_cited: float = 0.5
    w_used: float = 0.5
    min_value_len: int = 1               # min distinctive tokens needed to claim value-overlap
    loo_max_items: int = 8               # cap on counterfactual replays per task
    loo_policy: str = "all"              # all | ambiguous: replay only memories where exactly one of cited/used holds
    ignore_outcome: bool = False         # ablation: credit usage signals as if every task succeeded; no blame


@dataclass
class ContradictionConfig:
    enabled: bool = False
    exclusive_attributes: tuple[str, ...] = ()   # attribute_keys with a single valid value
    update_cues: tuple[str, ...] = (
        "migrated", "changed", "switched", "replaced", "longer", "moved", "updated",
    )
    roles: tuple[str, ...] = ("state", "preference", "constraint")  # extractor semantic_role values that assert current state
    min_sim: float = 0.45                # cue path only: min text similarity for two facts to be "about the same thing"
    max_value_df: float = 0.2            # a token is value-bearing if it appears in <= this share of memories (or has a digit)
    min_corpus_for_df: int = 30          # below this many memories, document frequencies are unreliable: use all tokens
    conf_half_life_days: float = 0.0     # 0 disables confidence decay
    conf_floor: float = 0.3
    outcome_invalidation: bool = True    # a failed action that used M while a newer conflicting fact exists
    as_of_validity: bool = True          # honour valid_from/valid_to for historical queries
    learn_schema: bool = False           # learn which attributes are single-valued from outcome evidence
    schema_min_evidence: int = 2         # confirmations needed before an attribute is treated as single-valued
    schema_threshold: float = 0.6        # posterior (e+1)/(e+m+2) needed, e=confirmations, m=counter-evidence


@dataclass
class RetentionConfig:
    mode: str = "none"                   # none | frequency_prune | adaptive
    # adaptive: thresholds are on the Beta-posterior contribution rate and on *exposures*
    # (times the memory was actually shown). Low frequency alone never triggers a transition.
    min_exposures_low: int = 5
    low_rate: float = 0.15
    recover_rate: float = 0.30
    compress_exposures: int = 10
    compress_rate: float = 0.10
    archive_exposures: int = 16
    archive_rate: float = 0.06
    low_value_penalty: float = 0.5       # multiplicative rank-score penalty while LOW_VALUE / COMPRESSED
    min_class_evidence: int = 3          # a demotion only applies to a task class where the memory
                                         # has been shown >= this many times ("useless for deploy"
                                         # is not "useless for approvals")
    protect_credit: float = 3.0          # memories with >= this much total credit are never archived
    compress_max_df: float = 0.2
    # frequency_prune (control): archive anything retrieved < min_retrievals within `window` tasks
    prune_window: int = 150
    prune_min_retrievals: int = 2


@dataclass
class SelectionConfig:
    mode: str = "rank_pack"              # rank_pack | cost_aware
    top_k: int = 5
    budget_tokens: int | None = None
    min_items: int = 1
    p_floor: float = 0.12                # cost_aware: drop candidates with calibrated p below this
    cost_lambda: float = 0.0             # cost_aware: penalty per 100 tokens subtracted from p
    guard_top: int = 0                   # always serve the top-m most similar candidates unless there is evidence they are useless
    guard_min_evidence: int = 3          # exposures in this task class before 'useless' evidence counts
    guard_rate: float = 0.2              # class contribution rate below which a guarded candidate is dropped


@dataclass
class SystemConfig:
    name: str = "remem"
    backend: str = "vector"              # candidate generator; see remem/backends
    ranker: str = "learned"              # sim | static | heuristic | learned
    outcome_feedback: bool = True
    pool_size: int = 20
    utility: UtilityConfig = field(default_factory=UtilityConfig)
    attribution: AttributionConfig = field(default_factory=AttributionConfig)
    contradiction: ContradictionConfig = field(default_factory=ContradictionConfig)
    retention: RetentionConfig = field(default_factory=RetentionConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    collect_exploration: int = 0         # extra random pool items shown per task (trace collection only)


# ── dict <-> dataclass ──────────────────────────────────────────────────────────

def from_dict(cls: type, data: dict[str, Any]):
    """Recursively build dataclass `cls` from a dict; unknown keys raise (typos fail loudly)."""
    hints = get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown config keys {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        hint = hints[key]
        if dataclasses.is_dataclass(hint) and isinstance(value, dict):
            kwargs[key] = from_dict(hint, value)
        elif isinstance(value, list):
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def to_dict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, tuple):
        return [to_dict(x) for x in obj]
    return obj


def load_systems(path: str | Path) -> dict[str, SystemConfig]:
    """Load `systems:` from a YAML file. Each entry may `extends:` another entry."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    entries: dict[str, dict[str, Any]] = raw.get("systems", {})
    resolved: dict[str, dict[str, Any]] = {}

    def resolve(name: str, stack: tuple[str, ...] = ()) -> dict[str, Any]:
        if name in resolved:
            return resolved[name]
        if name in stack:
            raise ValueError(f"cyclic extends: {stack + (name,)}")
        entry = dict(entries[name])
        parent = entry.pop("extends", None)
        merged = deep_merge(resolve(parent, stack + (name,)), entry) if parent else entry
        resolved[name] = merged
        return merged

    out: dict[str, SystemConfig] = {}
    for name in entries:
        d = dict(resolve(name))
        d["name"] = name
        out[name] = from_dict(SystemConfig, d)
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
