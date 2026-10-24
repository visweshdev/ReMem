"""Seeded synthetic worlds for the controlled memory-utility tasks.

A world is a project/user history: facts that arrive over time ("ingest" events, weekly
sessions) interleaved with tasks ("task" events) that need some of those facts. Every task has a
hidden ground truth (the value that was true at the task's reference time) that is used *only*
by the environment and the metrics, never by the memory layer.

What the generator controls (see FAMILIES):
  traps       lexical look-alikes of a specific query ("Earlier question: '...' unresolved").
              They are the "more semantically similar but never useful" memories.
  hubs        generic notes that mention many attribute words and match every infra query.
  updates     newer values for an (entity, attribute) pair, with or without an explicit cue.
  critical    rarely-needed memories: 'dormant' (never retrieved before the one task that needs
              it) and 'shown' (retrieved as a distractor for other task classes first).
  verbosity   gold facts written long vs short, to make token budgets bite.
  recurrence  whether the same (entity, attribute) pairs recur. Per-memory outcome history can
              only help when pairs recur; the `distractor_norecur` family is the control that
              removes this.

The world also carries the vocabulary the simulated agent uses to "understand" text, and the
synonym lexicon for the offline embedder.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from typing import Any

from remem.offline import count_tokens, tokenize

DAY = 86400.0
T0 = 1_700_000_000.0
WEEK = 7 * DAY

SUBJECTS = [
    "payments", "billing", "search", "auth", "inventory", "checkout", "catalog", "shipping", "ledger",
    "analytics", "messaging", "profiles", "ranking", "recs", "gateway", "scheduler", "reports", "audit",
    "media", "pricing", "support", "fraud", "notify", "quota", "storage", "sessions", "orders", "returns",
]
PEOPLE = ["alice", "bob", "carol", "dave", "erin", "frank", "grace", "heidi", "ivan", "judy"]
REGIONS = ["us-east-1", "eu-west-1", "ap-south-1", "us-west-2", "eu-central-1", "sa-east-1"]
API_VERSIONS = ["v1", "v2", "v3", "v4"]


@dataclass(frozen=True)
class AttrSpec:
    name: str
    pool: tuple[str, ...]
    exclusive: bool
    cues: tuple[str, ...]                    # tokens that identify the attribute in free text
    fact_templates: tuple[str, ...]
    update_templates: tuple[str, ...]        # explicit-cue updates
    query_templates: tuple[str, ...]
    verbose_templates: tuple[str, ...] = ()


ATTRS: dict[str, AttrSpec] = {
    "region": AttrSpec(
        "region", tuple(REGIONS), True, ("region", "zone"),
        ("{s} service is hosted in region {v}.", "The {s} service deploys to region {v}.", "Region for {s} deployment: {v}."),
        ("{s} migrated; it is now hosted in region {v}.", "{s} deployment changed and now runs in region {v}."),
        ("Which region does {s} deploy to?", "What is the deployment region for {s}?", "Tell me the region where {s} is hosted."),
        ("During the quarterly infrastructure planning review attended by platform leads, finance partners and the reliability group, "
         "it was confirmed after long discussion about latency and cost that the {s} service is hosted in region {v} for now.",),
    ),
    "owner": AttrSpec(
        "owner", tuple(PEOPLE), True, ("owner", "owned", "owns", "responsible"),
        ("{s} is owned by {v}.", "The owner of {s} is {v}.", "{v} is responsible for {s}."),
        ("Ownership of {s} changed; it is now owned by {v}.", "{s} was handed over and is now owned by {v}."),
        ("Who owns {s}?", "Which person is the owner of {s}?", "Who is responsible for {s}?"),
        ("At the end of the long reorganisation meeting covering headcount, budgets and priorities across all groups, "
         "it was noted that {s} is owned by {v} going forward.",),
    ),
    "api_version": AttrSpec(
        "api_version", tuple(API_VERSIONS), True, ("api", "version", "release"),
        ("{s} exposes API version {v}.", "The {s} service serves API {v}.", "API release for {s}: {v}."),
        ("{s} migrated to a new interface and now exposes API version {v}.", "{s} switched; the API version is now {v}."),
        ("Which API version does {s} expose?", "What version of the API is {s} running?", "Which API release does {s} serve?"),
        ("Following the long compatibility discussion with client teams about deprecations, timelines and migration tooling, "
         "the group agreed that {s} exposes API version {v} at present.",),
    ),
    "oncall": AttrSpec(
        "oncall", tuple(PEOPLE), True, ("call", "oncall", "pager"),
        ("The on-call engineer for {s} is {v}.", "{v} holds the pager for {s}.", "{s} on-call rotation: {v}."),
        ("The rotation changed; the on-call engineer for {s} is now {v}.", "{s} pager was handed over and is now held by {v}."),
        ("Who is on call for {s}?", "Who holds the pager for {s}?", "Who is the on-call engineer for {s}?"),
    ),
    # rarely needed attribute used by the rare/critical family
    "approver": AttrSpec(
        "approver", tuple(PEOPLE), True, ("approver", "approves", "approval"),
        ("The backup approver for {s} is {v}.", "{v} is the backup approver covering {s}."),
        ("The backup approver for {s} changed and is now {v}.",),
        ("Who is the backup approver for {s}?", "Which person approves {s} when the primary is away?"),
    ),
}

#: multi-valued attribute used only as non-conflicting noise (must never be superseded)
DEPENDS_POOL = ["redis", "kafka", "postgres", "etcd", "s3", "memcached"]

FILLER_SUBJECTS = ["offsite", "badge", "parking", "lunch", "holiday", "survey", "laptop", "keyboard", "swag", "newsletter"]
FILLER_PREDICATES = [
    "was rescheduled to thursday afternoon", "got a new vendor next quarter", "will be announced in the all-hands",
    "needs feedback from everyone by friday", "was praised in the retrospective", "ran out again last week",
    "is on hold until the budget review", "was moved to the third floor", "will be reviewed by the facilities team",
]

LEXICON: dict[str, str] = {}
for _group, _words in {
    "cmp_deploy": ["deploy", "deploys", "deployed", "deployment", "hosted", "hosting", "runs", "running"],
    "cmp_region": ["region", "regions", "zone"],
    "cmp_owner": ["owner", "owners", "owns", "owned", "responsible", "ownership"],
    "cmp_api": ["api", "apis", "interface"],
    "cmp_version": ["version", "release"],
    "cmp_oncall": ["oncall", "call", "pager", "rotation"],
    "cmp_approver": ["approver", "approves", "approval"],
}.items():
    for _w in _words:
        LEXICON[_w] = _group


@dataclass
class Vocab:
    subjects: tuple[str, ...]
    attrs: dict[str, AttrSpec]

    def exclusive_attrs(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.attrs.values() if a.exclusive)


# ── parsing (the agent's "reading comprehension") ──────────────────────────────

def parse_text(text: str, vocab: Vocab) -> tuple[str, str, str] | None:
    """Return (subject, attribute, value) if `text` asserts exactly one attribute value for one entity.

    Content-based, template-independent: works on paraphrases, update phrasings and the
    stopword-stripped form produced by compression. Returns None for questions, hubs and
    anything ambiguous.
    """
    toks = tokenize(text)
    joined = " " + " ".join(toks) + " "
    subs = [s for s in vocab.subjects if s in toks]
    if len(subs) != 1:
        return None
    hit_attrs = [a for a, sp in vocab.attrs.items() if any(c in toks for c in sp.cues)]
    # an "api version" statement matches both cue tokens of one attribute; different attributes must not both match
    if len(hit_attrs) != 1:
        return None
    attr = hit_attrs[0]
    values = {v for v in vocab.attrs[attr].pool if (" " + " ".join(tokenize(v)) + " ") in joined}
    if len(values) != 1:
        return None
    if "?" in text:
        return None
    return subs[0], attr, next(iter(values))


# ── dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class WorldFact:
    fact_id: str
    text: str
    time: float
    subject: str
    attr: str
    value: str | None
    kind: str                       # gold update trap hub filler critical depends
    role: str                       # semantic_role as an LLM fact extractor would label it
    entities: tuple[str, ...]
    attribute_keys: tuple[str, ...]
    session: str


@dataclass
class Task:
    task_id: int
    time: float
    query: str
    task_class: str
    slots: tuple[tuple[str, str], ...]      # required (subject, attribute) pairs
    as_of: float | None = None
    kind: str = "current"                   # current | historical | critical_dormant | critical_shown
    pair_key: str = ""


@dataclass
class Event:
    time: float
    kind: str                       # ingest | task
    facts: list[WorldFact] = field(default_factory=list)
    task: Task | None = None
    session: str = ""


@dataclass
class World:
    family: str
    seed: int
    vocab: Vocab
    facts: list[WorldFact]
    events: list[Event]
    tasks: list[Task]
    history: dict[tuple[str, str], list[tuple[float, str, str]]]   # (s,a) -> [(time, value, fact_id)]
    params: dict[str, Any]

    def truth(self, s: str, a: str, ref: float) -> tuple[str, str]:
        """(value, fact_id) true at reference time `ref`."""
        best = None
        for t, v, fid in self.history[(s, a)]:
            if t <= ref:
                best = (v, fid)
        if best is None:
            raise KeyError(f"no fact for {(s, a)} at {ref}")
        return best

    def gold_ids(self, task: Task) -> list[str]:
        ref = task.as_of if task.as_of is not None else task.time
        return [self.truth(s, a, ref)[1] for s, a in task.slots]

    def answer(self, task: Task) -> dict[str, str]:
        ref = task.as_of if task.as_of is not None else task.time
        return {f"{s}.{a}": self.truth(s, a, ref)[0] for s, a in task.slots}


# ── family parameters ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FamilyParams:
    name: str
    n_subjects: int = 24
    attrs: tuple[str, ...] = ("region", "owner", "api_version")
    n_hot: int = 12
    n_tasks: int = 300
    horizon_days: int = 300
    first_task_day: int = 40
    zipf_s: float = 1.0
    recurrence: bool = True
    trap_rate: float = 0.0
    traps_per_pair: int = 1
    hubs: int = 0
    filler: int = 80
    update_rate: float = 0.0
    update_cue_rate: float = 0.5
    max_updates: int = 1
    update_window: tuple[float, float] = (0.2, 0.8)
    historical_frac: float = 0.0
    multi_slot_frac: float = 0.15
    critical_dormant: int = 0
    critical_shown: int = 0
    critical_after: float = 0.85
    verbose_gold_frac: float = 0.0
    depends_noise: int = 0
    budget_tokens: int | None = None
    top_k: int = 5
    extraction_noise: float = 0.0


FAMILIES: dict[str, FamilyParams] = {
    # Task 1 – useful vs irrelevant memories among many distractors
    "distractor": FamilyParams("distractor", trap_rate=0.8, traps_per_pair=2, filler=120),
    # control: same distractors, but no pair ever recurs, so per-memory history cannot help
    "distractor_norecur": FamilyParams("distractor_norecur", recurrence=False, n_tasks=70, trap_rate=0.8,
                                       traps_per_pair=2, filler=120, multi_slot_frac=0.0, n_subjects=24),
    # Task 2 – needed facts are old; many interactions and fresh look-alikes in between
    "long_horizon": FamilyParams("long_horizon", n_tasks=320, horizon_days=400, first_task_day=120,
                                 trap_rate=0.3, filler=300, depends_noise=0),
    # Task 3 – old value conflicts with newer value
    "contradiction": FamilyParams("contradiction", update_rate=0.7, update_cue_rate=0.5, trap_rate=0.2,
                                  depends_noise=10),
    # Task 4 – rare but critical memories
    "rare_critical": FamilyParams("rare_critical", n_tasks=300, trap_rate=0.2, critical_dormant=8, critical_shown=8,
                                  attrs=("region", "owner", "api_version", "oncall")),
    # Task 5 – frequently retrieved but useless (hub notes + traps on every hot query)
    "useless_frequent": FamilyParams("useless_frequent", trap_rate=1.0, traps_per_pair=1, hubs=3, filler=60),
    # Task 6 – strict token budgets; gold facts partly verbose
    "budget_tight": FamilyParams("budget_tight", trap_rate=0.6, traps_per_pair=2, verbose_gold_frac=0.5,
                                 budget_tokens=45, top_k=6),
    "budget_medium": FamilyParams("budget_medium", trap_rate=0.6, traps_per_pair=2, verbose_gold_frac=0.5,
                                  budget_tokens=75, top_k=6),
    # Task 7 – changing state with historical (as_of) queries
    "state_change": FamilyParams("state_change", update_rate=0.8, max_updates=3, update_cue_rate=0.4,
                                 historical_frac=0.2, trap_rate=0.2, depends_noise=10),
}

#: families aggregated in the headline numbers
CORE_FAMILIES = (
    "distractor", "long_horizon", "contradiction", "rare_critical", "useless_frequent", "budget_tight", "state_change",
)


# ── generator ──────────────────────────────────────────────────────────────────

def make_world(family: str | FamilyParams, seed: int, overrides: dict[str, Any] | None = None) -> World:
    p = FAMILIES[family] if isinstance(family, str) else family
    if overrides:
        p = replace(p, **overrides)
    rng = random.Random(f"{p.name}:{seed}")
    subjects = tuple(rng.sample(SUBJECTS, p.n_subjects))
    attr_names = tuple(dict.fromkeys(list(p.attrs) + (["approver"] if (p.critical_dormant or p.critical_shown) else [])))
    vocab = Vocab(subjects, {a: ATTRS[a] for a in attr_names})

    facts: list[WorldFact] = []
    history: dict[tuple[str, str], list[tuple[float, str, str]]] = {}
    counter = [0]

    def week_time(week: int) -> float:
        return T0 + week * WEEK

    def new_fact(text, time, subject, attr, value, kind, role, keys=True) -> WorldFact:
        counter[0] += 1
        ents = (subject,) if subject else ()
        akeys = (attr,) if (attr and keys) else ()
        # extraction noise: the extractor misses the attribute key for a fraction of facts
        if p.extraction_noise and kind in ("gold", "update", "critical") and rng.random() < p.extraction_noise:
            akeys = ()
        f = WorldFact(f"fact_{counter[0]:05d}", text, time, subject, attr, value, kind, role, ents, akeys,
                      f"s{int((time - T0) // WEEK)}")
        facts.append(f)
        return f

    horizon_weeks = p.horizon_days // 7
    base_weeks = max(1, int(0.1 * horizon_weeks))

    pairs = [(s, a) for s in subjects for a in p.attrs]

    def render_fact(spec: AttrSpec, s: str, v: str, verbose: bool) -> str:
        tmpls = spec.verbose_templates if (verbose and spec.verbose_templates) else spec.fact_templates
        return rng.choice(tmpls).format(s=s, v=v)

    # 1. gold facts: one per (subject, attribute), created in the first weeks
    for s, a in pairs:
        spec = ATTRS[a]
        v = rng.choice(spec.pool)
        verbose = rng.random() < p.verbose_gold_frac
        f = new_fact(render_fact(spec, s, v, verbose), week_time(rng.randrange(base_weeks)), s, a, v, "gold", "state")
        history.setdefault((s, a), []).append((f.time, v, f.fact_id))

    # 2. choose hot pairs
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    hot = shuffled[: p.n_hot] if p.recurrence else []

    # 3. traps: lexical look-alikes of specific hot queries (or of any pair when no recurrence)
    trap_pairs = hot if p.recurrence else pairs
    for (s, a) in trap_pairs:
        if rng.random() < p.trap_rate:
            spec = ATTRS[a]
            qs = rng.sample(spec.query_templates, k=min(p.traps_per_pair, len(spec.query_templates)))
            for q in qs:
                text = f"Earlier question: '{q.format(s=s)}' still unresolved."
                new_fact(text, week_time(rng.randrange(base_weeks)), s, a, None, "trap", "event")

    # 4. hubs
    hub_texts = [
        "Weekly infrastructure sync notes reviewed deploy region, ownership and API version across all services; no changes decided.",
        "Platform checklist: confirm hosted region, owner and API release for each service before the monthly review.",
        "Operations digest mentions deployment zone, who owns what and the API version, but records no concrete decisions.",
    ]
    for i in range(p.hubs):
        new_fact(hub_texts[i % len(hub_texts)], week_time(rng.randrange(base_weeks)), "", "", None, "hub", "event")

    # 5. updates (new values over time)
    update_weeks: dict[tuple[str, str], list[int]] = {}
    if p.update_rate > 0:
        lo, hi = int(p.update_window[0] * horizon_weeks), int(p.update_window[1] * horizon_weeks)
        for s, a in (hot if p.recurrence else pairs):
            if ATTRS[a].exclusive and rng.random() < p.update_rate:
                n_up = rng.randint(1, p.max_updates)
                weeks = sorted(rng.sample(range(max(lo, base_weeks + 1), max(hi, lo + n_up + 1)), n_up))
                update_weeks[(s, a)] = weeks
                spec = ATTRS[a]
                cur = history[(s, a)][-1][1]
                for w in weeks:
                    nv = rng.choice([x for x in spec.pool if x != cur])
                    cue = rng.random() < p.update_cue_rate
                    tmpl = rng.choice(spec.update_templates) if cue else rng.choice(spec.fact_templates)
                    f = new_fact(tmpl.format(s=s, v=nv), week_time(w), s, a, nv, "update", "state")
                    history[(s, a)].append((f.time, nv, f.fact_id))
                    cur = nv

    # 6. multi-valued noise that must never be superseded
    for _ in range(p.depends_noise):
        s = rng.choice(subjects)
        a_dep = rng.choice(DEPENDS_POOL)
        new_fact(f"{s} depends on {a_dep}.", week_time(rng.randrange(horizon_weeks)), s, "depends_on", a_dep, "depends", "state")

    # 7. critical rarely-needed memories
    crit_subjects = [s for s in subjects if s not in {x for x, _ in hot}] if p.recurrence else list(subjects)
    rng.shuffle(crit_subjects)
    crit_tasks: list[Task] = []
    n_crit = p.critical_dormant + p.critical_shown
    hot_subjects = sorted({x for x, _ in hot}) or list(subjects)
    for i in range(n_crit):
        dormant = i < p.critical_dormant
        if dormant:
            s = crit_subjects[i % len(crit_subjects)]
            v = rng.choice(ATTRS["approver"].pool)
            text = f"{v} is the backup approver covering {s}."
        else:
            s = hot_subjects[i % len(hot_subjects)]
            v = rng.choice(ATTRS["approver"].pool)
            # looks like an answer to the subject's frequent infra queries, but is only needed for approvals
            text = f"The backup approver for {s} deploys is {v}."
        f = new_fact(text, week_time(rng.randrange(base_weeks)), s, "approver", v, "critical", "state")
        history[(s, "approver")] = [(f.time, v, f.fact_id)]
        spec = ATTRS["approver"]
        q = rng.choice(spec.query_templates).format(s=s)
        t = T0 + (p.critical_after * p.horizon_days + rng.uniform(0, (1 - p.critical_after) * p.horizon_days * 0.95)) * DAY
        crit_tasks.append(Task(-1, t, q, "approver", ((s, "approver"),), None,
                               "critical_dormant" if dormant else "critical_shown", f"{s}|approver"))

    # 8. filler chatter
    for _ in range(p.filler):
        text = f"The {rng.choice(FILLER_SUBJECTS)} {rng.choice(FILLER_PREDICATES)}."
        new_fact(text, week_time(rng.randrange(horizon_weeks)), "", "", None, "filler", "event", keys=False)

    # 9. tasks
    tasks: list[Task] = []
    weights = [1.0 / ((i + 1) ** p.zipf_s) for i in range(len(hot))] if hot else []
    day_lo, day_hi = p.first_task_day, int(p.critical_after * p.horizon_days) if n_crit else p.horizon_days
    unique_pool = pairs[:] if not p.recurrence else []
    rng.shuffle(unique_pool)
    times = sorted(T0 + (rng.uniform(day_lo, day_hi)) * DAY for _ in range(p.n_tasks))
    for t in times:
        if p.recurrence:
            s, a = rng.choices(hot, weights=weights, k=1)[0]
        else:
            if not unique_pool:
                break
            s, a = unique_pool.pop()
        slots = [(s, a)]
        if rng.random() < p.multi_slot_frac:
            others = [x for x in (hot if p.recurrence else pairs) if x[0] == s and x[1] != a]
            if others:
                slots.append(rng.choice(others))
        slots = sorted(slots)
        as_of = None
        kind = "current"
        if p.historical_frac and rng.random() < p.historical_frac:
            # need a pair with a version boundary before t; choose as_of between two versions
            versions = history[slots[0]]
            if len(versions) >= 2:
                k = rng.randrange(len(versions) - 1)
                lo, hi = versions[k][0], min(versions[k + 1][0], t)
                if hi > lo + DAY:
                    as_of = rng.uniform(lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo))
                    kind = "historical"
                    slots = [slots[0]]
        parts = [rng.choice(ATTRS[sa[1]].query_templates).format(s=sa[0]) for sa in slots]
        q = parts[0] if len(parts) == 1 else parts[0].rstrip("?") + " and " + parts[1][0].lower() + parts[1][1:]
        if as_of is not None:
            q = q.rstrip("?") + f" as of day {int((as_of - T0) // DAY)}?"
        tasks.append(Task(-1, t, q, "+".join(sorted({sa[1] for sa in slots})), tuple(slots), as_of, kind,
                          "|".join(f"{x}.{y}" for x, y in slots)))
    tasks.extend(crit_tasks)
    tasks.sort(key=lambda x: (x.time, x.pair_key))
    tasks = [replace(t, task_id=i) for i, t in enumerate(tasks)]

    # 10. events: facts ingested at their (weekly) time; tasks interleaved
    events: list[Event] = []
    by_week: dict[int, list[WorldFact]] = {}
    for f in facts:
        by_week.setdefault(int((f.time - T0) // WEEK), []).append(f)
    for w in sorted(by_week):
        events.append(Event(week_time(w), "ingest", facts=sorted(by_week[w], key=lambda x: x.fact_id), session=f"s{w}"))
    for t in tasks:
        events.append(Event(t.time, "task", task=t))
    events.sort(key=lambda e: (e.time, 0 if e.kind == "ingest" else 1))

    params = {k: getattr(p, k) for k in p.__dataclass_fields__}
    return World(p.name, seed, vocab, facts, events, tasks, history, params)


def describe(world: World) -> dict[str, Any]:
    kinds: dict[str, int] = {}
    for f in world.facts:
        kinds[f.kind] = kinds.get(f.kind, 0) + 1
    return {
        "family": world.family, "seed": world.seed, "n_facts": len(world.facts), "n_tasks": len(world.tasks),
        "fact_kinds": kinds, "mean_tokens": sum(count_tokens(f.text) for f in world.facts) / max(1, len(world.facts)),
        "task_kinds": {k: sum(1 for t in world.tasks if t.kind == k) for k in {t.kind for t in world.tasks}},
    }
