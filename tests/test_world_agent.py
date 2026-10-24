import random

from remem.eval.agent import AgentConfig, SimAgent
from remem.eval.metrics import auc, brier_and_ece, mean_ci
from remem.eval.world import FAMILIES, make_world, parse_text


def test_world_is_deterministic_and_truth_consistent():
    w1, w2 = make_world("contradiction", 3), make_world("contradiction", 3)
    assert [f.text for f in w1.facts] == [f.text for f in w2.facts]
    assert [t.query for t in w1.tasks] == [t.query for t in w2.tasks]
    for t in w1.tasks:
        ref = t.as_of if t.as_of is not None else t.time
        for (s, a), gid in zip(t.slots, w1.gold_ids(t)):
            fact = next(f for f in w1.facts if f.fact_id == gid)
            assert fact.time <= ref and (fact.subject, fact.attr) == (s, a)
            # no later (but still earlier-than-ref) version exists
            assert not any(tm <= ref and tm > fact.time for tm, _, _ in w1.history[(s, a)])


def test_every_family_generates():
    for name in FAMILIES:
        w = make_world(name, 1)
        assert w.tasks and w.facts
        assert all(e.kind in ("ingest", "task") for e in w.events)
        times = [e.time for e in w.events]
        assert times == sorted(times)


def test_updates_actually_change_the_value():
    w = make_world("state_change", 2)
    changed = 0
    for hist in w.history.values():
        for (_, v1, _), (_, v2, _) in zip(hist, hist[1:]):
            assert v1 != v2
            changed += 1
    assert changed > 0


def test_parser_handles_paraphrase_update_compression_and_rejects_questions():
    w = make_world("distractor", 0)
    v = w.vocab
    s = next(iter(v.subjects))
    assert parse_text(f"{s} is hosted in region eu-west-1.", v) == (s, "region", "eu-west-1")
    assert parse_text(f"{s} migrated; it is now hosted in region ap-south-1.", v) == (s, "region", "ap-south-1")
    assert parse_text(f"{s} hosted region eu west 1", v) == (s, "region", "eu-west-1")   # compressed form
    assert parse_text(f"Which region does {s} deploy to?", v) is None
    assert parse_text(f"Earlier question: 'Which region does {s} deploy to?' still unresolved.", v) is None
    assert parse_text("Weekly infrastructure sync notes reviewed deploy region, ownership and API version.", v) is None


def test_agent_is_deterministic_and_counterfactual_replay_changes_only_context():
    w = make_world("distractor", 0)
    t = w.tasks[0]
    gold = w.gold_ids(t)[0]
    g = next(f for f in w.facts if f.fact_id == gold)
    items = [(gold, g.text, g.time)]
    ag = SimAgent(w, AgentConfig(mode="first", base_distract=0.0, per_irrelevant=0.0), seed=1)
    r1, r2 = ag.act(t, items), ag.act(t, items)
    assert r1.success and r1.trace.answer == r2.trace.answer
    without = ag.act(t, [])
    assert without.guessed_slots            # no memory -> guesses (may be right by chance)


def test_agent_modes_resolve_conflicts_differently():
    w = make_world("contradiction", 0)
    pair = next(k for k, h in w.history.items() if len(h) >= 2)
    (t_old, v_old, id_old), (t_new, v_new, id_new) = w.history[pair][0], w.history[pair][-1]
    by_id = {f.fact_id: f for f in w.facts}
    task = next((t for t in w.tasks if t.slots == (pair,) and t.time > t_new), None)
    if task is None:
        task = type(w.tasks[0])(0, t_new + 1, "q", pair[1], (pair,), None, "current", "")
    items = [(id_old, by_id[id_old].text, t_old), (id_new, by_id[id_new].text, t_new)]
    cfg = dict(base_distract=0.0, per_irrelevant=0.0)
    first = SimAgent(w, AgentConfig(mode="first", **cfg), 0).act(task, items)
    latest = SimAgent(w, AgentConfig(mode="latest", **cfg), 0).act(task, items)
    assert first.trace.answer[f"{pair[0]}.{pair[1]}"] == v_old
    assert latest.trace.answer[f"{pair[0]}.{pair[1]}"] == v_new


def test_metrics_auc_matches_bruteforce_and_ci():
    rng = random.Random(0)
    s = [rng.random() for _ in range(200)]
    y = [1 if rng.random() < x else 0 for x in s]
    pos = [a for a, b in zip(s, y) if b]
    neg = [a for a, b in zip(s, y) if not b]
    brute = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
    assert abs(auc(s, y) - brute) < 1e-12
    brier, ece = brier_and_ece([0.0, 1.0], [0, 1])
    assert brier == 0 and ece == 0
    m, ci = mean_ci([1.0, 2.0, 3.0])
    assert m == 2.0 and ci > 0
