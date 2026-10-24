import numpy as np

from remem.features import ALL_FEATURES, FEATURE_INDEX
from remem.utility import HeuristicUtility, LearnedUtility

ALL_GROUPS = ("static", "history", "outcome", "contradiction")


def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(ALL_FEATURES)))
    logit = 2.0 * X[:, FEATURE_INDEX["class_rate"]] - 1.0 * X[:, FEATURE_INDEX["waste_rate"]] + 0.3
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


def test_logistic_regression_recovers_signs_and_is_deterministic():
    X, y = _data()
    m1, rep = LearnedUtility.fit(X, y, ALL_GROUPS, l2=1.0)
    m2, _ = LearnedUtility.fit(X, y, ALL_GROUPS, l2=1.0)
    assert rep.converged
    w = dict(zip(m1.names, m1.w))
    assert w["class_rate"] > 0.5 and w["waste_rate"] < -0.2
    assert np.allclose(m1.w, m2.w)


def test_save_load_roundtrip_gives_same_scores(tmp_path):
    X, y = _data()
    m, _ = LearnedUtility.fit(X, y, ("static", "outcome"), l2=1.0)
    m.save(tmp_path / "m.json")
    m2 = LearnedUtility.load(tmp_path / "m.json")
    f = list(X[0])
    assert m.score(f)[1] == m2.score(f)[1]


def test_static_model_ignores_outcome_columns():
    X, y = _data()
    m, _ = LearnedUtility.fit(X, y, ("static",), l2=1.0)
    f, g = list(X[0]), list(X[0])
    g[FEATURE_INDEX["class_rate"]] += 5
    assert m.score(f)[1] == m.score(g)[1]


def test_heuristic_outcome_terms_move_score():
    h = HeuristicUtility(("static", "outcome"))
    base = [0.0] * len(ALL_FEATURES)
    base[FEATURE_INDEX["sim"]] = 0.6
    base[FEATURE_INDEX["class_rate"]] = 0.25
    good, bad = list(base), list(base)
    good[FEATURE_INDEX["class_rate"]] = 0.8
    bad[FEATURE_INDEX["waste_rate"]] = 1.0
    assert h.score(good)[0] > h.score(base)[0] > h.score(bad)[0]
    s = HeuristicUtility(("static",))
    assert s.score(good)[0] == s.score(base)[0]     # outcome terms inert when the group is off
