"""The RMM-style baseline: exact gradients and the intended learning direction."""

import numpy as np

from remem.baselines.rmm import RMMReranker


def _setup(seed=0, d=12):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=d)
    M = rng.normal(size=(4, d))
    return rng, q / np.linalg.norm(q), M / np.linalg.norm(M, axis=1, keepdims=True)


def test_reinforce_gradient_matches_numerical_gradient():
    rng, q, M = _setup()
    r = RMMReranker(q.size, lr=1.0, tau=0.5)
    r.Wq, r.Wm = rng.normal(size=r.Wq.shape) * 0.1, rng.normal(size=r.Wm.shape) * 0.1
    Wq0, Wm0 = r.Wq.copy(), r.Wm.copy()

    def logp(Wq, Wm, i):
        z = (M + M @ Wm.T) @ (q + Wq @ q) / r.tau
        z -= z.max()
        return z[i] - np.log(np.exp(z).sum())

    r.update(q, M, [2], [1.0])                     # one step of size 1 along grad log p_2
    eps = 1e-6
    for W0, delta, which in ((Wq0, r.Wq - Wq0, "q"), (Wm0, r.Wm - Wm0, "m")):
        num = np.zeros_like(W0)
        for a in range(W0.shape[0]):
            for b in range(W0.shape[1]):
                E = np.zeros_like(W0)
                E[a, b] = eps
                hi = logp(Wq0 + E, Wm0, 2) if which == "q" else logp(Wq0, Wm0 + E, 2)
                lo = logp(Wq0 - E, Wm0, 2) if which == "q" else logp(Wq0, Wm0 - E, 2)
                num[a, b] = (hi - lo) / (2 * eps)
        assert np.abs(delta - num).max() < 1e-6


def test_cited_memory_gains_score_and_sampling_is_seeded():
    _, q, M = _setup(1)
    r = RMMReranker(q.size, lr=1e-2, tau=0.5)
    gap0 = np.subtract(*r.scores(q, M)[:2])
    for _ in range(50):
        r.update(q, M, [0, 1], [1.0, -1.0])
    assert np.subtract(*r.scores(q, M)[:2]) > gap0
    a, b = RMMReranker(q.size, seed=3), RMMReranker(q.size, seed=3)
    s = a.scores(q, M)
    assert a.sample_order(s) == b.sample_order(s)
