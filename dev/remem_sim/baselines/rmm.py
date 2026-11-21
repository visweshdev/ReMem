"""RMM-style reranker: a reimplementation of the "Retrospective Reflection" reranker in
Tan et al., *In Prospect and Retrospect: Reflective Memory Management for Long-term Personalized Dialogue
Agents* (arXiv 2503.08026), §6.1–6.3. This is NOT the authors' code; it follows the paper's description:

  * embedding adaptation with residual linear layers:   q' = q + W_q q,   m_i' = m_i + W_m m_i
  * relevance s_i = q'·m_i'; selection of top-M by Gumbel-perturbed scores (exploration)
  * reward per selected memory: +1 if the generator cited it, −1 otherwise (task outcome is NOT used)
  * REINFORCE update  Δφ = η (R − b) ∇φ log P(selected | q, candidates; φ)

Approximation (documented): log P(selected) is taken as Σ_i log softmax(s/τ)_i over the selected items, with each
item's own reward, i.e. a per-item policy gradient, rather than the exact probability of the sampled top-M set.
τ and η were chosen on dev seeds (see experiments/tune_rmm.py); b = 0.
RMM's other half ("Prospective Reflection", multi-granularity summarisation) is not reimplemented.
"""

from __future__ import annotations

import math
import random

import numpy as np


class RMMReranker:
    def __init__(self, dim: int, *, lr: float = 1e-2, tau: float = 0.05, baseline: float = 0.0, seed: int = 0) -> None:
        self.dim, self.lr, self.tau, self.b = dim, lr, tau, baseline
        self.Wq = np.zeros((dim, dim))
        self.Wm = np.zeros((dim, dim))
        self.rng = random.Random(f"rmm:{seed}")
        self.updates = 0

    def scores(self, q: np.ndarray, M: np.ndarray) -> np.ndarray:
        qa = q + self.Wq @ q
        Ma = M + M @ self.Wm.T
        return Ma @ qa

    def sample_order(self, s: np.ndarray) -> list[int]:
        """Indices ordered by Gumbel-perturbed score (descending): sampling without replacement."""
        g = [-math.log(-math.log(max(1e-12, self.rng.random()))) for _ in range(len(s))]
        return sorted(range(len(s)), key=lambda i: -(s[i] / self.tau + g[i]))

    def update(self, q: np.ndarray, M: np.ndarray, selected: list[int], rewards: list[float]) -> None:
        if not selected:
            return
        qa = q + self.Wq @ q
        Ma = M + M @ self.Wm.T
        s = Ma @ qa / self.tau
        p = np.exp(s - s.max())
        p /= p.sum()
        gWq = np.zeros_like(self.Wq)
        gWm = np.zeros_like(self.Wm)
        for i, r in zip(selected, rewards):
            # ∇ log p_i with respect to the logits z: e_i − p
            dz = -p.copy()
            dz[i] += 1.0
            dz *= (r - self.b) / self.tau
            # z_j = qa · Ma_j  →  dz/dqa = Σ_j dz_j Ma_j ; dz/dMa_j = dz_j qa
            d_qa = Ma.T @ dz
            d_Ma = np.outer(dz, qa)
            gWq += np.outer(d_qa, q)              # qa = q + Wq q
            gWm += d_Ma.T @ M                     # Ma_j = M_j + Wm M_j
        self.Wq += self.lr * gWq / len(selected)
        self.Wm += self.lr * gWm / len(selected)
        self.updates += 1
