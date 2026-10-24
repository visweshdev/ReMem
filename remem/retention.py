"""Adaptive retention: explicit, inspectable memory state transitions.

States:  ACTIVE -> LOW_VALUE -> COMPRESSED -> ARCHIVED     (usefulness evidence)
         any     -> INVALIDATED                             (contradiction; see contradiction.py)
         LOW_VALUE/COMPRESSED/ARCHIVED -> ACTIVE            (recovery / revival)

Design rules that distinguish this from frequency-based pruning:

  1. Transitions depend on the Beta-posterior *contribution rate when shown* and on the number of
     *exposures* (times shown inside a successful task). A memory that was almost never shown has
     no evidence either way and never moves. Low retrieval frequency alone never demotes.
  2. Nothing is deleted. COMPRESSED keeps `original_text` (reversible); ARCHIVED only changes
     eligibility. Every transition is appended to the memory's lifecycle log with its reason.
  3. Demotions are class-conditional at retrieval time (`effective_state`): being useless for
     'deploy' tasks does not hide a memory from 'approval' tasks it has never been shown in.
  4. Memories with substantial accumulated credit are protected from ARCHIVED.

`FrequencyPrune` is the *control* policy used in the evaluation to show what rule 1 avoids.
"""

from __future__ import annotations

from remem.config import RetentionConfig
from remem.ledger import MemoryLedger
from remem.offline import content_tokens
from remem.types import MemoryRecord, MemState


def compress_text(text: str, ledger: MemoryLedger, max_df: float) -> str:
    """Drop stopwords and corpus-common tokens, keep order. Always keeps >= 2 tokens.

    Deterministic extractive compression; a deployment could swap in an LLM summariser.
    """
    toks = content_tokens(text)
    kept = [t for t in toks if ledger.df_fraction(t) <= max_df]
    if len(kept) < 2:
        kept = toks[:]
    return " ".join(kept)


class RetentionPolicy:
    def __init__(self, cfg: RetentionConfig, ledger: MemoryLedger, prior_a: float, prior_b: float) -> None:
        self.cfg = cfg
        self.ledger = ledger
        self.a = prior_a
        self.b = prior_b

    # ── effective state used for ranking ───────────────────────────────────────
    def effective_state(self, rec: MemoryRecord, task_class: str) -> MemState:
        if self.cfg.mode != "adaptive":
            return MemState.ACTIVE if rec.state != MemState.INVALIDATED else rec.state
        if rec.state in (MemState.LOW_VALUE, MemState.COMPRESSED, MemState.ARCHIVED):
            cs = rec.classes.get(task_class)
            if cs is None or cs.n_ctx < self.cfg.min_class_evidence:
                return MemState.ACTIVE
        return rec.state

    def served_text(self, rec: MemoryRecord, eff: MemState) -> tuple[str, int]:
        """Text actually offered to the agent: compressed only when the effective state says so."""
        if eff == MemState.COMPRESSED:
            return rec.text, rec.tokens
        return rec.original_text, rec.original_tokens

    # ── transitions after feedback ─────────────────────────────────────────────
    def update(self, rec: MemoryRecord) -> None:
        if self.cfg.mode != "adaptive" or rec.state == MemState.INVALIDATED:
            return
        led, cfg = self.ledger, self.cfg
        n = led.exposures(rec)
        rate = led.contrib_rate(rec, self.a, self.b)
        why = f"exposures={n} rate={rate:.3f} credit={rec.n_credited:.1f}"
        s = rec.state
        if s == MemState.ACTIVE:
            if n >= cfg.min_exposures_low and rate < cfg.low_rate:
                led.set_state(rec, MemState.LOW_VALUE, why)
        elif s == MemState.LOW_VALUE:
            if rate >= cfg.recover_rate:
                led.set_state(rec, MemState.ACTIVE, "recovered: " + why)
            elif n >= cfg.compress_exposures and rate < cfg.compress_rate:
                self._compress(rec, why)
        elif s == MemState.COMPRESSED:
            if rate >= cfg.recover_rate:
                self._restore(rec, "recovered: " + why)
            elif (n >= cfg.archive_exposures and rate < cfg.archive_rate
                  and rec.n_credited < cfg.protect_credit):
                led.set_state(rec, MemState.ARCHIVED, why)
        elif s == MemState.ARCHIVED:
            if rate >= cfg.recover_rate:
                self._restore(rec, "revived: " + why)

    def on_credited(self, rec: MemoryRecord) -> None:
        """A credited use of a demoted memory revives it immediately (evidence beats history)."""
        if self.cfg.mode != "adaptive":
            return
        if rec.state in (MemState.COMPRESSED, MemState.ARCHIVED, MemState.LOW_VALUE):
            self._restore(rec, "credited after demotion")

    def _compress(self, rec: MemoryRecord, why: str) -> None:
        from remem.offline import count_tokens
        rec.text = compress_text(rec.original_text, self.ledger, self.cfg.compress_max_df)
        rec.tokens = count_tokens(rec.text)
        self.ledger.set_state(rec, MemState.COMPRESSED, f"{why}; tokens {rec.original_tokens}->{rec.tokens}")

    def _restore(self, rec: MemoryRecord, why: str) -> None:
        rec.text, rec.tokens = rec.original_text, rec.original_tokens
        self.ledger.set_state(rec, MemState.ACTIVE, why)

    # ── control policy ─────────────────────────────────────────────────────────
    def frequency_prune(self, step: int) -> None:
        """Control: archive any memory old enough whose retrieval *count* is low. Ignores outcomes."""
        cfg = self.cfg
        if cfg.mode != "frequency_prune":
            return
        for rec in self.ledger.records.values():
            if rec.state != MemState.ACTIVE:
                continue
            age = step - rec.created_step
            if age < cfg.prune_window:
                continue
            need = cfg.prune_min_retrievals * (age // cfg.prune_window)
            if rec.n_retrieved < need:
                self.ledger.set_state(rec, MemState.ARCHIVED, f"frequency prune: {rec.n_retrieved}<{need} retrievals in {age} tasks")
