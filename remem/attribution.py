"""Memory credit assignment.

Given the context an agent saw, its trace and the environment outcome, decide which
retrieved memories plausibly *contributed*. Nothing here sees ground truth.

Methods (`AttributionConfig.method`):

  presence      equal credit to every memory in context on success, equal blame on failure.
                The naïve baseline the brief warns against.
  signals       per-memory credit from two observable signals:
                  cited       the agent reported relying on the memory (self-reported, noisy)
                  used        the memory text contains a value the agent committed to in its
                              answer / tool call ("decision dependency")
                credit = w_cited*cited + w_used*used, only on success.
                blame  = used (on failure): the action depended on this memory and failed.
  signals_loo   `signals` plus a counterfactual replay: re-run the agent without memory m;
                if success flips to failure, m was necessary and gets credit 1.0. Memories
                that are not necessary keep only 40% of their signal credit (< the 0.5 contribution
                threshold): they may be redundant, or spurious citations. Caveat: two redundant copies
                of the same fact each look "unnecessary" when removed alone (redundancy masking).

Known limitations (also in the README): citations are self-reported; "used" is lexical and is
also true of a decoy that merely mentions the answer value; LOO assumes a replayable,
deterministic agent and costs one extra agent call per memory; none of this proves causality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from remem.config import AttributionConfig
from remem.offline import tokenize
from remem.types import AgentTrace, ScoredCandidate


#: share of signal credit kept by a memory that LOO shows to be unnecessary; must stay below the
#: contribution threshold (0.5) so that spurious/redundant credit stops counting as a contribution
LOO_DISCOUNT = 0.4


@dataclass
class Attribution:
    credit: float
    blame: float
    cited: bool
    used: bool
    necessary: bool | None = None


def _contains_value(text: str, values: list[str], min_len: int) -> bool:
    toks = tokenize(text)
    joined = " " + " ".join(toks) + " "
    for v in values:
        vt = tokenize(v)
        if len(vt) >= min_len and (" " + " ".join(vt) + " ") in joined:
            return True
    return False


class Attributor:
    def __init__(self, cfg: AttributionConfig) -> None:
        self.cfg = cfg
        self.extra_agent_calls = 0

    def attribute(
        self,
        selected: list[ScoredCandidate],
        trace: AgentTrace,
        success: bool,
        replay: Callable[[list[ScoredCandidate]], bool] | None = None,
    ) -> dict[str, Attribution]:
        cfg = self.cfg
        values = list(trace.answer.values()) + list(trace.tool_args.values())
        cited = set(trace.cited_ids)
        out: dict[str, Attribution] = {}

        for sc in selected:
            fid = sc.cand.fact_id
            if cfg.method == "presence":
                out[fid] = Attribution(credit=1.0 if success else 0.0, blame=0.0 if success else 1.0,
                                       cited=fid in cited, used=False)
                continue
            is_cited = fid in cited
            used = _contains_value(sc.text, values, cfg.min_value_len)
            if success:
                credit = cfg.w_cited * is_cited + cfg.w_used * used
                out[fid] = Attribution(credit=min(1.0, credit), blame=0.0, cited=is_cited, used=used)
            else:
                out[fid] = Attribution(credit=0.0, blame=1.0 if used else 0.0, cited=is_cited, used=used)

        if cfg.method == "signals_loo" and success and replay is not None:
            if cfg.loo_policy == "ambiguous":
                # cheap variant: only memories where exactly one of cited/used holds are uncertain enough to replay;
                # both signals => accepted as necessary without a replay, neither => no credit
                to_replay = [sc for sc in selected if 0.0 < out[sc.cand.fact_id].credit < 1.0][: cfg.loo_max_items]
            else:
                to_replay = sorted(selected, key=lambda s: -s.utility)[: cfg.loo_max_items]
            for sc in to_replay:
                rest = [s for s in selected if s.cand.fact_id != sc.cand.fact_id]
                still_ok = replay(rest)
                self.extra_agent_calls += 1
                att = out[sc.cand.fact_id]
                att.necessary = not still_ok
                att.credit = 1.0 if att.necessary else LOO_DISCOUNT * att.credit
            if cfg.loo_policy == "all":
                for sc in selected:
                    att = out[sc.cand.fact_id]
                    if att.necessary is None:   # beyond loo_max_items: shrink like non-necessary
                        att.credit = LOO_DISCOUNT * att.credit
        return out
