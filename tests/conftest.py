import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from remem.eval.pools import make_embedder
from remem.offline import count_tokens
from remem.types import Candidate, ScoredCandidate


@pytest.fixture(scope="session")
def embedder():
    return make_embedder()


def mk_cand(fid, text="x y z", sim=0.5, rank=0, tokens=None, time=1.7e9):
    return Candidate(fact_id=fid, text=text, tokens=tokens or count_tokens(text), time=time, sim=sim, base_rank=rank)


def mk_scored(fid, prob, tokens=10, sim=0.5, text="x", rank=0):
    c = mk_cand(fid, text, sim, rank, tokens)
    return ScoredCandidate(cand=c, utility=prob, prob=prob, components={}, state="ACTIVE", text=text, tokens=tokens)
