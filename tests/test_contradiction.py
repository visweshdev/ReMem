import pytest

from remem.config import ContradictionConfig
from remem.contradiction import ContradictionDetector
from remem.ledger import MemoryLedger
from remem.types import MemState

DAY = 86400.0


def _sim(embedder):
    return lambda a, b: sum(x * y for x, y in zip(embedder.embed_one(a), embedder.embed_one(b)))


def _det(embedder, **kw):
    led = MemoryLedger()
    cfg = ContradictionConfig(enabled=True, exclusive_attributes=("region",), **kw)
    return led, ContradictionDetector(cfg, led, _sim(embedder))


def _add(led, det, fid, text, t, attr="region", role="state", ent="payments"):
    rec = led.register(fact_id=fid, text=text, time=t, entities=(ent,), attribute_keys=(attr,), semantic_role=role)
    return det.on_ingest(rec)


def test_exclusive_attribute_newer_value_supersedes_old(embedder):
    led, det = _det(embedder)
    _add(led, det, "fact_1", "The payments service is hosted in region us-east-1.", 1 * DAY)
    conflicts = _add(led, det, "fact_2", "The payments service is hosted in region eu-west-1.", 20 * DAY)
    assert [(c.old_id, c.new_id, c.kind) for c in conflicts] == [("fact_1", "fact_2", "exclusive")]
    old = led.get("fact_1")
    assert old.state == MemState.INVALIDATED and old.valid_to == 20 * DAY and old.superseded_by == "fact_2"
    assert led.get("fact_2").supersedes == ["fact_1"]
    assert old.original_text                                    # not deleted
    assert not ContradictionDetector.valid_at(old, 25 * DAY)    # invalid now ...
    assert ContradictionDetector.valid_at(old, 10 * DAY)        # ... but valid for a historical query


def _pad(led, det, n=40):
    for i in range(n):
        _add(led, det, f"fact_pad{i}", f"The pad{i} service is hosted in region ap-south-{i % 3}.", 0.0, ent=f"pad{i}")


def test_restatement_is_not_a_conflict(embedder):
    led, det = _det(embedder)
    _pad(led, det)
    _add(led, det, "fact_1", "The payments service is hosted in region us-east-1.", 1 * DAY)
    assert _add(led, det, "fact_2", "Region for payments deployment: us-east-1.", 5 * DAY) == []


def test_multivalued_attribute_without_cue_is_not_superseded(embedder):
    led, det = _det(embedder)
    _add(led, det, "fact_1", "payments depends on redis.", 1 * DAY, attr="depends_on")
    assert _add(led, det, "fact_2", "payments depends on kafka.", 9 * DAY, attr="depends_on") == []
    assert led.get("fact_1").state == MemState.ACTIVE


def test_update_cue_supersedes_nonexclusive_attribute(embedder):
    led, det = _det(embedder)
    _add(led, det, "fact_1", "payments depends on redis.", 1 * DAY, attr="depends_on")
    c = _add(led, det, "fact_2", "payments migrated and now depends on kafka.", 9 * DAY, attr="depends_on")
    assert c and c[0].kind == "cue"


def test_non_state_roles_are_ignored(embedder):
    led, det = _det(embedder)
    _add(led, det, "fact_1", "The payments service is hosted in region us-east-1.", 1 * DAY)
    assert _add(led, det, "fact_2", "Earlier question: which region does payments deploy to? unresolved eu-west-1.", 3 * DAY, role="event") == []


def test_late_arriving_old_fact_is_the_one_invalidated(embedder):
    led, det = _det(embedder)
    _add(led, det, "fact_2", "The payments service is hosted in region eu-west-1.", 20 * DAY)
    c = _add(led, det, "fact_1", "The payments service is hosted in region us-east-1.", 1 * DAY)
    assert c and c[0].old_id == "fact_1" and led.get("fact_1").state == MemState.INVALIDATED


def test_outcome_blame_supersedes_even_without_schema(embedder):
    led = MemoryLedger()
    det = ContradictionDetector(ContradictionConfig(enabled=True, exclusive_attributes=()), led, _sim(embedder))
    _add(led, det, "fact_1", "The payments service is hosted in region us-east-1.", 1 * DAY)
    assert _add(led, det, "fact_2", "The payments service is hosted in region eu-west-1.", 20 * DAY) == []   # schema unknown
    c = det.on_failure_blame(led.get("fact_1"))
    assert c is not None and c.kind == "outcome" and led.get("fact_1").state == MemState.INVALIDATED


def test_confidence_decays_only_for_volatile_attributes(embedder):
    led, det = _det(embedder, conf_half_life_days=100.0)
    _add(led, det, "fact_1", "The payments service is hosted in region us-east-1.", 0.0)
    _add(led, det, "fact_2", "The payments service is hosted in region eu-west-1.", 1 * DAY)         # makes 'region' volatile
    _add(led, det, "fact_3", "payments owner is alice.", 0.0, attr="owner")
    det.decay(300 * DAY)
    assert led.get("fact_2").confidence < 0.2 + 0.2             # ~0.5**(299/100), floored at 0.3
    assert led.get("fact_3").confidence == 1.0
    assert led.get("fact_2").confidence >= 0.3
