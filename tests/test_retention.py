from remem.config import RetentionConfig
from remem.ledger import MemoryLedger
from remem.retention import RetentionPolicy, compress_text
from remem.types import MemState


def _setup(mode="adaptive", **kw):
    led = MemoryLedger()
    rec = led.register(fact_id="fact_1", text="The payments service is hosted in region eu-west-1.", time=1.0)
    for i in range(30):
        led.register(fact_id=f"fact_o{i}", text=f"The other service {i} is hosted in region eu-west-1.", time=1.0)
    return led, rec, RetentionPolicy(RetentionConfig(mode=mode, **kw), led, 1.0, 3.0)


def _shown_useless(led, rec, pol, n):
    for _ in range(n):
        rec.n_success_ctx += 1
        rec.n_wasted += 1
        led.class_stats(rec, "region").n_ctx += 1
        pol.update(rec)


def test_useless_memory_walks_down_the_state_machine_and_logs_reasons():
    led, rec, pol = _setup()
    seen = []
    for _ in range(20):
        _shown_useless(led, rec, pol, 1)
        if not seen or seen[-1] != rec.state:
            seen.append(rec.state)
    assert seen == [MemState.ACTIVE, MemState.LOW_VALUE, MemState.COMPRESSED, MemState.ARCHIVED]
    changes = [e for e in rec.events if e.kind == "STATE_CHANGE"]
    assert len(changes) == 3 and all("exposures=" in e.detail for e in changes)
    assert rec.original_text.startswith("The payments")     # nothing is deleted


def test_rarely_shown_memory_is_never_demoted():
    led, rec, pol = _setup()
    _shown_useless(led, rec, pol, 4)                         # below min_exposures_low
    assert rec.state == MemState.ACTIVE
    for _ in range(1000):                                    # age alone changes nothing
        led.tick()
        pol.update(rec)
    assert rec.state == MemState.ACTIVE


def test_credit_revives_and_protects():
    led, rec, pol = _setup()
    _shown_useless(led, rec, pol, 12)
    assert rec.state == MemState.COMPRESSED and rec.tokens < rec.original_tokens
    rec.n_credited += 1
    rec.n_success_ctx += 1
    pol.on_credited(rec)
    assert rec.state == MemState.ACTIVE and rec.tokens == rec.original_tokens
    led2, rec2, pol2 = _setup()
    rec2.n_credited = 5.0
    _shown_useless(led2, rec2, pol2, 60)
    assert rec2.state != MemState.ARCHIVED                   # protected by accumulated credit


def test_demotion_is_class_conditional():
    led, rec, pol = _setup()
    _shown_useless(led, rec, pol, 12)
    assert rec.state == MemState.COMPRESSED
    assert pol.effective_state(rec, "region") == MemState.COMPRESSED
    assert pol.effective_state(rec, "approver") == MemState.ACTIVE      # never shown for that class


def test_frequency_prune_control_ignores_outcomes():
    led, rec, pol = _setup("frequency_prune", prune_window=10, prune_min_retrievals=2)
    led.step = 25
    pol.frequency_prune(25)
    assert rec.state == MemState.ARCHIVED and "frequency prune" in rec.events[-1].detail


def test_compress_keeps_value_tokens():
    led, rec, pol = _setup()
    out = compress_text(rec.original_text, led, 0.2)
    assert "eu" in out.split() and "payments" in out.split()
