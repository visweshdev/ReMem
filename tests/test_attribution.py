from remem.attribution import Attributor
from remem.config import AttributionConfig
from remem.types import AgentTrace
from tests.conftest import mk_scored


def _ctx():
    return [
        mk_scored("a", 0.9, text="payments is hosted in eu-west-1"),
        mk_scored("b", 0.8, text="billing is hosted in us-east-1"),
        mk_scored("c", 0.7, text="weekly sync notes"),
    ]


def test_presence_gives_everything_equal_credit():
    atts = Attributor(AttributionConfig(method="presence")).attribute(_ctx(), AgentTrace({"x": "eu-west-1"}, [], {}), True)
    assert {a.credit for a in atts.values()} == {1.0}


def test_signals_credit_only_memories_the_action_depended_on():
    tr = AgentTrace({"payments.region": "eu-west-1"}, ["a", "c"], {"payments.region": "eu-west-1"})
    atts = Attributor(AttributionConfig(method="signals")).attribute(_ctx(), tr, True)
    assert atts["a"].credit == 1.0                              # cited and value used
    assert atts["b"].credit == 0.0                              # neither
    assert atts["c"].credit == 0.5 and not atts["c"].used       # spurious citation only


def test_failure_blames_memories_whose_value_was_used():
    tr = AgentTrace({"payments.region": "us-east-1"}, [], {"payments.region": "us-east-1"})
    atts = Attributor(AttributionConfig(method="signals")).attribute(_ctx(), tr, False)
    assert atts["b"].blame == 1.0 and atts["a"].blame == 0.0
    assert all(a.credit == 0 for a in atts.values())


def test_loo_upgrades_necessary_and_discounts_redundant():
    tr = AgentTrace({"k": "eu-west-1"}, ["a", "r"], {"k": "eu-west-1"})
    ctx = [mk_scored("a", 0.9, text="payments eu-west-1"), mk_scored("r", 0.8, text="payments eu-west-1 again")]
    replay = lambda subset: any(s.cand.fact_id == "a" for s in subset)   # noqa: E731  fails without 'a'
    at = Attributor(AttributionConfig(method="signals_loo"))
    atts = at.attribute(ctx, tr, True, replay)
    assert atts["a"].necessary is True and atts["a"].credit == 1.0
    assert atts["r"].necessary is False and atts["r"].credit < 0.5
    assert at.extra_agent_calls == 2
