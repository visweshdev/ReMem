"""Evaluation protocol constants shared by training and benchmarking.

Seed discipline (to keep the reported numbers honest):
  TRAIN_SEEDS  worlds used to collect traces and fit the learned utility model
  DEV_SEEDS    worlds used while developing / choosing hyper-parameters
  TEST_SEEDS   worlds used for every reported number; never used for fitting or tuning
"""

from __future__ import annotations

TRAIN_SEEDS = (100, 101, 102, 103)
DEV_SEEDS = (200, 201)
TEST_SEEDS = (0, 1, 2, 3, 4)

#: families used to collect training traces (budget_tight covers the budget regime; test uses budget_tight too
#: but with different worlds)
TRAIN_FAMILIES = (
    "distractor", "long_horizon", "contradiction", "rare_critical", "useless_frequent", "budget_tight",
    "state_change", "distractor_norecur",
)

#: families aggregated into the headline table
HEADLINE_FAMILIES = (
    "distractor", "long_horizon", "contradiction", "rare_critical", "useless_frequent", "budget_tight", "state_change",
)
#: additional families reported separately
EXTRA_FAMILIES = ("distractor_norecur", "budget_medium")

READER_MODES = ("first", "latest")


def train_reader_mode(seed: int) -> str:
    """Training worlds alternate reader modes so the model is not tied to one reader."""
    return "first" if seed % 2 == 0 else "latest"
