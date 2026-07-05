"""Synthetic M1 path generators (Phase 0 correctness battery).

See :mod:`tests.synthetic.paths` for the generators. Re-exported here
so tests can do ``from tests.synthetic import all_cases``.
"""

from .paths import (  # noqa: F401
    SyntheticCase,
    all_cases,
    flat_timeout,
    late_tp_long,
    linear_sl_long,
    linear_tp_long,
    linear_tp_short,
    same_bar_ambiguity_long,
)
