"""Back-compat re-export — the harness now lives at ``triage.leakage``.

The shuffled-label harness is a reusable deliverable (the build prompt:
"it will be reused every phase going forward"), so it was promoted from
``tests/`` into the main package at :mod:`triage.leakage`. This file
keeps the old import path working.
"""

from triage.leakage import (  # noqa: F401
    ShuffledLabelResult,
    assert_no_leakage,
    shuffled_label_auc,
)
