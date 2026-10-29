"""Step 2 (well mapping) validation against tests/ground_truth/step2.

Each dataset is run once (in the orientation recorded in the manifest) and
scored on all of its checks; a failure lists every check that missed its
tolerance, not just the first.
"""

import pytest

from tests.conftest import dataset_params, report
from tests.validation import metrics

pytestmark = pytest.mark.validation
STEP = 2


@pytest.mark.parametrize("entry", dataset_params(2))
def test_step2_matches_reference(entry, step2_run, reference, request):
    checks = metrics.compare_step2(step2_run, reference, tol=entry.get("tolerances"))
    report(request, 2, entry, step2_run, reference, checks)
