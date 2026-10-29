"""Step 1 (phi optimisation) validation against tests/ground_truth/step1."""

import pytest

from tests.conftest import dataset_params, report
from tests.validation import metrics

pytestmark = pytest.mark.validation
STEP = 1


@pytest.mark.parametrize("entry", dataset_params(1))
def test_step1_matches_reference(entry, step1_run, reference, request):
    checks = metrics.compare_step1(step1_run, reference, tol=entry.get("tolerances"))
    report(request, 1, entry, step1_run, reference, checks)
