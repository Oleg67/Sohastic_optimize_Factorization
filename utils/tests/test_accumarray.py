import numpy as np
from ..accumarray import step_count, step_indices


def test_step_count():
    assert step_count(np.array([1, 4, 2, 3, 4, 4, 2, 5])) == 7


def test_step_indices():
    chk = np.array([1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4])
    np.testing.assert_equal(step_indices(chk), [0, 5, 8, 10, len(chk)])
