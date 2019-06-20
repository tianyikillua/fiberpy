import numpy as np
import pytest
from fiberpy.closure import A4_linear, A4_quadratic, A4_hybrid, A4_orthotropic, A4_invariants, A4_exact

# Define test 2nd order orientation tensors
a_random = 1 / 3 * np.ones(3)
a_UD = np.array([1, 0, 0])


@pytest.mark.parametrize(
    "closure", [A4_hybrid, A4_orthotropic, A4_invariants, A4_exact])
def test_random(closure):
    # The linear closure is exact for the random orientation state
    A4 = A4_linear(a_random)
    assert np.allclose(A4, closure(a_random))


@pytest.mark.parametrize(
    "closure", [A4_hybrid, A4_orthotropic, A4_invariants, A4_exact])
def test_UD(closure):
    # The quadratic closure is exact for the UD orientation state
    A4 = A4_quadratic(a_UD)
    assert np.allclose(A4, closure(a_UD), atol=1e-3)
