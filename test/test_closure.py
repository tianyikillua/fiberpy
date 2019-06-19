import numpy as np
from fiberpy.closure import A4_quadratic, A4_hybrid, A4_orthotropic

# Define a random 2nd order orientation tensor
a = np.random.rand(3, 3)
a = a + a.T
_, v = np.linalg.eigh(a)
e = np.array([0.7, 0.2, 0.1])
a = v @ np.diag(e) @ v.T
assert np.isclose(np.trace(a), 1)


def test_quadratic():
    return A4_quadratic(a)


def test_hybrid():
    return A4_hybrid(a)


def test_orthotropic():
    return A4_orthotropic(e)
