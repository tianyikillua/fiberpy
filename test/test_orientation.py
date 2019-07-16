import numpy as np
import pytest

from fiberpy.orientation import (Icosphere, Shear, distribution_function,
                                 fiber_orientation)


def test_FT_RSC_steady():
    ci = 1e-3
    ar = 25
    t = np.logspace(-1, 4, 100)
    a0 = np.eye(3) / 3

    a = fiber_orientation(a0, t, Shear, ci, ar)
    a11_FT = a[0, -1]

    kappa = 0.1
    a = fiber_orientation(a0, t, Shear, ci, ar, kappa)
    a11_RSC = a[0, -1]

    assert np.isclose(a11_FT, a11_RSC, rtol=1e-3)


@pytest.mark.parametrize(
    "fun_name, ref_value",
    [("constant", 4 * np.pi),
     ("isotropic_orientation", np.eye(3) / 3),
     ("normal", np.zeros(3))]
)
def test_Icosphere(fun_name, ref_value):
    icosphere = Icosphere(n_refinement=3)

    def fun_constant(x):
        return 1

    def fun_isotropic_orientation(x):
        return np.outer(x, x) / (4 * np.pi)

    def fun_normal(x):
        return x

    fun = eval("fun_" + fun_name)
    assert np.allclose(icosphere.integrate(fun), ref_value, rtol=1e-1)


def test_distribution_function():
    _, mesh = distribution_function(np.eye(3) / 3, n_refinement=3, return_mesh=True)
    assert np.isclose(mesh.point_data["ODF (points)"].mean(), 1 / (4 * np.pi), rtol=1e-2)
