from fiberpy.orientation import fiber_orientation, Shear
import numpy as np


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
