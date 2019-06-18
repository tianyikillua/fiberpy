import numpy as np
from fiberpy.mechanics import FiberComposite, A2Eij


# RVE data for Moldflow A218V50
rve_data = {
    "rho0": 1.14e-9,
    "E0": 631.66,
    "nu0": 0.42925,
    "alpha0": 5.86e-5,
    "rho1": 2.55e-9,
    "E1": 72000,
    "nu1": 0.22,
    "alpha1": 5e-6,
    "mf": 0.5,
    "aspect_ratio": 17.983,
}
fiber = FiberComposite(rve_data)


def test_elastic_properties():
    A = A2Eij(fiber.TandonWeng())
    ref = (9469.855923148461, 1316.8115034771508, 1316.8115034771508,
           834.3455847760557, 787.828547991758, 834.3455847760557,
           0.4002931728382047, 0.6714442588223231, 0.05566195082949749)
    assert np.allclose(A, ref)


def test_thermal_properties():
    alpha = np.diag(fiber.alphaBar(fiber.TandonWeng()))
    ref = (8.29181165e-06, 5.44419385e-05, 5.44419385e-05)
    assert np.allclose(alpha, ref)
