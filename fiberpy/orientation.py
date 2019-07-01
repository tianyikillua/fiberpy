import numpy as np
from scipy import integrate

from .closure import A4_linear, A4_quadratic, A4_hybrid, A4_orthotropic, A4_invariants, A4_exact
from .tensor import Mat2

Shear = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
UElongation = np.array([[1, 0, 0], [0, -0.5, 0], [0, 0, -0.5]])
PElongation = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])


def apply_Ax(v, A, x):
    """
    Compute the contraction of a 4th-order tensor (with minor symmetry)
    given in the principal frame on a symmetric 2nd-order tensor given in the global frame

    Args:
        v (ndarray of shape (3, 3)): Principal directions along its columns
        A (ndarray of shape (6, 6)): 4th-order tensor (using :py:func:`fiberpy.tensor.Mat4`)
        x (ndarray of shape (3, 3)): 2nd-order tensor

    Returns:
        ndarray of shape (3, 3): Result array
    """

    # Express D in the principal frame
    x_ = v.T @ x @ v
    x_ = Mat2(x_)

    # Compute Ax and transform back to the original basis
    res = A @ x_
    res = Mat2(res)
    return v @ res @ v.T


def fiber_orientation(a0, t, L, ci, ar, kappa=1.0, D3=None, closure="orthotropic", method="Radau", debug=False, **kwargs):
    """
    Compute fiber orientation tensor evolution using the Folgar-Tucker model
    as its variants

    Args:
        a0 (ndarray of shape (3, 3)): Initial fiber orientation tensor
        t (ndarray of shape (1, )): Time instants
        L (ndarray of shape (3, 3)): Velocity gradient
        ci (float): Interaction coefficient
        ar (float): Aspect ratio
        kappa (float): Reduction coefficient when using the RSC model, ``0 < kappa <= 1``
        D3 (ndarray of shape (3, )): Coefficients :math:`(D_1,D_2,D_3)` when using the MRD model
        closure (str): 4th-order fiber orientation closure model ``A4_*``, see :py:mod:`fiberpy.closure`
        method (str): Numerical method to integrate the IVP, see :py:func:`scipy.integrate.solve_ivp`
        debug (bool): Return instead the ``sol`` object and ``dadt``
    """
    D_ = 0.5 * (L + L.T)
    W_ = 0.5 * (L - L.T)
    lmbda = (ar ** 2 - 1) / (ar ** 2 + 1)
    gamma = np.sqrt(2) * np.linalg.norm(D_)
    a0_ = a0.flatten()

    # MRD model
    if D3 is not None:
        trD3 = np.sum(D3)
        D3 = np.diag(D3)
        kappa = 1.0

    def dadt(t, a):
        a_ = a.reshape((3, 3))
        e, v = np.linalg.eigh(a_)
        e = e[::-1]
        v = v[:, ::-1]
        A = eval("A4_" + closure + "(e)")
        if not np.isclose(kappa, 1):
            A[:3, :3] = kappa * A[:3, :3]
            A[:3, :3] += (1 - kappa) * np.diag(e)

        # Folgar-Tucker or RSC
        if D3 is None:
            diffusion_part = np.eye(3) - 3 * a_
        else:
            # MRD
            diffusion_part = v @ D3 @ v.T - trD3 * a_

        dadt_ = (
            W_ @ a_ - a_ @ W_
            + lmbda * (D_ @ a_ + a_ @ D_ - 2 * apply_Ax(v, A, D_))
            + 2 * kappa * ci * gamma * diffusion_part
        )
        return dadt_.flatten()

    sol = integrate.solve_ivp(
        dadt, (t[0], t[-1]), a0_, t_eval=t, method=method, **kwargs)

    if not debug:
        return sol.y
    else:
        return sol, dadt


def shear_steady_state(ci, ar, closure="orthotropic", a0_isotropic="3d"):
    """
    Fiber orientation at steady state under simple shear

    Args:
        ci (float): Interaction coefficient
        ar (float): Aspect ratio
        closure (str): 4th-order fiber orientation closure model ``A4_*``, see :py:mod:`fiberpy.closure`
        a0_isotropic (str): If ``3d``, using 3-d initial isotropic orientation; if ``2d``, using planar initial isotropic orientation
    """
    t = np.logspace(0, 4, 100)
    if a0_isotropic == "3d":
        a0 = np.eye(3) / 3
    else:
        a0 = np.diag(np.array([0.5, 0, 0.5]))
    a = fiber_orientation(a0, t, Shear, ci, ar, closure=closure)
    return a[:, -1]
