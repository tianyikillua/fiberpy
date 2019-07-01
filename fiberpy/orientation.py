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


def a_RSC(a0, t, L, ci, kappa, ar, closure="orthotropic", method="Radau"):
    """
    Compute fiber orientation tensor evolution using the RSC model

    Args:
        a0 (ndarray of shape (3, 3)): Initial fiber orientation tensor
        t (ndarray of shape (1, )): Time instants
        L (ndarray of shape (3, 3)): Velocity gradient
        ci (float): Interaction coefficient
        kappa (float): Reduction coefficient, ``0 < kappa <= 1``
        ar (float): Aspect ratio
        closure (str): 4th-order fiber orientation closure model ``A4_*``, see :py:mod:`fiberpy.closure`
        method (str): Numerical method to integrate the IVP, see :py:func:`scipy.integrate.solve_ivp`
    """
    D_ = 0.5 * (L + L.T)
    W_ = 0.5 * (L - L.T)
    lmbda = (ar ** 2 - 1) / (ar ** 2 + 1)
    gamma = np.sqrt(2) * np.linalg.norm(D_)
    a0_ = a0.reshape(-1)

    def dadt(t, a):
        a_ = a.reshape((3, 3))
        e, v = np.linalg.eigh(a_)
        e = e[::-1]
        v = v[:, ::-1]
        A = eval("A4_" + closure + "(e)")
        A[:3, :3] = kappa * A[:3, :3]
        A[:3, :3] += (1 - kappa) * np.diag(e)
        dadt_ = (
            W_.dot(a_)
            - a_.dot(W_)
            + lmbda * (D_.dot(a_) + a_.dot(D_) - 2 * apply_Ax(v, A, D_))
            + 2 * kappa * ci * gamma * (np.eye(3) - 3 * a_)
        )
        return dadt_.reshape(-1)

    sol = integrate.solve_ivp(dadt, (t[0], t[-1]), a0_, t_eval=t, method=method)
    return sol.y


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
    kappa = 1
    a = a_RSC(a0, t, Shear, ci, kappa, ar, closure=closure)
    return a[:, -1]
