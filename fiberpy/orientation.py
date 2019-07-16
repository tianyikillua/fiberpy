import numpy as np
from scipy import integrate, optimize

from .closure import (
    A4_linear,
    A4_quadratic,
    A4_hybrid,
    A4_orthotropic,
    A4_invariants,
    A4_exact,
)
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


def fiber_orientation(
    a0,
    t,
    L,
    ci,
    ar,
    kappa=1,
    D3=None,
    closure="orthotropic",
    method="RK45",
    debug=False,
    **kwargs
):
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
        kappa = 1

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
            W_ @ a_
            - a_ @ W_
            + lmbda * (D_ @ a_ + a_ @ D_ - 2 * apply_Ax(v, A, D_))
            + 2 * kappa * ci * gamma * diffusion_part
        )
        return dadt_.flatten()

    sol = integrate.solve_ivp(
        dadt, (t[0], t[-1]), a0_, t_eval=t, method=method, **kwargs
    )

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


class Icosphere:
    """
    Discretization of a sphere using subdivision of a icosahedron

    Args:
        radius (float): Radius of the sphere
        n_refinement (int): Number of subdivision of the icosahedron
    """

    def __init__(self, radius=1, n_refinement=5):
        # Initial icosahedron
        r = (1 + np.sqrt(5)) / 2
        points = np.array(
            [
                [-1, r, 0],
                [1, r, 0],
                [-1, -r, 0],
                [1, -r, 0],
                [0, -1, r],
                [0, 1, r],
                [0, -1, -r],
                [0, 1, -r],
                [r, 0, -1],
                [r, 0, 1],
                [-r, 0, -1],
                [-r, 0, 1],
            ]
        )
        triangles = np.array(
            [
                [0, 11, 5],
                [0, 5, 1],
                [0, 1, 7],
                [0, 7, 10],
                [0, 10, 11],
                [1, 5, 9],
                [5, 11, 4],
                [11, 10, 2],
                [10, 7, 6],
                [7, 1, 8],
                [3, 9, 4],
                [3, 4, 2],
                [3, 2, 6],
                [3, 6, 8],
                [3, 8, 9],
                [5, 4, 9],
                [2, 4, 11],
                [6, 2, 10],
                [8, 6, 7],
                [9, 8, 1],
            ],
            dtype=int,
        )
        for _ in range(n_refinement):
            triangles_ = np.empty((0, 3), dtype=int)
            for triangle in triangles:
                v0, v1, v2 = triangle
                coord_vertices = points[triangle]
                midpoints = np.array(
                    [
                        np.mean(coord_vertices[[0, 1], :], axis=0),
                        np.mean(coord_vertices[[1, 2], :], axis=0),
                        np.mean(coord_vertices[[2, 0], :], axis=0),
                    ]
                )
                n_points = len(points)
                v3, v4, v5 = n_points, n_points + 1, n_points + 2
                triangles_new = np.array(
                    [[v0, v3, v5], [v1, v4, v3], [v2, v5, v4], [v3, v4, v5]]
                )
                triangles_ = np.vstack([triangles_, triangles_new])
                points = np.vstack([points, midpoints])

            r_points = np.linalg.norm(points, axis=1)
            points *= radius / r_points[:, None]
            triangles = triangles_.copy()

        self.triangles = triangles
        self.points = points
        self.cells = {"triangle": triangles}
        self.point_data = {}
        self.cell_data = {}
        self.field_data = {}

    def equal_earth_projection(self):
        """
        Project node points from the sphere to 2-d plane using
        the `Equal Earch projection <https://en.wikipedia.org/wiki/Equal_Earth_projection>`_
        """
        theta = np.arccos(self.points[:, 2])
        phi = np.arctan2(self.points[:, 1], self.points[:, 0])
        latitude = np.pi / 2 - theta
        longitude = phi
        theta_ = np.arcsin(np.sqrt(3) / 2 * np.sin(latitude))
        A1 = 1.340264
        A2 = -0.081106
        A3 = 0.000893
        A4 = 0.003796
        x = 2 * np.sqrt(3) * longitude * np.cos(theta_) / (3 * (A1 + 3 * A2 * theta_**2 + theta_**6 * (7 * A3 + 9 * A4 * theta_**2)))
        y = theta_ * (A1 + A2 * theta_**2 + theta_**6 * (A3 + A4 * theta_**2))
        return x, y

    def centroid(self):
        """
        Centroid coordinates of cells
        """
        try:
            return self.centroid_
        except AttributeError:
            self.centroid_ = np.empty((len(self.triangles), 3))
            for i, triangle in enumerate(self.triangles):
                coord_vertices = self.points[triangle]
                self.centroid_[i] = np.mean(coord_vertices, axis=0)
            return self.centroid_

    def integrate(self, fun):
        """
        Integrate a function defined on the icosphere

        We assume that function is piecewisely constant in each cell

        Args:
            f (callable): Function depending on x
        """
        res = 0 * np.asarray(fun(np.array([1, 0, 0])), dtype=float)
        for i, triangle in enumerate(self.triangles):
            coord_vertices = self.points[triangle]
            v0, v1, v2 = coord_vertices[0], coord_vertices[1], coord_vertices[2]
            x1 = v1 - v0
            x2 = v2 - v0
            area = 0.5 * np.linalg.norm(np.cross(x1, x2))
            fun_centroid = fun(self.centroid()[i])
            res += fun_centroid * area
        return res


def distribution_function(a, n_refinement=5, return_mesh=False):
    r"""
    Reconstruct the orientation distribution function (ODF) from the
    2nd-order orientation tensor

    The ODF is assumed to follow the Bingham distribution. Its probability
    density function is proportional to

    .. math::

       \exp(\mathbf{x}^\mathsf{T}\mathbf{v}\mathbf{Z}\mathbf{v}^\mathsf{T}\mathbf{x})

    where :math:`\mathbf{v}` are the principal directions and :math:`\mathbf{Z}` is a
    diagonal matrix of trace 1

    Args:
        a (array_like of shape (3, 3) or (3,)): Fiber orientation tensor (or its principal values in decreasing order)
        n_refinement (int): Number of subdivision of the icosahedron
        return_mesh (bool): Also return the icosphere mesh containing the values on cells

    Returns:
        callable: Orientation distribution function ``odf(x)`` defined for normal vectors on the unit sphere
    """
    a = np.asarray(a)
    if a.ndim == 1:
        e = a
        v = np.eye(3)
    else:
        e, v = np.linalg.eigh(a)
        e = e[::-1]
        v = v[:, ::-1]

    icosphere = Icosphere(n_refinement=n_refinement)

    def Bingham(x, Z):
        x = np.asarray(x, dtype=float)
        if Z.ndim == 1:
            return np.exp(np.sum(Z * x * x, axis=-1))
        else:
            return np.exp(np.sum(x * (x @ Z.T), axis=-1))

    def fun(z0_z1):
        z0, z1 = z0_z1
        Z = np.array([z0, z1, 1 - z0 - z1])

        def Bingham_Z(x):
            return Bingham(x, Z)

        factor = icosphere.integrate(Bingham_Z)

        def moment_2nd(x):
            return Bingham_Z(x) * x**2 / factor

        res = icosphere.integrate(moment_2nd)[:2]
        return res - e[:2]

    x0 = np.array([1, 0])
    z0_z1 = optimize.root(fun, x0).x
    z0, z1 = z0_z1
    Z = np.diag([z0, z1, 1 - z0 - z1])
    Zv = v @ Z @ v.T

    def Bingham_Z(x):
        return Bingham(x, Zv)

    factor = icosphere.integrate(Bingham_Z)

    def Bingham_Z_(x):
        return Bingham_Z(x) / factor

    if not return_mesh:
        return Bingham_Z_
    else:
        icosphere.point_data = {"ODF (points)": Bingham_Z_(icosphere.points)}
        icosphere.cell_data["triangle"] = {"ODF (cells)": Bingham_Z_(icosphere.centroid())}
        return Bingham_Z_, icosphere
