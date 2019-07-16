import numpy as np

from .closure import (
    A4_linear,
    A4_quadratic,
    A4_hybrid,
    A4_orthotropic,
    A4_invariants,
    A4_exact,
)
from .tensor import Mat4


class FiberComposite:
    """
    Class for computing homogenized properties of a fiber-reinforced
    composite

    Args:
        rve_data (dict): Dictionary defining the microstructure
    """

    def __init__(self, rve_data):
        self.rve_data = rve_data
        self.read_rve_data()
        self.UD = None  # unidirectional mechanical properties

    def get(self, variables):
        """
        Retrieve the RVE variables
        """
        return [self.rve_data[var] for var in variables]

    def read_rve_data(self):
        """
        Parse a RVE data defining the microstructure
        """
        self.E0, self.nu0 = self.get(["E0", "nu0"])
        self.E1, self.nu1 = self.get(["E1", "nu1"])
        self.ar = self.rve_data["aspect_ratio"]

        # Volume fraction of fibers
        if "vf" in self.rve_data:
            self.vf = self.rve_data["vf"]
        else:
            assert "mf" in self.rve_data
            mf = self.rve_data["mf"]
            assert "rho1" in self.rve_data
            rho1 = self.rve_data["rho1"]
            if "rho0" in self.rve_data:
                rho0 = self.rve_data["rho0"]
            else:
                assert "rho" in self.rve_data
                rho = self.rve_data["rho"]
                rho0 = (1 - mf) * rho * rho1 / (rho1 - rho * mf)
            self.vf = mf * rho0 / (mf * rho0 + (1 - mf) * rho1)

    def vBar(self, x0, x1):
        """
        Volume average
        """
        return x0 * (1 - self.vf) + x1 * self.vf

    def Eshelby(self):
        """
        Eshelby's tensor

        References:
            Tandon, G. P. & Weng, G. J. The effect of aspect ratio of inclusions on the elastic properties of unidirectionally aligned composites. Polymer Composites, Wiley Online Library, 1984, 5, 327-333
        """
        asq = self.ar ** 2
        asm1 = asq - 1
        hnu = 1 / (2 * (1 - self.nu0))
        q = self.ar / asm1 ** (3 / 2) * (self.ar * np.sqrt(asm1) - np.arccosh(self.ar))

        E = np.zeros((3, 3, 3, 3))
        E[0, 0, 0, 0] = hnu * (
            1
            - 2 * self.nu0
            + (3 * asq - 1) / asm1
            - (1 - 2 * self.nu0 + 3 * asq / asm1) * q
        )
        E[1, 1, 1, 1] = E[2, 2, 2, 2] = (
            0.75 * hnu * (asq / asm1)
            + 0.5 * hnu * (1 - 2 * self.nu0 - 9 / 4 / asm1) * q
        )
        E[1, 1, 2, 2] = E[2, 2, 1, 1] = (
            0.5 * hnu * (0.5 * asq / asm1 - (1 - 2 * self.nu0 + 0.75 / asm1) * q)
        )
        E[1, 1, 0, 0] = E[2, 2, 0, 0] = hnu * (
            -asq / asm1 + 0.5 * (3 * asq / asm1 - (1 - 2 * self.nu0)) * q
        )
        E[0, 0, 1, 1] = E[0, 0, 2, 2] = hnu * (
            2 * self.nu0 - 1 - 1 / asm1 + (1 - 2 * self.nu0 + 1.5 / asm1) * q
        )
        E[1, 2, 1, 2] = E[2, 1, 2, 1] = (
            0.5 * hnu * (0.5 * asq / asm1 + (1 - 2 * self.nu0 - 0.75 / asm1) * q)
        )
        E[0, 1, 0, 1] = E[0, 2, 0, 2] = (
            0.5
            * hnu
            * (
                1
                - 2 * self.nu0
                - (asq + 1) / asm1
                - 0.5 * (1 - 2 * self.nu0 - 3 * (asq + 1) / asm1) * q
            )
        )
        return E

    def MoriTanaka(self):
        r"""
        Elasticity tensor for a unidirectional RVE using the
        original Mori-Tanaka formulation

        Returns:
            array of shape (6, 6): Elasticity tensor using the :math:`(\phi, \phi)` bases
        """

        def H(E, C0, C1):
            """
            Concentration tensor H (= B according to the M-T model)
            """
            S0 = np.linalg.inv(C0)
            eye = np.eye(6)
            return np.linalg.inv(eye + E @ S0 @ (C1 - C0))

        C0 = AIsotropic(self.E0, self.nu0)
        C1 = AIsotropic(self.E1, self.nu1)
        eye = np.eye(6)
        E = Mat4(self.Eshelby())
        B = H(E, C0, C1)
        UD = (self.vf * C1 @ B + (1 - self.vf) * C0) @ (
            np.linalg.inv(self.vf * B + (1 - self.vf) * eye)
        )
        return UD

    def TandonWeng(self):
        r"""
        Elasticity tensor for a unidirectional RVE using
        Tandon-Weng's equations

        Returns:
            array of shape (6, 6): Elasticity tensor using the :math:`(\phi, \phi)` bases

        References:
            Tandon, G. P. & Weng, G. J. The effect of aspect ratio of inclusions on the elastic properties of unidirectionally aligned composites. Polymer Composites, Wiley Online Library, 1984, 5, 327-333
        """
        lmbda0, mu0 = lmbda_mu(self.E0, self.nu0)
        lmbda1, mu1 = lmbda_mu(self.E1, self.nu1)
        E = self.Eshelby()

        D1 = 1 + 2 * (mu1 - mu0) / (lmbda1 - lmbda0)
        D2 = (lmbda0 + 2 * mu0) / (lmbda1 - lmbda0)
        D3 = lmbda0 / (lmbda1 - lmbda0)
        B1 = (
            self.vf * D1 + D2 + (1 - self.vf) * (D1 * E[0, 0, 0, 0] + 2 * E[1, 1, 0, 0])
        )
        B2 = (
            self.vf
            + D3
            + (1 - self.vf) * (D1 * E[0, 0, 1, 1] + E[1, 1, 1, 1] + E[1, 1, 2, 2])
        )
        B3 = self.vf + D3 + (1 - self.vf) * (E[0, 0, 0, 0] + (1 + D1) * E[1, 1, 0, 0])
        B4 = (
            self.vf * D1
            + D2
            + (1 - self.vf) * (E[0, 0, 1, 1] + D1 * E[1, 1, 1, 1] + E[1, 1, 2, 2])
        )
        B5 = (
            self.vf
            + D3
            + (1 - self.vf) * (E[0, 0, 1, 1] + E[1, 1, 1, 1] + D1 * E[1, 1, 2, 2])
        )
        A1 = D1 * (B4 + B5) - 2 * B2
        A2 = (1 + D1) * B2 - (B4 + B5)
        A3 = B1 - D1 * B3
        A4 = (1 + D1) * B1 - 2 * B3
        A5 = (1 - D1) / (B4 - B5)
        A = 2 * B2 * B3 - B1 * (B4 + B5)

        E11 = self.E0 / (1 + self.vf * (A1 + 2 * self.nu0 * A2) / A)
        E22 = self.E0 / (
            1
            + self.vf
            * (-2 * self.nu0 * A3 + (1 - self.nu0) * A4 + (1 + self.nu0) * A5 * A)
            / (2 * A)
        )
        mu12 = mu0 * (
            1 + self.vf / (mu0 / (mu1 - mu0) + 2 * (1 - self.vf) * E[0, 1, 0, 1])
        )
        mu23 = mu0 * (
            1 + self.vf / (mu0 / (mu1 - mu0) + 2 * (1 - self.vf) * E[1, 2, 1, 2])
        )
        nu12 = (self.nu0 * A - self.vf * (A3 - self.nu0 * A4)) / (
            A + self.vf * (A1 + 2 * self.nu0 * A2)
        )
        nu23 = E22 / (2 * mu23) - 1

        S = np.zeros((6, 6))
        S[0, 0] = 1 / E11
        S[0, 1] = -nu12 / E11
        S[0, 2] = -nu12 / E11
        S[1, 1] = 1 / E22
        S[1, 2] = -nu23 / E22
        S[2, 2] = 1 / E22
        S[3, 3] = S[5, 5] = 1 / (2 * mu12)
        S[4, 4] = 1 / (2 * mu23)

        # Symmetrize the tensor
        for i in range(6):
            for j in range(i):
                S[i, j] = S[j, i]

        UD = np.linalg.inv(S)
        return UD

    def ABar(self, a, model="TandonWeng", closure="orthotropic", recompute_UD=False):
        r"""
        Homogenized elasticity tensor in the principal frame

        Args:
            a (array_like of shape (3,)): Principal values of the 2nd fiber orientation tensor, ``a[0] >= a[1] >= a[2]``
            model (str): Micromechanical model for the unidirectional RVE (``TandonWeng`` or ``MoriTanaka``)
            closure (str): 4th-order fiber orientation closure model ``A4_*``, see :py:mod:`fiberpy.closure`
            recompute_UD (bool): Whether force recomputing elastic properties of the unidirectional RVE

        Returns:
            array of shape (6, 6): Effective elasticity tensor using the :math:`(\phi_2, \phi)` bases

        References:
            Advani, S. G. & Tucker III, C. L. The use of tensors to describe and predict fiber orientation in short fiber composites. Journal of Rheology, SOR, 1987, 31, 751-784
        """

        # Perform UD computations
        if self.UD is None or recompute_UD:
            if model == "TandonWeng":
                self.UD = self.TandonWeng()
            elif model == "MoriTanaka":
                self.UD = self.MoriTanaka()

        # Constants from UD
        B1 = self.UD[0, 0] + self.UD[1, 1] - 2 * self.UD[0, 1] - 2 * self.UD[3, 3]
        B2 = self.UD[0, 1] - self.UD[1, 2]
        B3 = (self.UD[3, 3] + self.UD[1, 2] - self.UD[1, 1]) / 2
        B4 = self.UD[1, 2]
        B5 = (self.UD[1, 1] - self.UD[1, 2]) / 2
        eye = np.eye(3)

        # 4th-order orientation tensor
        assert a.shape == (3,)
        A4 = eval("A4_" + closure + "(a)")

        # Orientation averaging using the orientation tensor
        a = np.diag(a)
        A = (
            B1 * A4
            + B2 * Mat4(np.einsum("ij,kl", a, eye) + np.einsum("kl,ij", a, eye))
            + B3
            * Mat4(
                np.einsum("ik,jl", a, eye)
                + np.einsum("il,jk", a, eye)
                + np.einsum("jl,ik", a, eye)
                + np.einsum("jk,il", a, eye)
            )
            + B4 * Mat4(np.einsum("ij,kl", eye, eye))
            + B5 * Mat4(np.einsum("ik,jl", eye, eye) + np.einsum("il,jk", eye, eye))
        )
        A[3:, 3:] /= 2  # converting to the Voigt notation (2*eps12)
        return A

    def alphaBar(self, ABar):
        """
        Homogenized thermal expansion coefficients in the principal frame

        Args:
            ABar (array_like of shape (6, 6)): Elasticity tensor

        Returns:
            array of shape (3, 3): Effective thermal dilatation coefficient matrix

        References:
            Rosen, B. W. & Hashin, Z. Effective thermal expansion coefficients and specific heats of composite materials. International Journal of Engineering Science, Elsevier BV, 1970, 8, 157-173
        """
        alpha0, alpha1 = self.get(["alpha0", "alpha1"])

        K0 = bulk_modulus(self.E0, self.nu0)
        K1 = bulk_modulus(self.E1, self.nu1)
        invKBar = self.vBar(1 / K0, 1 / K1)
        alphaBar = self.vBar(alpha0, alpha1)

        SBar = np.diag(np.sum(np.linalg.inv(ABar)[:3, :3], axis=0))
        alphaBar = alphaBar * np.eye(3) + (
            (alpha0 - alpha1) / (1 / K0 - 1 / K1) * (3 * SBar - invKBar * np.eye(3))
        )
        return alphaBar


def lmbda_mu(E, nu):
    r"""
    Convert :math:`(E, \nu)` to :math:`(\lambda, \mu)`
    """
    lmbda = nu * E / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lmbda, mu


def bulk_modulus(E, nu):
    r"""
    Bulk modulus from :math:`(E, \nu)`
    """
    return E / (3 * (1 - 2 * nu))


def AIsotropic(E, nu):
    r"""
    Isotropic elasticity tensor given in the :math:`(\phi, \phi)` bases

    Args:
        E (float): Young's modulus
        nu (float): Poisson coefficient

    Returns:
        array of shape (6, 6): Elasticity tensor
    """
    lmbda, mu = lmbda_mu(E, nu)
    A = np.array(
        (
            [
                [lmbda + 2 * mu, lmbda, lmbda, 0, 0, 0],
                [lmbda, lmbda + 2 * mu, lmbda, 0, 0, 0],
                [lmbda, lmbda, lmbda + 2 * mu, 0, 0, 0],
                [0, 0, 0, 2 * mu, 0, 0],
                [0, 0, 0, 0, 2 * mu, 0],
                [0, 0, 0, 0, 0, 2 * mu],
            ]
        )
    )
    return A


def A2Eij(A):
    r"""
    Calculate the orthotropic moduli from an elasticity tensor
    written using the :math:`(\phi_2, \phi)` bases

    Args:
        A (array_like of shape (6, 6)): Elasticity tensor

    Returns:
        :math:`(E_1, E_2, E_3, \mu_{12}, \mu_{23}, \mu_{13}, \nu_{12}, \nu_{23}, \nu_{31})`
    """
    assert A.shape == (6, 6), "Elasticity tensor A is not 6 by 6"
    S = np.linalg.inv(A)
    E1 = 1 / S[0, 0]
    E2 = 1 / S[1, 1]
    E3 = 1 / S[2, 2]
    mu12 = 1 / S[3, 3]
    mu23 = 1 / S[4, 4]
    mu13 = 1 / S[5, 5]
    nu12 = -S[1, 0] * E1
    nu23 = -S[2, 1] * E2
    nu31 = -S[0, 2] * E3
    return E1, E2, E3, mu12, mu23, mu13, nu12, nu23, nu31
