from itertools import permutations

import numpy as np

from .tensor import Mat4


def A4_linear(a):
    r"""
    Compute the linear closure

    Args:
        a (array_like of shape (3, 3)): Fiber orientation tensor

    Returns:
        array of shape (6, 6): 4th-order orientation tensor written using the :math:`(\phi,\phi)` bases
    """
    eye = np.eye(3)
    A_lin = -1 / 35 * (
        np.einsum("ij,kl", eye, eye)
        + np.einsum("ik,jl", eye, eye)
        + np.einsum("il,jk", eye, eye)
    ) + 1 / 7 * (
        np.einsum("ij,kl", eye, a)
        + np.einsum("ik,jl", eye, a)
        + np.einsum("il,jk", eye, a)
        + np.einsum("jk,il", eye, a)
        + np.einsum("jl,ik", eye, a)
        + np.einsum("kl,ij", eye, a)
    )
    return Mat4(A_lin)


def A4_quadratic(a):
    r"""
    Compute the quadratic closure

    Args:
        a (array_like of shape (3, 3)): Fiber orientation tensor

    Returns:
        array of shape (6, 6): 4th-order orientation tensor written using the :math:`(\phi,\phi)` bases
    """
    return Mat4(np.einsum("ij,kl", a, a))


def A4_hybrid(a):
    r"""
    Compute the hybrid closure

    Args:
        a (array_like of shape (3, 3)): Fiber orientation tensor

    Returns:
        array of shape (6, 6): 4th-order orientation tensor written using the :math:`(\phi,\phi)` bases
    """
    f = 1 - 27 * np.linalg.det(a)
    return (1 - f) * A4_linear(a) + f * A4_quadratic(a)


def A4_invariants(a):
    r"""
    Compute the IBOF closure

    Args:
        a (array_like of shape (3, 3)): Fiber orientation tensor

    Returns:
        array of shape (6, 6): 4th-order orientation tensor written using the :math:`(\phi,\phi)` bases
    """

    def symmetrize(a):
        S = np.zeros_like(a)
        perm = list(permutations(range(a.ndim)))
        for p in perm:
            S = S + np.transpose(a, p)
        S = S / len(perm)
        return S

    II = (
        a[0, 0] * a[1, 1]
        + a[0, 0] * a[2, 2]
        + a[1, 1] * a[2, 2]
        - a[0, 1] * a[1, 0]
        - a[0, 2] * a[2, 0]
        - a[1, 2] * a[2, 1]
    )
    III = np.linalg.det(a)

    x = np.array(
        [
            1,
            II,
            II ** 2,
            III,
            III ** 2,
            II * III,
            II ** 2 * III,
            II * III ** 2,
            II ** 3,
            III ** 3,
            II ** 3 * III,
            II ** 2 * III ** 2,
            II * III ** 3,
            II ** 4,
            III ** 4,
            II ** 4 * III,
            II ** 3 * III ** 2,
            II ** 2 * III ** 3,
            II * III ** 4,
            II ** 5,
            III ** 5,
        ]
    )
    a2 = np.array(
        [
            0.24940908165786e2,
            -0.435101153160329e3,
            0.372389335663877e4,
            0.703443657916476e4,
            0.823995187366106e6,
            -0.133931929894245e6,
            0.880683515327916e6,
            -0.991630690741981e7,
            -0.159392396237307e5,
            0.800970026849796e7,
            -0.237010458689252e7,
            0.379010599355267e8,
            -0.337010820273821e8,
            0.322219416256417e5,
            -0.257258805870567e9,
            0.214419090344474e7,
            -0.449275591851490e8,
            -0.213133920223355e8,
            0.157076702372204e10,
            -0.232153488525298e5,
            -0.395769398304473e10,
        ]
    )
    a3 = np.array(
        [
            -0.497217790110754,
            0.234980797511405e2,
            -0.391044251397838e3,
            0.153965820593506e3,
            0.152772950743819e6,
            -0.213755248785646e4,
            -0.400138947092812e4,
            -0.185949305922308e7,
            0.296004865275814e4,
            0.247717810054366e7,
            0.101013983339062e6,
            0.732341494213578e7,
            -0.147919027644202e8,
            -0.104092072189767e5,
            -0.635149929624336e8,
            -0.247435106210237e6,
            -0.902980378929272e7,
            0.724969796807399e7,
            0.487093452892595e9,
            0.138088690964946e5,
            -0.160162178614234e10,
        ]
    )
    a5 = np.array(
        [
            0.234146291570999e2,
            -0.412048043372534e3,
            0.319553200392089e4,
            0.573259594331015e4,
            -0.485212803064813e5,
            -0.605006113515592e5,
            -0.477173740017567e5,
            0.599066486689836e7,
            -0.110656935176569e5,
            -0.460543580680696e8,
            0.203042960322874e7,
            -0.556606156734835e8,
            0.567424911007837e9,
            0.128967058686204e5,
            -0.152752854956514e10,
            -0.499321746092534e7,
            0.132124828143333e9,
            -0.162359994620983e10,
            0.792526849882218e10,
            0.466767581292985e4,
            -0.128050778279459e11,
        ]
    )

    beta = np.zeros(6)
    beta[2] = a2.dot(x)
    beta[3] = a3.dot(x)
    beta[5] = a5.dot(x)
    beta[0] = (
        3
        / 5
        * (
            -1 / 7
            + 1 / 5 * beta[2] * (1 / 7 + 4 / 7 * II + 8 / 3 * III)
            - beta[3] * (1 / 5 - 8 / 15 * II - 14 / 15 * III)
            - beta[5]
            * (
                1 / 35
                - 24 / 105 * III
                - 4 / 35 * II
                + 16 / 15 * II * III
                + 8 / 35 * II ** 2
            )
        )
    )
    beta[1] = (
        6
        / 7
        * (
            1
            - 1 / 5 * beta[2] * (1 + 4 * II)
            + 7 / 5 * beta[3] * (1 / 6 - II)
            - beta[5] * (-1 / 5 + 2 / 3 * III + 4 / 5 * II - 8 / 5 * II ** 2)
        )
    )
    beta[4] = -4 / 5 * beta[2] - 7 / 5 * beta[3] - 6 / 5 * beta[5] * (1 - 4 / 3 * II)

    eye = np.eye(a.shape[0])
    A = (
        beta[0] * symmetrize(np.tensordot(eye, eye, 0))
        + beta[1] * symmetrize(np.tensordot(eye, a, 0))
        + beta[2] * symmetrize(np.tensordot(a, a, 0))
        + beta[3] * symmetrize(np.tensordot(eye, a.dot(a), 0))
        + beta[4] * symmetrize(np.tensordot(a, a.dot(a), 0))
        + beta[5] * symmetrize(np.tensordot(a.dot(a), a.dot(a), 0))
    )
    return A


def A4_orthotropic(a):
    r"""
    Compute the orthotropic closure in the principal frame

    Args:
        a (array_like of shape (3,)): Fiber orientation principal values, ``a[0] >= a[1] >= a[2]``

    Returns:
        array of shape (6, 6): 4th-order orientation tensor using the :math:`(\phi,\phi)` bases

    References:
        VerWeyst, B. E. Numerical predictions of flow-induced fiber orientation in three-dimensional geometries. University of Illinois at Urbana-Champaign, 1998
    """

    # Fitted coefficients (pp. 47)
    C = np.array(
        [
            [
                0.636256796880687,
                -1.872662963738140,
                -4.479708731937380,
                11.958956233232000,
                3.844596924200860,
                11.342092427815900,
                -10.958262606969100,
                -20.727799468413200,
                -2.116232144710040,
                -12.387563285561900,
                9.815983897167480,
                3.479015105674390,
                11.749291117702600,
                0.508041387366637,
                4.883665977714890,
            ],
            [
                0.636256796880687,
                -3.315272297421460,
                -3.037099398254060,
                11.827328596885200,
                6.881539520580440,
                8.436777467783250,
                -15.912066715764100,
                -15.151587260630700,
                -6.487289336419260,
                -8.638914192840160,
                9.325203434526610,
                7.746837517132950,
                7.481468706244410,
                2.284765316379580,
                3.597722511342540,
            ],
            [
                2.740532895602530,
                -9.121965097826920,
                -12.257058703625400,
                34.319901891698700,
                13.829469912194000,
                25.868475525388400,
                -37.702911802938400,
                -50.275643192748500,
                -10.880176113317400,
                -26.963691523971600,
                27.334679805448800,
                15.265068614865100,
                26.113491400537500,
                3.432138403347790,
                10.611741806606000,
            ],
        ]
    )

    x = np.array(
        [
            1,
            a[0],
            a[1],
            a[0] * a[1],
            a[0] ** 2,
            a[1] ** 2,
            a[0] ** 2 * a[1],
            a[0] * a[1] ** 2,
            a[0] ** 3,
            a[1] ** 3,
            a[0] ** 2 * a[1] ** 2,
            a[0] ** 3 * a[1],
            a[0] * a[1] ** 3,
            a[0] ** 4,
            a[1] ** 4,
        ]
    )
    A_123 = C.dot(x)  # A11, A22 and A33

    # Solve A44, A55 and A66 (or A66, A44 and A55 due to a change of basis)
    invM = np.array([[0.5, 0.5, -0.5], [-0.5, 0.5, 0.5], [0.5, -0.5, 0.5]])
    b = a - A_123
    A_456 = invM.dot(b)

    # Construct A in the (phi, phi) bases
    A = np.diag(np.hstack([A_123, A_456]))
    A[0, 1] = A[1, 0] = A[3, 3]
    A[1, 2] = A[2, 1] = A[4, 4]
    A[0, 2] = A[2, 0] = A[5, 5]
    A[:, 3:] *= 2
    return A


def A4_exact(a):
    r"""
    Compute the exact closure in the principal frame

    Args:
        a (array_like of shape (3,)): Fiber orientation principal values, ``a[0] >= a[1] >= a[2]``

    Returns:
        array of shape (6, 6): 4th-order orientation tensor written using the :math:`(\phi,\phi)` bases
    """
    from scipy import optimize
    import mpmath as mp

    def func(s):
        b1 = np.exp(s[0] + s[1])
        b2 = np.exp(s[0] - s[1])
        b3 = 1 / (b1 * b2)
        return (
            1
            / 3
            * np.array(
                [np.real(mp.fp.elliprd(b2, b3, b1)), np.real(mp.fp.elliprd(b1, b3, b2))]
            )
            - a[:2]
        )

    sol = optimize.root(func, np.ones(2))
    s = sol.x
    b1 = np.exp(s[0] + s[1])
    b2 = np.exp(s[0] - s[1])
    b = np.array([b1, b2, 1 / (b1 * b2)])

    # If b[0] ~= b[1] ~= b[2]
    if np.isclose(b[0], b[1]) and np.isclose(b[1], b[2]) and np.isclose(b[0], b[2]):
        c = b - 1
        A1111 = 1 / 5 - 3 / 14 * c[0] - 3 / 70 * c[1] - 3 / 70 * c[2]
        A2222 = 1 / 5 - 3 / 70 * c[0] - 3 / 14 * c[1] - 3 / 70 * c[2]
        A3333 = 1 / 5 - 3 / 70 * c[0] - 3 / 70 * c[1] - 3 / 14 * c[2]
        A1122 = 1 / 15 - 3 / 70 * c[0] - 3 / 70 * c[1] - 1 / 70 * c[2]
        A1133 = 1 / 15 - 3 / 70 * c[0] - 1 / 70 * c[1] - 3 / 70 * c[2]
        A2233 = 1 / 15 - 1 / 70 * c[0] - 3 / 70 * c[1] - 3 / 70 * c[2]
    else:
        if np.isclose(b[0], b[1]):
            A1133 = (b[0] * a[0] - b[2] * a[2]) / (2 * (b[0] - b[2]))
            A2233 = (b[1] * a[1] - b[2] * a[2]) / (2 * (b[1] - b[2]))
            A1122 = 1 / 8 * (a[0] + a[1] - A1133 - A2233)
        elif np.isclose(b[1], b[2]):
            A1122 = (b[0] * a[0] - b[1] * a[1]) / (2 * (b[0] - b[1]))
            A1133 = (b[0] * a[0] - b[2] * a[2]) / (2 * (b[0] - b[2]))
            A2233 = 1 / 8 * (a[1] + a[2] - A1133 - A1122)
        elif np.isclose(b[0], b[2]):
            A2233 = (b[1] * a[1] - b[2] * a[2]) / (2 * (b[1] - b[2]))
            A1122 = (b[0] * a[0] - b[1] * a[1]) / (2 * (b[0] - b[1]))
            A1133 = 1 / 8 * (a[0] + a[2] - A2233 - A1122)
        else:
            A1122 = (b[0] * a[0] - b[1] * a[1]) / (2 * (b[0] - b[1]))
            A1133 = (b[0] * a[0] - b[2] * a[2]) / (2 * (b[0] - b[2]))
            A2233 = (b[1] * a[1] - b[2] * a[2]) / (2 * (b[1] - b[2]))
        A1111 = a[0] - A1122 - A1133
        A2222 = a[1] - A1122 - A2233
        A3333 = a[2] - A1133 - A2233

    # Pack everything up
    A = np.zeros((6, 6))
    A1212 = A1122
    A2323 = A2233
    A1313 = A1133
    A[:3, :3] = np.array(
        [[A1111, A1122, A1133], [A1122, A2222, A2233], [A1133, A2233, A3333]]
    )
    A[3:, 3:] = 2 * np.diag([A1212, A2323, A1313])
    return A
