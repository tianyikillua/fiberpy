import numpy as np


def Mat4(A):
    r"""
    Matrix representation of a 4th order tensor
    with minor symmety using the :math:`(\phi,\phi)` bases

    Args:
        A (ndarray of shape (3, 3, 3, 3)): 4th-order tensor
    """
    assert A.ndim == 4, "Only support 4th order tensor"
    M = np.array(
        [
            [
                A[0, 0, 0, 0],
                A[0, 0, 1, 1],
                A[0, 0, 2, 2],
                2 * A[0, 0, 0, 1],
                2 * A[0, 0, 1, 2],
                2 * A[0, 0, 0, 2],
            ],
            [
                A[1, 1, 0, 0],
                A[1, 1, 1, 1],
                A[1, 1, 2, 2],
                2 * A[1, 1, 0, 1],
                2 * A[1, 1, 1, 2],
                2 * A[1, 1, 0, 2],
            ],
            [
                A[2, 2, 0, 0],
                A[2, 2, 1, 1],
                A[2, 2, 2, 2],
                2 * A[2, 2, 0, 1],
                2 * A[2, 2, 1, 2],
                2 * A[2, 2, 0, 2],
            ],
            [
                A[0, 1, 0, 0],
                A[0, 1, 1, 1],
                A[0, 1, 2, 2],
                2 * A[0, 1, 0, 1],
                2 * A[0, 1, 1, 2],
                2 * A[0, 1, 0, 2],
            ],
            [
                A[1, 2, 0, 0],
                A[1, 2, 1, 1],
                A[1, 2, 2, 2],
                2 * A[1, 2, 0, 1],
                2 * A[1, 2, 1, 2],
                2 * A[1, 2, 0, 2],
            ],
            [
                A[0, 2, 0, 0],
                A[0, 2, 1, 1],
                A[0, 2, 2, 2],
                2 * A[0, 2, 0, 1],
                2 * A[0, 2, 1, 2],
                2 * A[0, 2, 0, 2],
            ],
        ]
    )
    return M


def Mat2(sig):
    r"""
    Bijection between a symmetric 2nd order tensor space
    and 6-dim vector space using the :math:`\phi` basis

    .. math::

        \begin{bmatrix}
        s_{11} & s_{12} & s_{13} \\
        s_{12} & s_{22} & s_{23} \\
        s_{13} & s_{23} & s_{33} \\
        \end{bmatrix}\iff\begin{bmatrix}
        s_{11} \\ s_{22} \\ s_{33} \\ s_{12} \\ s_{23} \\ s_{13}
        \end{bmatrix}

    Args:
        sig (ndarray of dim 1 or 2): 2nd-order tensor
    """
    if sig.ndim == 1:  # vector to matrix
        return np.array(
            [
                [sig[0], sig[3], sig[5]],
                [sig[3], sig[1], sig[4]],
                [sig[5], sig[4], sig[2]],
            ]
        )
    elif sig.ndim == 2:  # matrix to vector
        return np.array(
            [sig[0, 0], sig[1, 1], sig[2, 2], sig[0, 1], sig[1, 2], sig[0, 2]]
        )
    else:
        raise NotImplementedError("Only support vector or 2th order tensor")


def Mat22(eps):
    r"""
    Bijection between a symmetric 2nd order tensor space
    and 6-dim vector space using the :math:`\phi_2` basis

    .. math::

        \begin{bmatrix}
        e_{11} & e_{12} & e_{13} \\
        e_{12} & e_{22} & e_{23} \\
        e_{13} & e_{23} & e_{33} \\
        \end{bmatrix}\iff\begin{bmatrix}
        e_{11} \\ e_{22} \\ e_{33} \\ 2e_{12} \\ 2e_{23} \\ 2e_{13}
        \end{bmatrix}

    Args:
        eps (ndarray of dim 1 or 2): 2nd-order tensor
    """
    if eps.ndim == 1:  # vector to matrix
        return np.array(
            [
                [eps[0], eps[3] / 2, eps[5] / 2],
                [eps[3] / 2, eps[1], eps[4] / 2],
                [eps[5] / 2, eps[4] / 2, eps[2]],
            ]
        )
    elif eps.ndim == 2:  # matrix to vector
        return np.array(
            [
                eps[0, 0],
                eps[1, 1],
                eps[2, 2],
                2 * eps[0, 1],
                2 * eps[1, 2],
                2 * eps[0, 2],
            ]
        )
    else:
        raise Exception("Only support vector or 2th order tensor")


def Mat2S(eps):
    r"""
    Bijection between a symmetric 2nd order tensor space
    and 6-dim vector space using the :math:`\phi_\mathrm{S}` basis

    .. math::

        \begin{bmatrix}
        e_{11} & e_{12} & e_{13} \\
        e_{12} & e_{22} & e_{23} \\
        e_{13} & e_{23} & e_{33} \\
        \end{bmatrix}\iff\begin{bmatrix}
        e_{11} \\ e_{22} \\ e_{33} \\ \sqrt{2}e_{12} \\ \sqrt{2}e_{23} \\ \sqrt{2}e_{13}
        \end{bmatrix}

    Args:
        eps (ndarray of dim 1 or 2): 2nd-order tensor
    """
    sq2 = np.sqrt(2)
    if eps.ndim == 1:  # vector to matrix
        return np.array(
            [
                [eps[0], eps[3] / sq2, eps[5] / sq2],
                [eps[3] / sq2, eps[1], eps[4] / sq2],
                [eps[5] / sq2, eps[4] / sq2, eps[2]],
            ]
        )
    elif eps.ndim == 2:  # matrix to vector
        return np.array(
            [
                eps[0, 0],
                eps[1, 1],
                eps[2, 2],
                sq2 * eps[0, 1],
                sq2 * eps[1, 2],
                sq2 * eps[0, 2],
            ]
        )
    else:
        raise Exception("Only support vector or 2th order tensor")


def ij2M(ij):
    """
    Convert (i, j) indices of a symmetric
    2nd-order tensor to its vector index
    """
    if ij == "11":
        return 0
    elif ij == "22":
        return 1
    elif ij == "33":
        return 2
    elif ij == "12" or ij == "21":
        return 3
    elif ij == "23" or ij == "32":
        return 4
    elif ij == "13" or ij == "31":
        return 5


def ijkl2MN(ijkl):
    """
    Convert (i, j, k, l) indices of a symmetric
    4nd-order tensor to its matrix index
    """
    ij = ijkl[:2]
    kl = ijkl[2:]
    M = ij2M(ij)
    N = ij2M(kl)
    return M, N


def MatPG(v):
    r"""
    Matrix that converts a 2nd-order tensor
    from the principal frame (:math:`\phi` basis) to the global frame (:math:`\phi` basis)

    Args:
        v (ndarray of shape (3, 3)): Principal directions along its columns
    """

    return np.array(
        [
            [
                v[0, 0] ** 2,
                v[0, 1] ** 2,
                v[0, 2] ** 2,
                2 * v[0, 0] * v[0, 1],
                2 * v[0, 1] * v[0, 2],
                2 * v[0, 0] * v[0, 2],
            ],
            [
                v[1, 0] ** 2,
                v[1, 1] ** 2,
                v[1, 2] ** 2,
                2 * v[1, 0] * v[1, 1],
                2 * v[1, 1] * v[1, 2],
                2 * v[1, 0] * v[1, 2],
            ],
            [
                v[2, 0] ** 2,
                v[2, 1] ** 2,
                v[2, 2] ** 2,
                2 * v[2, 0] * v[2, 1],
                2 * v[2, 1] * v[2, 2],
                2 * v[2, 0] * v[2, 2],
            ],
            [
                v[0, 0] * v[1, 0],
                v[0, 1] * v[1, 1],
                v[0, 2] * v[1, 2],
                v[0, 0] * v[1, 1] + v[0, 1] * v[1, 0],
                v[0, 1] * v[1, 2] + v[0, 2] * v[1, 1],
                v[0, 0] * v[1, 2] + v[0, 2] * v[1, 0],
            ],
            [
                v[1, 0] * v[2, 0],
                v[1, 1] * v[2, 1],
                v[1, 2] * v[2, 2],
                v[1, 0] * v[2, 1] + v[1, 1] * v[2, 0],
                v[1, 1] * v[2, 2] + v[1, 2] * v[2, 1],
                v[1, 0] * v[2, 2] + v[1, 2] * v[2, 0],
            ],
            [
                v[0, 0] * v[2, 0],
                v[0, 1] * v[2, 1],
                v[0, 2] * v[2, 2],
                v[0, 0] * v[2, 1] + v[0, 1] * v[2, 0],
                v[0, 1] * v[2, 2] + v[0, 2] * v[2, 1],
                v[0, 0] * v[2, 2] + v[0, 2] * v[2, 0],
            ],
        ]
    )


def MatGP(v):
    r"""
    Matrix that converts a 2nd-order tensor
    from the global frame (:math:`\phi` basis) to the principal frame (:math:`\phi` basis)

    Args:
        v (ndarray of shape (3, 3)): Principal directions along its columns
    """

    return np.array(
        [
            [
                v[0, 0] ** 2,
                v[1, 0] ** 2,
                v[2, 0] ** 2,
                2 * v[0, 0] * v[1, 0],
                2 * v[1, 0] * v[2, 0],
                2 * v[0, 0] * v[2, 0],
            ],
            [
                v[0, 1] ** 2,
                v[1, 1] ** 2,
                v[2, 1] ** 2,
                2 * v[0, 1] * v[1, 1],
                2 * v[1, 1] * v[2, 1],
                2 * v[0, 1] * v[2, 1],
            ],
            [
                v[0, 2] ** 2,
                v[1, 2] ** 2,
                v[2, 2] ** 2,
                2 * v[0, 2] * v[1, 2],
                2 * v[1, 2] * v[2, 2],
                2 * v[0, 2] * v[2, 2],
            ],
            [
                v[0, 0] * v[0, 1],
                v[1, 0] * v[1, 1],
                v[2, 0] * v[2, 1],
                v[0, 0] * v[1, 1] + v[0, 1] * v[1, 0],
                v[1, 0] * v[2, 1] + v[1, 1] * v[2, 0],
                v[0, 0] * v[2, 1] + v[0, 1] * v[2, 0],
            ],
            [
                v[0, 1] * v[0, 2],
                v[1, 1] * v[1, 2],
                v[2, 1] * v[2, 2],
                v[0, 1] * v[1, 2] + v[0, 2] * v[1, 1],
                v[1, 1] * v[2, 2] + v[1, 2] * v[2, 1],
                v[0, 1] * v[2, 2] + v[0, 2] * v[2, 1],
            ],
            [
                v[0, 0] * v[0, 2],
                v[1, 0] * v[1, 2],
                v[2, 0] * v[2, 2],
                v[0, 0] * v[1, 2] + v[0, 2] * v[1, 0],
                v[1, 0] * v[2, 2] + v[1, 2] * v[2, 0],
                v[0, 0] * v[2, 2] + v[0, 2] * v[2, 0],
            ],
        ]
    )


def MatGP2(v):
    r"""
    Matrix that converts a 2nd-order tensor
    from the global frame (:math:`\phi_2` basis) to the principal frame (:math:`\phi` basis)

    Args:
        v (ndarray of shape (3, 3)): Principal directions along its columns
    """

    return np.array(
        [
            [
                v[0, 0] ** 2,
                v[1, 0] ** 2,
                v[2, 0] ** 2,
                v[0, 0] * v[1, 0],
                v[1, 0] * v[2, 0],
                v[0, 0] * v[2, 0],
            ],
            [
                v[0, 1] ** 2,
                v[1, 1] ** 2,
                v[2, 1] ** 2,
                v[0, 1] * v[1, 1],
                v[1, 1] * v[2, 1],
                v[0, 1] * v[2, 1],
            ],
            [
                v[0, 2] ** 2,
                v[1, 2] ** 2,
                v[2, 2] ** 2,
                v[0, 2] * v[1, 2],
                v[1, 2] * v[2, 2],
                v[0, 2] * v[2, 2],
            ],
            [
                v[0, 0] * v[0, 1],
                v[1, 0] * v[1, 1],
                v[2, 0] * v[2, 1],
                (v[0, 0] * v[1, 1] + v[0, 1] * v[1, 0]) / 2,
                (v[1, 0] * v[2, 1] + v[1, 1] * v[2, 0]) / 2,
                (v[0, 0] * v[2, 1] + v[0, 1] * v[2, 0]) / 2,
            ],
            [
                v[0, 1] * v[0, 2],
                v[1, 1] * v[1, 2],
                v[2, 1] * v[2, 2],
                (v[0, 1] * v[1, 2] + v[0, 2] * v[1, 1]) / 2,
                (v[1, 1] * v[2, 2] + v[1, 2] * v[2, 1]) / 2,
                (v[0, 1] * v[2, 2] + v[0, 2] * v[2, 1]) / 2,
            ],
            [
                v[0, 0] * v[0, 2],
                v[1, 0] * v[1, 2],
                v[2, 0] * v[2, 2],
                (v[0, 0] * v[1, 2] + v[0, 2] * v[1, 0]) / 2,
                (v[1, 0] * v[2, 2] + v[1, 2] * v[2, 0]) / 2,
                (v[0, 0] * v[2, 2] + v[0, 2] * v[2, 0]) / 2,
            ],
        ]
    )
