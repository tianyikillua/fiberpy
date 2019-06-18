import numpy as np

from fiberpy.tensor import Mat2, Mat4, Mat22, MatGP, MatPG, ij2M, ijkl2MN


def test_Mat4():
    eye = np.eye(3)
    A = np.einsum("ij,kl", eye, eye)  # trace operator on 2nd order tensors
    M = Mat4(A)
    assert np.allclose(M[:3, :3], np.ones((3, 3)))
    M[:3, :3] -= np.ones((3, 3))
    assert np.isclose(np.linalg.norm(M), 0)


def test_Mat2():
    a = np.random.rand(6)
    assert np.allclose(Mat2(Mat2(a)), a)

    A = np.random.rand(3, 3)
    A = (A + A.T) / 2
    assert np.allclose(Mat2(Mat2(A)), A)


def test_Mat22():
    a = np.random.rand(6)
    assert np.allclose(Mat22(Mat22(a)), a)

    A = np.random.rand(3, 3)
    A = (A + A.T) / 2
    assert np.allclose(Mat22(Mat22(A)), A)


def test_ij2M():
    assert ij2M("11") == 0
    assert ij2M("13") == 5


def test_ijkl2MN():
    assert ijkl2MN("2212") == (1, 3)
    assert ijkl2MN("2313") == (4, 5)


def test_MatPG_MatGP():
    A = np.random.rand(3, 3)
    A = (A + A.T) / 2
    _, v = np.linalg.eigh(A)
    assert np.allclose(MatPG(v) @ MatGP(v), np.eye(6))
