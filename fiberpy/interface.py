import dmsh
import numpy as np
from matplotlib.tri import Triangulation

from .mechanics import A2Eij
from .orientation import project_aij


class TUBMesh:
    """
    Triangular mesh for the TUB orientation space
    """

    def __init__(self, h):
        geo = dmsh.Polygon([[1 / 3, 1 / 3], [1, 0], [0.5, 0.5]])
        self.points, self.triangles = dmsh.generate(geo, h)
        self.tri = Triangulation(self.points[:, 0], self.points[:, 1], self.triangles)
        self.trifinder = self.tri.get_trifinder()

    def centroids(self):
        centroids = np.empty((len(self.triangles), 2))
        for i, triangle in enumerate(self.triangles):
            centroids[i] = np.mean(self.points[triangle], axis=0)
        return centroids

    def locate(self, points):
        return self.trifinder(points[:, 0], points[:, 1])


class FEAInterface:
    """
    Generic FEA interface for integrative simulations

    Args:
        fiber_composite: Fiber composite object
        TUB_h (float): Mesh size for the TUB orientation space
    """
    def __init__(self, fiber_composite, TUB_h):
        self.fiber_composite = fiber_composite
        self.mesh = TUBMesh(TUB_h)

    def ABarTUB(self):
        """
        Calculate the effective elasticity and thermal dilatation tensors on
        the TUB orientation space
        """
        ABar = np.empty((len(self.mesh.triangles), 6, 6))
        alphaBar = np.empty((len(ABar), 3))
        for i, a in enumerate(self.mesh.centroids()):
            ABar[i, :, :] = self.fiber_composite.ABar(
                np.array([a[0], a[1], 1 - a[0] - a[1]])
            )
            alphaBar[i, :] = self.fiber_composite.alphaBar(ABar[i, :, :])

        return ABar, alphaBar

    def locate(self, a):
        """
        Locate the element in TUB containing the given fiber orientation

        Args:
            a (ndarray of shape (n, 3)): Principal values of fiber orientation tensor in decreasing order
        """
        return self.mesh.locate(a[:, :2])


class OptistructInterface(FEAInterface):
    """
    FEA interface for Altair Optistruct
    """
    def generate(self, a_dict, infile, outfile):
        """
        Generate an integrative simulation file for Altair Optistruct

        Args:
            a_dict (dict): Mapped fiber orientation tensor for given elements associated with PID / MID as keys.
                           Its components are ordered as ``a11, a22, a33, a12, a23, a13``
            infile (str): Input original isotropic simulation file
            outfile (str): Output modified integrative simulation file
        """

        # Transform orientations to principal bases
        print("Processing mapped orientations...")
        composite_ids = list(a_dict.keys())
        composite_nelems = sum([len(a) for a in a_dict.values()])
        eig = {}
        e3_e1 = {}
        for key, a in a_dict.items():
            nelems = len(a)
            eig[key] = np.empty((nelems, 3))
            e3_e1[key] = np.empty((nelems, 6))
            for i in range(nelems):
                a_ = a[i, :]
                a_tensor = np.array(
                    [
                        [a_[0], a_[3], a_[5]],
                        [a_[3], a_[1], a_[4]],
                        [a_[5], a_[4], a_[2]],
                    ]
                )
                e, v = np.linalg.eigh(a_tensor)
                eig[key][i, :] = project_aij(e[::-1])
                v = v[:, ::-1]
                e3_e1[key][i, :] = np.hstack([v[:, 2], v[:, 0]])

        # Find the corresponding MID
        print("Finding corresponding MID...")
        mid_dict = {}
        midset = set()
        for key in composite_ids:
            mid_dict[key] = self.locate(eig[key]) + 1
            midset = midset.union(set(mid_dict[key]))

        # Read PSOLID info
        print("Reading PSOLID info...")
        psolid_info = self._read_PSOLID(composite_ids, infile)

        # Compute needed ABar and alphaBar on TUB
        print("Computing MAT9ORT cards...")
        ABar, alphaBar = self.ABarTUB()

        # Here we begins
        print("Reading and generating FEA file...")
        infile_fh = open(infile, "r")
        outfile_fh = open(outfile, "w")

        pid_mid = 0
        composite_ind = {}
        for key in composite_ids:
            composite_ind[key] = 0

        materials_written = False
        for line in infile_fh:
            keyword = line[0:8].strip()
            # Elements
            if keyword == "CHEXA" or keyword == "CTETRA":
                pid = int(line[16:24])
                if pid in composite_ids:
                    ind = composite_ind[pid]
                    mid_ = mid_dict[pid]
                    e3_e1_ = e3_e1[pid]
                    psolid_info_ = psolid_info[pid]
                    pid_mid += 1
                    self._write_CORD2R(pid_mid, e3_e1_[ind], fh=outfile_fh)
                    self._write_PSOLID(
                        pid_mid, mid_[ind], pid_mid, psolid_info_, fh=outfile_fh
                    )
                    line = line[:16] + f"{pid_mid:8d}" + line[24:]
                    composite_ind[pid] += 1
                else:
                    line = line[:16] + f"{composite_nelems + pid:8d}" + line[24:]
                outfile_fh.write(line)

            # Properties
            elif keyword == "PSOLID":
                pid = int(line[8:16])
                mid = int(line[16:24])
                if pid in composite_ids:
                    outfile_fh.write("$" + line)
                else:
                    line = (
                        line[:8]
                        + f"{composite_nelems + pid:8d}"
                        + f"{composite_nelems + mid:8d}"
                        + line[24:]
                    )
                    outfile_fh.write(line)

            # Materials
            elif keyword == "MAT1":
                mid = int(line[8:16])
                if mid in composite_ids:
                    outfile_fh.write("$" + line)
                else:
                    line = line[:8] + f"{composite_nelems + mid:8d}" + line[16:]
                    outfile_fh.write(line)
                if not materials_written:
                    for i in range(len(ABar)):
                        mid = i + 1
                        if mid in midset:
                            self._write_MAT9ORT(
                                ABar[i], mid, alpha=alphaBar[i], fh=outfile_fh
                            )
                    materials_written = True
            else:
                outfile_fh.write(line)

        # Check all composite elements are treated
        assert composite_nelems == pid_mid

        infile_fh.close()
        outfile_fh.close()

        # Create a file without COO
        self._comment_CORD2R(outfile)

    def _read_PSOLID(self, composite_ids, infile):
        """
        Read the additional PSOLID information for given PID's
        """
        psolid_info = {}
        with open(infile) as fh:
            for line in fh:
                row = line.strip()
                if row[0:8].strip() == "PSOLID":
                    pid = int(row[8:16])
                    if pid in composite_ids and pid not in psolid_info:
                        psolid_info[pid] = row[32:]
                    if set(composite_ids) == set(psolid_info.keys()):
                        break
        return psolid_info

    def _write_MAT9ORT(self, A, mid, rho=None, alpha=None, fh=None):
        """
        Print the orthotropic elasticty components from an elasticity tensor
        in the optistruct MAT9ORT format
        """
        assert A.shape == (6, 6)
        Eij = A2Eij(A)  # E1, E2, E3, mu12, mu23, mu13, nu12, nu23, nu31

        def str_Eij(a=None, b=None):
            return [f"{x:g}" for x in Eij[a:b]]

        line = (
            "MAT9ORT, "
            + f"{mid:d}, "
            + ", ".join(str_Eij(b=3))
            + ", "
            + ", ".join(str_Eij(6, 9))
        )
        print(line, end="", file=fh)
        if rho is not None:
            print(f", {rho:g}", end="", file=fh)
        print("", file=fh)
        line = "," + ", ".join(str_Eij(3, 6))
        print(line, end="", file=fh)
        if alpha is not None:
            print(", " + ", ".join([f"{x:g}" for x in alpha]), end="", file=fh)
        print("", file=fh)

    def _write_PSOLID(self, pid, mid, cordm, info, fh=None):
        """
        Write an Optistruct PSOLID property
        """
        print("PSOLID  " + f"{pid:8d}" + f"{mid:8d}" + f"{cordm:8d}" + info, file=fh)

    def _write_CORD2R(self, cid, e3_e1, fh=None):
        """
        Write an Optistruct CORD2R coordinate system
        """

        def str_e(a=None, b=None):
            return [f"{x:g}" for x in e3_e1[a:b]]

        line = "CORD2R, " + f"{cid:d},, 0, 0, 0, " + ", ".join(str_e(b=3))
        print(line, file=fh)
        line = ", " + ", ".join(str_e(a=3, b=6))
        print(line, file=fh)

    def _comment_CORD2R(self, infile):
        """
        Comment CORD2R lines
        """
        outfile = open(infile.replace(".fem", "_without_CORD2R.fem"), "w")
        infile = open(infile, "r")
        comment = False
        for line in infile:
            if line.startswith("CORD2R"):
                line = "$" + line
                comment = True
            elif comment is True:
                line = "$" + line
                comment = False
            else:
                pass
            outfile.write(line)
        infile.close()
        outfile.close()
