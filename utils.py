import os
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

import matplotlib.pyplot as plt
from matplotlib import cm


def Rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])

class CrystalUtils:
    def __init__(self, a=0.246, d=0.335):
        self.a = a
        self.d = d

    def make_dodecagonal_graphene(self, n=20):
        """
        Returns:
            layers = [
                [A_points_layer1, B_points_layer1],
                [A_points_layer2, B_points_layer2],
            ]

        All points are 3D arrays in nm.
        """

        ez = np.array([0.0, 0.0, 1.0])

        a1 = self.a * np.array([1.0, 0.0, 0.0])
        a2 = self.a * np.array([0.5, np.sqrt(3) / 2, 0.0])

        theta = np.pi / 6
        R = Rz(theta)

        a1p = R @ a1
        a2p = R @ a2

        t1 = self.a * np.array([0.0, 1 / np.sqrt(3), 0.0])

        tA = -t1
        tB =  t1

        tAp = -(R @ t1) + self.d * ez
        tBp =  (R @ t1) + self.d * ez

        A1, B1, A2, B2 = [], [], [], []

        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                Rcell1 = i * a1 + j * a2
                Rcell2 = i * a1p + j * a2p

                A1.append(Rcell1 + tA)
                B1.append(Rcell1 + tB)

                A2.append(Rcell2 + tAp)
                B2.append(Rcell2 + tBp)

        return [
            np.concatenate([np.array(A1), np.array(B1)]),
            np.concatenate([np.array(A2), np.array(B2)])
        ]

    def cut_sublattices_to_circle(self, sublattices, radius, center=(0.0, 0.0)):
        """
        Keeps only points whose x-y coordinates lie inside a centered circle.

        Parameters
        ----------
        sublattices : list of np.ndarray
            Example: [A1, B1, A2, B2], each array shape (N, 3).

        radius : float
            Circle radius in nm.

        center : tuple
            Circle center in x-y plane, in nm.

        Returns
        -------
        cut_sublattices : list of np.ndarray
            Sublattice arrays after cutting.
        """

        center = np.asarray(center, dtype=float)

        cut_sublattices = []

        for S in sublattices:
            xy = S[:, :2]
            r = np.linalg.norm(xy - center, axis=1)
            mask = r <= radius
            cut_sublattices.append(S[mask])

        return cut_sublattices

    def find_connections(self, sublattices, cutoff):
        """
        Parameters
        ----------
        sublattices : list of np.ndarray
            Example:
                sublattices = [A1, B1, A2, B2]
            where each item has shape (N, 3).

        cutoff : float
            Maximum neighbor distance in nm.

        Returns
        -------
        connections : np.ndarray
            Each row:
            [si, i, sj, j, dx, dy, dz, distance]
        """

        connections = []

        trees = [cKDTree(S) for S in sublattices]

        for si, Si in enumerate(sublattices):
            for sj, Sj in enumerate(sublattices):
                if sj < si:
                    continue

                pairs = trees[si].query_ball_tree(trees[sj], r=cutoff)

                for i, neighbors in enumerate(pairs):
                    for j in neighbors:
                        if si == sj and j <= i:
                            continue

                        vec = Sj[j] - Si[i]   # full displacement vector
                        dist = np.linalg.norm(vec)
                        pos = (Sj[j] + Si[i])/2

                        connections.append([
                            si, i,
                            sj, j,
                            vec[0], vec[1], vec[2],
                            dist
                        ])
                        
        return np.array(connections, dtype=float)
    
    def save(self, sublattices, indices, path='./', sublattices_filename='sublattices.npy', indices_filename='indices.npy'):
        np.save(os.path.join(path, sublattices_filename), sublattices)
        np.save(os.path.join(path, indices_filename), indices)

class HamUtils:
    
    def __init__(self, crystal, Vpp_pi0=-2.7, Vpp_sigma0=0.48, r0=0.0453):    
        self.Vpp_pi0 = Vpp_pi0
        self.Vpp_sigma0 = Vpp_sigma0
        self.r0 = r0
        self.a = crystal.a
        self.d = crystal.d

    def hopping_from_R(self, Rvec):
        """
        Slater-Koster hopping: Eq. (3) in Phys. Rev. B 99, 165430 (2019).

        Parameters
        ----------
        Rvec : array-like, shape (3,)
            Displacement vector R = r_j - r_i in nm.

        Returns
        -------
        h : float
            Hopping matrix element in eV.
        """
        Rvec = np.asarray(Rvec, dtype=float)
        R = np.linalg.norm(Rvec)
        z_ratio = Rvec[2]/R
        Vpp_pi = self.Vpp_pi0 * np.exp(-(R - self.a / np.sqrt(3)) / self.r0)
        Vpp_sigma = self.Vpp_sigma0 * np.exp(-(R - self.d) / self.r0)
        T = Vpp_pi * (1.0 - z_ratio**2) + Vpp_sigma * z_ratio**2
        return -T
    
    def build_sparse_hamiltonian(self, sublattices, connections, Rb=None):
        """
        Builds sparse tight-binding Hamiltonian from connection array.

        connections rows must be:
            [si, i, sj, j, dx, dy, dz, distance]

        Parameters
        ----------
        sublattices : list[np.ndarray]
            Sublattice point arrays, e.g. [A1, B1, A2, B2].

        connections : np.ndarray
            Output of find_connections_with_vectors.

        onsite : float
            Onsite energy in eV.

        Returns
        -------
        H : scipy.sparse.csr_matrix
            Sparse Hamiltonian matrix.

        global_indices : list[np.ndarray]
            global_indices[s][i] gives global node index of node i
            in sublattice s.
        """

        sizes = [len(S) for S in sublattices]
        offsets = np.cumsum([0] + sizes[:-1])
        total_nodes = sum(sizes)

        global_indices = [
            np.arange(offsets[s], offsets[s] + sizes[s])
            for s in range(len(sublattices))
        ]

        rows = []
        cols = []
        data = []

        # potential barrier to decouple edge states, if Rb is given
        if Rb is not None:
            for j, s in enumerate(sublattices):
                for i, xs in enumerate(s):
                    gi = global_indices[j][i]
                    Rp = np.sqrt(xs[0]**2+xs[1]**2) 
                    if Rp > Rb:
                        u = np.exp((Rp-Rb)/Rb*10.)-1.
                        rows.append(gi)
                        cols.append(gi)
                        data.append(u)

        # hopping terms
        for conn in connections:
            si = int(conn[0])
            i = int(conn[1])
            sj = int(conn[2])
            j = int(conn[3])

            Rvec = conn[4:7]

            gi = global_indices[si][i]
            gj = global_indices[sj][j]

            t = self.hopping_from_R(Rvec)

            rows.append(gi)
            cols.append(gj)
            data.append(t)

            rows.append(gj)
            cols.append(gi)
            data.append(t)

        H = coo_matrix(
            (data, (rows, cols)),
            shape=(total_nodes, total_nodes),
            dtype=float
        ).tocsr()
        
        return H, global_indices
    
    def diagonalize_hamiltonian(self, H, k=None, sigma=None, return_eigenvectors=False):
        """
        Diagonalize Hamiltonian.
        If k is None: full diagonalization.
        If k is integer: sparse diagonalization of k eigenvalues.

        Parameters
        ----------
        H : scipy sparse matrix
        k : int or None
            Number of eigenvalues. Use None for full spectrum.
        sigma : float or None
            Energy around which to find eigenvalues.

        Returns
        -------
        evals : np.ndarray
            Eigenvalues in eV.
        """
        if return_eigenvectors:
            return eigsh(H, k=k, sigma=sigma, return_eigenvectors=True)
        else:    
            evals = eigsh(H, k=k, sigma=sigma, return_eigenvectors=False)
            evals = np.sort(evals)
            return evals
        
    def save(self, evals, evecs, path='./', evals_filename='evals.npy', evecs_filename='evecs.npy'):
        np.save(os.path.join(path, evals_filename), evals)
        np.save(os.path.join(path, evecs_filename), evecs)


def plot_top_view(layers, s=8, path='./', filename='crystal_top_view.png'):
    """
    Plots all A sublattice points in red and all B sublattice points in blue.
    Top view: x-y plane.
    """
    colors = ['red', 'blue']
    fig, ax = plt.subplots(figsize=(7, 7))

    for layer, color in zip(layers, colors):
        ax.scatter(layer[:, 0], layer[:, 1], s=s, c=color, label=f'Layer {color}')

    ax.set_aspect("equal")
    ax.set_xlabel("x [nm]")
    ax.set_ylabel("y [nm]")
    ax.set_title("Dodecagonal graphene quasicrystal, top view")

    fig.savefig(os.path.join(path, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_connections(connections, sublattices, l=0.1, path='./', filename='connections.png'):

    fig, ax = plt.subplots(figsize=(7, 7))

    for S in sublattices:
        ax.scatter(S[:, 0], S[:, 1], s=1, c='black')

    # Draw lines between all pairs of points within a certain distance
    for ci, conn in enumerate(connections):
        p1 = sublattices[int(conn[0])][int(conn[1])]
        p2 = sublattices[int(conn[2])][int(conn[3])]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c=cm.hot(conn[-1]), lw=0.5)

    ax.set_aspect("equal")
    ax.set_xlabel("x [nm]")
    ax.set_ylabel("y [nm]")
    ax.set_title("Dodecagonal graphene quasicrystal, connections")

    fig.savefig(os.path.join(path, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_density(evec, sublattices, global_indices, ps=8, path='./', filename='eigenstate_density.png'):
    """
    Plots all A sublattice points in red and all B sublattice points in blue.
    Top view: x-y plane.
    """
    density = np.abs(evec)**2
    fig, ax = plt.subplots(figsize=(7, 7))

    xs0 = []
    xs1 = []
    gs = []
    for sj, s in enumerate(sublattices):
        for si, xs in enumerate(s):
            gs.append(global_indices[sj][si])
            xs0.append(xs[0])
            xs1.append(xs[1])

    ax.scatter(xs0, xs1, s=ps, c=density[gs], cmap='viridis', vmin=0, vmax=density.max())

    ax.set_aspect("equal")
    ax.set_xlabel("x [nm]")
    ax.set_ylabel("y [nm]")
    ax.set_title("Dodecagonal graphene quasicrystal, eigenstate density")

    fig.savefig(os.path.join(path, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_histogram_dos(evals, bins=100, path='./', filename='dos.png'):
    """
    Plot DOS using histogram.
    """
    
    dos, edges = np.histogram(
        evals,
        bins=bins,
        range=(evals.min(), evals.max()),
        density=True
    )
    E = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.step(E, dos, where='mid')
    ax.set_xlabel("Energy [eV]")
    ax.set_ylabel("DOS [arb. units]")
    ax.set_title("Density of States (Histogram)")

    fig.savefig(os.path.join(filename), dpi=300, bbox_inches='tight')
    plt.close()
