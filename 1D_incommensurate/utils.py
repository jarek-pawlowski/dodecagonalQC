import os
import numpy as np
from scipy.fft import fft, fft2, fftshift

import matplotlib.pyplot as plt


class RandomLattice1D:
    
    def __init__(self, supercell_size=1000):
        self.supercell_size = supercell_size

    def generate_sublattices_with_bonds(
        self,
        periods,
        offsets=None,
        start=0.0,
        stop=None,
        inter_cutoff=None,
    ):
        """Generate 1D sublattices and explicit intra/inter bond lists.

        Parameters
        ----------
        periods : sequence of float
            Uniform spacing of each sublattice.
        offsets : sequence of float, optional
            Offset of each sublattice. Defaults to zero for all sublattices.
        start, stop : float
            Spatial interval in which nodes are generated.
        inter_cutoff : float, optional
            Connect nodes from different sublattices when their distance is
            strictly smaller than this value. If None, no inter-sublattice
            bonds are generated.

        Returns
        -------
        positions, labels, intra_bonds, inter_bonds
            ``intra_bonds`` contains tuples ``(i, j, sublattice)`` and
            ``inter_bonds`` contains tuples ``(i, j, label_i, label_j)``.
        """
        periods = np.asarray(periods, dtype=float)
        if periods.ndim != 1 or len(periods) == 0:
            raise ValueError("periods must be a non-empty 1D sequence")
        if np.any(periods <= 0):
            raise ValueError("all sublattice periods must be positive")

        if offsets is None:
            offsets = np.zeros(len(periods), dtype=float)
        else:
            offsets = np.asarray(offsets, dtype=float)
        if offsets.shape != periods.shape:
            raise ValueError("periods and offsets must have the same length")

        if stop is None:
            stop = float(self.supercell_size)
        if stop <= start:
            raise ValueError("stop must be greater than start")
        if inter_cutoff is not None and inter_cutoff <= 0:
            raise ValueError("inter_cutoff must be positive")

        positions = []
        labels = []
        for label, (period, offset) in enumerate(zip(periods, offsets)):
            first = int(np.ceil((start - offset) / period))
            last = int(np.floor((stop - offset) / period))
            if last < first:
                continue
            indices = np.arange(first, last + 1, dtype=int)
            sub_positions = offset + period * indices
            positions.extend(sub_positions.tolist())
            labels.extend([label] * len(sub_positions))

        positions = np.asarray(positions, dtype=float)
        labels = np.asarray(labels, dtype=int)
        if len(positions) == 0:
            raise ValueError("no lattice nodes were generated")

        order = np.argsort(positions, kind='stable')
        self.positions = positions[order]
        self.labels = labels[order]
        self.sublattices = periods
        self.offsets = offsets
        self.lis = [np.flatnonzero(self.labels == s) for s in range(len(periods))]

        self.intra_bonds = []
        for s, indices in enumerate(self.lis):
            for i, j in zip(indices[:-1], indices[1:]):
                self.intra_bonds.append((int(i), int(j), int(s)))

        self.inter_bonds = []
        if inter_cutoff is not None:
            n = len(self.positions)
            for i in range(n):
                for j in range(i + 1, n):
                    distance = self.positions[j] - self.positions[i]
                    if distance >= inter_cutoff:
                        break
                    if self.labels[i] != self.labels[j]:
                        self.inter_bonds.append(
                            (int(i), int(j), int(self.labels[i]), int(self.labels[j]))
                        )

        return self.positions, self.labels, self.intra_bonds, self.inter_bonds
        
    def spring_c(self, x):
        return 1./(self.alpha_c+x)

    @staticmethod
    def _coupling_for_sublattice(coupling, sublattice):
        """Resolve scalar, sequence, or dict-valued intra coupling."""
        if np.isscalar(coupling):
            return float(coupling)
        if isinstance(coupling, dict):
            return float(coupling.get(sublattice, 1.0))
        values = np.asarray(coupling, dtype=float)
        return float(values[sublattice])

    @staticmethod
    def _coupling_for_pair(coupling, label_i, label_j):
        """Resolve scalar, matrix, or dict-valued inter coupling."""
        if np.isscalar(coupling):
            return float(coupling)
        if isinstance(coupling, dict):
            key = (label_i, label_j)
            reverse_key = (label_j, label_i)
            if key in coupling:
                return float(coupling[key])
            if reverse_key in coupling:
                return float(coupling[reverse_key])
            return 1.0
        matrix = np.asarray(coupling, dtype=float)
        return float(matrix[label_i, label_j])

    def dynamical_matrix(
        self,
        alpha_c,
        intra_coupling=1.0,
        inter_coupling=1.0,
    ):
        """Build a dynamical matrix from explicit intra/inter bond lists.

        ``intra_coupling`` may be a scalar, a sequence with one value per
        sublattice, or a dict keyed by sublattice index.

        ``inter_coupling`` may be a scalar, a square coupling matrix, or a
        dict keyed by ``(label_i, label_j)``.
        """
        if not hasattr(self, 'intra_bonds') or not hasattr(self, 'inter_bonds'):
            raise RuntimeError(
                "generate the lattice with generate_sublattices_with_bonds() first"
            )

        self.alpha_c = alpha_c
        n = len(self.positions)
        D_matrix = np.zeros((n, n), dtype=float)

        def add_spring(i, j, scale):
            distance = abs(self.positions[j] - self.positions[i])
            #if distance <= 1.e-12:
            #    raise ValueError(f"coincident bonded nodes detected: {i}, {j}")
            k = float(scale) * self.spring_c(distance)
            D_matrix[i, i] += k
            D_matrix[j, j] += k
            D_matrix[i, j] -= k
            D_matrix[j, i] -= k

        for i, j, sublattice in self.intra_bonds:
            scale = self._coupling_for_sublattice(intra_coupling, sublattice)
            add_spring(i, j, scale)

        for i, j, label_i, label_j in self.inter_bonds:
            scale = self._coupling_for_pair(inter_coupling, label_i, label_j)
            add_spring(i, j, scale)

        return D_matrix

    def plot_sublattices(
        self,
        filename=None,
        vertical_spacing=1.0,
        node_size=90,
        linewidth=1.0,
        show_indices=False,
        figsize=(12, 3),
    ):
        """Plot sublattices as colored balls with intra/inter connections.

        Sublattices are shifted vertically to make overlapping or nearby
        positions easier to distinguish.
        """
        if not hasattr(self, 'intra_bonds') or not hasattr(self, 'inter_bonds'):
            raise RuntimeError(
                "generate the lattice with generate_sublattices_with_bonds() first"
            )

        no_sublattices = len(self.lis)
        levels = (
            np.arange(no_sublattices, dtype=float)
            - 0.5 * (no_sublattices - 1)
        ) * vertical_spacing
        y = levels[self.labels]

        fig, ax = plt.subplots(figsize=figsize)
        cmap = plt.get_cmap('tab10')

        # Draw bonds first so nodes remain visible on top.
        for i, j, sublattice in self.intra_bonds:
            ax.plot(
                [self.positions[i], self.positions[j]],
                [y[i], y[j]],
                linewidth=linewidth,
                alpha=0.7,
                color=cmap(sublattice % 10),
                zorder=1,
            )

        for i, j, _, _ in self.inter_bonds:
            ax.plot(
                [self.positions[i], self.positions[j]],
                [y[i], y[j]],
                linewidth=linewidth,
                alpha=0.35,
                color='0.35',
                zorder=1,
            )

        for s, indices in enumerate(self.lis):
            ax.scatter(
                self.positions[indices],
                y[indices],
                s=node_size,
                color=cmap(s % 10),
                edgecolors='black',
                linewidths=0.5,
                label=f'Sublattice {s}',
                zorder=2,
            )

        if show_indices:
            for i, (x_i, y_i) in enumerate(zip(self.positions, y)):
                ax.annotate(
                    str(i),
                    (x_i, y_i),
                    xytext=(0, 7),
                    textcoords='offset points',
                    ha='center',
                    fontsize=7,
                )

        ax.set_xlabel('position')
        ax.set_yticks(levels)
        ax.set_yticklabels([f'S{s}' for s in range(no_sublattices)])
        ax.set_ylim(levels.min() - vertical_spacing, levels.max() + vertical_spacing)
        ax.legend(loc='upper right', ncol=min(no_sublattices, 4))
        ax.grid(axis='x', alpha=0.2)
        fig.tight_layout()

        if filename is not None:
            fig.savefig(filename, dpi=200, bbox_inches='tight')

        return fig, ax
    
    def set_phonon_phason_vectors(self):
        n = len(self.labels)
        A = self.labels == 0
        B = self.labels == 1

        # Jednorodna translacja całego układu
        phonon_vector = np.ones(n, dtype=float)
        phonon_vector /= np.linalg.norm(phonon_vector)

        # Względna translacja podsieci.
        # Współczynniki są dobrane tak, aby wektor był
        # ortogonalny do jednorodnej translacji.
        phason_vector = np.zeros(n, dtype=float)
        phason_vector[A] = 1.0
        phason_vector[B] = -A.sum() / B.sum()
        phason_vector /= np.linalg.norm(phason_vector)
        self.phonon_vector = phonon_vector
        self.phason_vector = phason_vector
        
    def mode_character_vs_k(self, modes, k_values):
        x = self.positions
        labels = self.labels

        A = labels == 0
        B = labels == 1

        eta = A.sum() / B.sum()

        phonon_intensity = np.zeros(
            (len(k_values), modes.shape[1])
        )
        phason_intensity = np.zeros_like(phonon_intensity)
        
        # Compute the fraction of each mode that resides on each sublattice.
        eps = 1e-12
        weight_a = np.sum(
            np.abs(modes[A, :]) ** 2,
            axis=0,
        )
        weight_b = np.sum(
            np.abs(modes[B, :]) ** 2,
            axis=0,
        )
        total_weight = weight_a + weight_b + eps
        fraction_a = weight_a / total_weight
        fraction_b = weight_b / total_weight
        # 0, when one sublattice dominates the mode, 1 when both sublattices contribute equally.
        participation_balance = (
            4.0 * fraction_a * fraction_b
        )

        for ik, k in enumerate(k_values):
            phase = np.exp(1j * k * x)

            phonon_probe = phase.copy()

            phason_probe = phase.copy()
            phason_probe[B] *= -eta

            phonon_probe /= np.linalg.norm(phonon_probe)
            phason_probe /= np.linalg.norm(phason_probe)

            phonon_intensity[ik] = np.abs(
                np.conjugate(phonon_probe) @ modes
            ) ** 2

            phason_intensity[ik] = np.abs(
                np.conjugate(phason_probe) @ modes
            ) ** 2

        return phonon_intensity, phason_intensity, participation_balance
    
    def sublattice_fourier_modes(self, modes, k_values):
        x = self.positions
        A = self.labels == 0
        B = self.labels == 1

        phase_A = np.exp(
            -1j * np.outer(k_values, x[A])
        )
        phase_B = np.exp(
            -1j * np.outer(k_values, x[B])
        )

        F_A = phase_A @ modes[A, :]
        F_B = phase_B @ modes[B, :]

        common = np.abs(F_A + F_B) ** 2
        relative = np.abs(F_A - F_B) ** 2

        return common, relative