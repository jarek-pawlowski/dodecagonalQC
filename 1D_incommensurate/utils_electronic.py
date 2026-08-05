"""Utilities for a one-dimensional two-sublattice tight-binding model."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


class IncommensurateLattice1D:
    """Finite one-dimensional composite lattice with two or more sublattices."""

    def __init__(self, system_size: float = 1000.0):
        if system_size <= 0:
            raise ValueError("system_size must be positive")
        self.system_size = float(system_size)

    def generate_sublattices(
        self,
        periods: Sequence[float],
        offsets: Sequence[float] | None = None,
        start: float = 0.0,
        stop: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate uniformly spaced sublattices and merge them by position.

        The positions of sublattice ``s`` are
        ``x_(s,n) = offsets[s] + n * periods[s]``.
        """
        periods_array = np.asarray(periods, dtype=float)
        if periods_array.ndim != 1 or periods_array.size == 0:
            raise ValueError("periods must be a non-empty one-dimensional sequence")
        if np.any(periods_array <= 0):
            raise ValueError("all periods must be positive")

        if offsets is None:
            offsets_array = np.zeros_like(periods_array)
        else:
            offsets_array = np.asarray(offsets, dtype=float)
        if offsets_array.shape != periods_array.shape:
            raise ValueError("periods and offsets must have the same shape")

        end = self.system_size if stop is None else float(stop)
        if end <= start:
            raise ValueError("stop must be greater than start")

        positions: list[float] = []
        labels: list[int] = []
        for label, (period, offset) in enumerate(zip(periods_array, offsets_array)):
            first = int(np.ceil((start - offset) / period))
            last = int(np.floor((end - offset) / period))
            if last < first:
                continue
            indices = np.arange(first, last + 1, dtype=int)
            sub_positions = offset + period * indices
            positions.extend(sub_positions.tolist())
            labels.extend([label] * sub_positions.size)

        if not positions:
            raise ValueError("no lattice sites were generated")

        positions_array = np.asarray(positions, dtype=float)
        labels_array = np.asarray(labels, dtype=int)
        order = np.argsort(positions_array, kind="stable")

        self.positions = positions_array[order]
        self.labels = labels_array[order]
        self.periods = periods_array
        self.offsets = offsets_array
        self.sublattice_indices = [
            np.flatnonzero(self.labels == label)
            for label in range(periods_array.size)
        ]
        return self.positions, self.labels

    @staticmethod
    def distance_hopping(distance: np.ndarray | float, amplitude: float, xi: float):
        """Return the exponentially decaying hopping t(d)=amplitude*exp(-d/xi)."""
        if xi <= 0:
            raise ValueError("xi must be positive")
        return float(amplitude) * np.exp(-np.asarray(distance, dtype=float) / xi)

    @staticmethod
    def _value_for_sublattice(value, label: int) -> float:
        """Resolve a scalar, sequence, or dictionary value for one sublattice."""
        if np.isscalar(value):
            return float(value)
        if isinstance(value, Mapping):
            return float(value.get(label, 0.0))
        values = np.asarray(value, dtype=float)
        return float(values[label])

    @staticmethod
    def _value_for_pair(value, label_i: int, label_j: int) -> float:
        """Resolve a scalar, matrix, or dictionary value for a sublattice pair."""
        if np.isscalar(value):
            return float(value)
        if isinstance(value, Mapping):
            if (label_i, label_j) in value:
                return float(value[(label_i, label_j)])
            if (label_j, label_i) in value:
                return float(value[(label_j, label_i)])
            return 0.0
        matrix = np.asarray(value, dtype=float)
        return float(matrix[label_i, label_j])

    def tight_binding_hamiltonian(
        self,
        xi: float,
        intra_hopping=1.0,
        inter_hopping=1.0,
        onsite_energy=0.0,
        hopping_cutoff: float | None = None,
        include_long_range_intra: bool = True,
    ) -> np.ndarray:
        """Construct a real symmetric distance-dependent tight-binding Hamiltonian.

        Off-diagonal matrix elements are ``H_ij = -t_ij`` with
        ``t_ij = g_ij * exp(-|x_i-x_j|/xi)``. No Laplacian diagonal correction
        is added. The diagonal contains only the requested onsite energies.

        Parameters
        ----------
        xi:
            Hopping decay length.
        intra_hopping:
            Scalar, sequence, or dictionary defining hopping amplitudes within
            each sublattice.
        inter_hopping:
            Scalar, pair dictionary, or matrix defining hopping amplitudes
            between sublattices.
        onsite_energy:
            Scalar, sequence, or dictionary defining onsite energies.
        hopping_cutoff:
            Optional numerical cutoff in real-space distance. It should be much
            larger than ``xi`` so that omitted hoppings are negligible.
        include_long_range_intra:
            If false, only consecutive sites of each sublattice are connected;
            inter-sublattice hopping remains distance dependent for all pairs
            within the optional cutoff.
        """
        if not hasattr(self, "positions"):
            raise RuntimeError("generate the lattice before building the Hamiltonian")
        if xi <= 0:
            raise ValueError("xi must be positive")
        if hopping_cutoff is not None and hopping_cutoff <= 0:
            raise ValueError("hopping_cutoff must be positive")

        n_sites = self.positions.size
        hamiltonian = np.zeros((n_sites, n_sites), dtype=float)

        for i, label in enumerate(self.labels):
            hamiltonian[i, i] = self._value_for_sublattice(onsite_energy, int(label))

        nearest_intra_pairs: set[tuple[int, int]] = set()
        if not include_long_range_intra:
            for indices in self.sublattice_indices:
                nearest_intra_pairs.update(
                    (int(i), int(j)) for i, j in zip(indices[:-1], indices[1:])
                )

        for i in range(n_sites):
            for j in range(i + 1, n_sites):
                distance = self.positions[j] - self.positions[i]
                if hopping_cutoff is not None and distance > hopping_cutoff:
                    break

                label_i = int(self.labels[i])
                label_j = int(self.labels[j])
                if label_i == label_j:
                    if not include_long_range_intra and (i, j) not in nearest_intra_pairs:
                        continue
                    amplitude = self._value_for_sublattice(intra_hopping, label_i)
                else:
                    amplitude = self._value_for_pair(inter_hopping, label_i, label_j)

                hopping = self.distance_hopping(distance, amplitude, xi)
                hamiltonian[i, j] = -hopping
                hamiltonian[j, i] = -hopping

        return hamiltonian

    def sublattice_weights(self, states: np.ndarray) -> np.ndarray:
        """Return the probability weight of every eigenstate on each sublattice."""
        return np.asarray([
            np.sum(np.abs(states[indices, :]) ** 2, axis=0)
            for indices in self.sublattice_indices
        ])

    @staticmethod
    def inverse_participation_ratio(states: np.ndarray) -> np.ndarray:
        """Return IPR_n=sum_i |psi_(i,n)|^4 for normalized eigenstates."""
        return np.sum(np.abs(states) ** 4, axis=0)

    def fourier_components(
        self,
        states: np.ndarray,
        k_values: np.ndarray,
        normalize_sublattices: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate sublattice, common, and relative Fourier spectral weights.

        For two sublattices, the returned arrays are based on
        ``F_A(k,n)`` and ``F_B(k,n)``. If normalization is enabled, each
        sublattice amplitude is divided by the square root of its site count.
        """
        if len(self.sublattice_indices) != 2:
            raise NotImplementedError("common/relative analysis currently requires two sublattices")

        k_array = np.asarray(k_values, dtype=float)
        components = []
        for indices in self.sublattice_indices:
            phase = np.exp(-1j * np.outer(k_array, self.positions[indices]))
            component = phase @ states[indices, :]
            if normalize_sublattices:
                component /= np.sqrt(indices.size)
            components.append(component)

        f_a, f_b = components
        common = 0.5 * np.abs(f_a + f_b) ** 2
        relative = 0.5 * np.abs(f_a - f_b) ** 2
        return f_a, f_b, common, relative

    def relative_composition_vs_k(
        self,
        states: np.ndarray,
        k_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return normalized common/relative projections and participation balance.

        This is the electronic analogue of the previous phonon/phason probe.
        It measures bonding-like versus antibonding-like sublattice composition;
        it is not by itself a collective phason excitation.
        """
        if len(self.sublattice_indices) != 2:
            raise NotImplementedError("relative composition currently requires two sublattices")

        positions = self.positions
        mask_a = self.labels == 0
        mask_b = self.labels == 1
        eta = mask_a.sum() / mask_b.sum()

        common_intensity = np.zeros((len(k_values), states.shape[1]))
        relative_intensity = np.zeros_like(common_intensity)

        weights = self.sublattice_weights(states)
        total = weights[0] + weights[1] + 1e-15
        fraction_a = weights[0] / total
        fraction_b = weights[1] / total
        participation_balance = 4.0 * fraction_a * fraction_b

        for ik, k in enumerate(k_values):
            phase = np.exp(1j * k * positions)
            common_probe = phase.astype(complex, copy=True)
            relative_probe = phase.astype(complex, copy=True)
            relative_probe[mask_b] *= -eta

            common_probe /= np.linalg.norm(common_probe)
            relative_probe /= np.linalg.norm(relative_probe)

            common_intensity[ik] = np.abs(common_probe.conj() @ states) ** 2
            relative_intensity[ik] = np.abs(relative_probe.conj() @ states) ** 2

        return common_intensity, relative_intensity, participation_balance

    def plot_sublattices(
        self,
        filename: str | Path | None = None,
        vertical_spacing: float = 1.0,
        node_size: float = 80.0,
        hopping_xi: float | None = None,
        intra_hopping=1.0,
        inter_hopping=1.0,
        hopping_cutoff: float | None = None,
        minimum_visible_hopping: float = 1e-3,
        figsize: tuple[float, float] = (12.0, 3.0),
    ):
        """Plot colored sublattice sites and optional distance-dependent hoppings."""
        if not hasattr(self, "positions"):
            raise RuntimeError("generate the lattice before plotting it")

        n_sublattices = len(self.sublattice_indices)
        levels = (
            np.arange(n_sublattices, dtype=float) - 0.5 * (n_sublattices - 1)
        ) * vertical_spacing
        y_values = levels[self.labels]
        cmap = plt.get_cmap("tab10")

        fig, ax = plt.subplots(figsize=figsize)

        if hopping_xi is not None:
            max_hopping = 0.0
            visible_edges = []
            for i in range(self.positions.size):
                for j in range(i + 1, self.positions.size):
                    distance = self.positions[j] - self.positions[i]
                    if hopping_cutoff is not None and distance > hopping_cutoff:
                        break
                    label_i = int(self.labels[i])
                    label_j = int(self.labels[j])
                    if label_i == label_j:
                        amplitude = self._value_for_sublattice(intra_hopping, label_i)
                    else:
                        amplitude = self._value_for_pair(inter_hopping, label_i, label_j)
                    hopping = abs(float(self.distance_hopping(distance, amplitude, hopping_xi)))
                    if hopping >= minimum_visible_hopping:
                        visible_edges.append((i, j, hopping, label_i == label_j))
                        max_hopping = max(max_hopping, hopping)

            for i, j, hopping, is_intra in visible_edges:
                relative_width = hopping / max_hopping if max_hopping > 0 else 0.0
                ax.plot(
                    [self.positions[i], self.positions[j]],
                    [y_values[i], y_values[j]],
                    linewidth=0.3 + 2.0 * relative_width,
                    alpha=0.25 + 0.55 * relative_width,
                    color=cmap(int(self.labels[i]) % 10) if is_intra else "0.35",
                    zorder=1,
                )

        for label, indices in enumerate(self.sublattice_indices):
            ax.scatter(
                self.positions[indices],
                y_values[indices],
                s=node_size,
                color=cmap(label % 10),
                edgecolors="black",
                linewidths=0.5,
                label=f"Sublattice {label}",
                zorder=2,
            )

        ax.set_xlabel("position")
        ax.set_yticks(levels)
        ax.set_yticklabels([f"S{label}" for label in range(n_sublattices)])
        ax.set_ylim(levels.min() - vertical_spacing, levels.max() + vertical_spacing)
        ax.grid(axis="x", alpha=0.2)
        ax.legend(loc="upper right")
        fig.tight_layout()

        if filename is not None:
            output = Path(filename)
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=200, bbox_inches="tight")
        return fig, ax
