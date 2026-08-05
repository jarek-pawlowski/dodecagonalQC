"""Scan the electronic spectrum and localization as a function of hopping range xi."""

from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

from utils_electronic import IncommensurateLattice1D


RESULTS_PATH = Path("./electronic_results")
RESULTS_PATH.mkdir(parents=True, exist_ok=True)


def scan_xi_spectrum(
    q: float = np.sqrt(2.0),
    system_size: float = 4000.0,
    r_a: float = 20.0,
    offset_b: float = 1.0,
    intra_hopping=(1.0, 1.0),
    inter_hopping: float = 0.2,
    onsite_energy=(0.0, 0.0),
    xi_min: float = 0.5,
    xi_max: float = 20.0,
    number_of_xi: int = 100,
    cutoff_factor: float = 8.0,
):
    """Diagonalize H(xi) and return energies and IPR for every xi.

    A single real-space cutoff based on the largest xi is used throughout the
    scan. This avoids artificial spectral changes caused by adding and removing
    hopping pairs as xi changes.
    """
    if xi_min <= 0 or xi_max <= xi_min:
        raise ValueError("Require 0 < xi_min < xi_max")
    if number_of_xi < 2:
        raise ValueError("number_of_xi must be at least two")

    lattice = IncommensurateLattice1D(system_size=system_size)
    r_b = q * r_a
    lattice.generate_sublattices(
        periods=[r_a, r_b],
        offsets=[0.0, offset_b],
    )

    xi_values = np.linspace(xi_min, xi_max, number_of_xi)
    hopping_cutoff = cutoff_factor * xi_max

    all_energies = []
    all_ipr = []

    for xi in xi_values:
        hamiltonian = lattice.tight_binding_hamiltonian(
            xi=xi,
            intra_hopping=intra_hopping,
            inter_hopping=inter_hopping,
            onsite_energy=onsite_energy,
            hopping_cutoff=hopping_cutoff,
            include_long_range_intra=True,
        )

        energies, states = eigh(hamiltonian)
        ipr = lattice.inverse_participation_ratio(states)

        all_energies.append(energies)
        all_ipr.append(ipr)

    return lattice, xi_values, np.asarray(all_energies), np.asarray(all_ipr)


def plot_spectrum_vs_xi(
    xi_values: np.ndarray,
    energies: np.ndarray,
    ipr: np.ndarray,
    filename: Path,
):
    """Plot E_n(xi), using the inverse participation ratio as point color."""
    x = np.repeat(xi_values, energies.shape[1])
    y = energies.ravel()
    c = ipr.ravel()

    positive_ipr = c[c > 0]
    normalization = colors.LogNorm(
        vmin=max(float(positive_ipr.min()), 1e-6),
        vmax=float(positive_ipr.max()),
    )

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    scatter = ax.scatter(
        x,
        y,
        c=c,
        s=.5,
        norm=normalization,
        rasterized=True,
    )
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$E$")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("IPR")
    fig.tight_layout()
    fig.savefig(filename, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_ipr_summary(
    xi_values: np.ndarray,
    energies: np.ndarray,
    ipr: np.ndarray,
    filename: Path,
    central_fraction: float = 0.2,
):
    """Plot global and band-center IPR summaries as functions of xi."""
    if not 0 < central_fraction <= 1:
        raise ValueError("central_fraction must lie in (0, 1]")

    mean_ipr = np.mean(ipr, axis=1)
    median_ipr = np.median(ipr, axis=1)

    central_mean_ipr = np.empty_like(xi_values)
    for index, (energy_row, ipr_row) in enumerate(zip(energies, ipr)):
        energy_limit = np.quantile(np.abs(energy_row), central_fraction)
        mask = np.abs(energy_row) <= energy_limit
        central_mean_ipr[index] = np.mean(ipr_row[mask])

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(xi_values, mean_ipr, label="mean IPR")
    ax.plot(xi_values, median_ipr, label="median IPR")
    ax.plot(
        xi_values,
        central_mean_ipr,
        label=f"mean IPR near band center ({central_fraction:.0%})",
    )
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel("IPR")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main():
    q = np.sqrt(2.0)
    q = 1.4
    inter_hopping = 0.2

    lattice, xi_values, energies, ipr = scan_xi_spectrum(
        q=q,
        system_size=8000.0,
        r_a=20.0,
        offset_b=1.0,
        intra_hopping=(1.0, 1.0),
        inter_hopping=inter_hopping,
        onsite_energy=(0.0, 0.0),
        xi_min=0.5,
        xi_max=15.0,
        number_of_xi=200,
        cutoff_factor=8.0,
    )

    tag = f"q_{q:.5f}_g_{inter_hopping:.3f}".replace(".", "p")

    plot_spectrum_vs_xi(
        xi_values,
        energies,
        ipr,
        RESULTS_PATH / f"spectrum_vs_xi_{tag}.png",
    )
    plot_ipr_summary(
        xi_values,
        energies,
        ipr,
        RESULTS_PATH / f"ipr_vs_xi_{tag}.png",
    )
    

if __name__ == "__main__":
    main()
