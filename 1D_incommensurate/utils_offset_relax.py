
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eigh
import matplotlib.pyplot as plt


def total_energy(
    positions,
    labels,
    R_A,
    R_B,
    K_intra=(1.0, 1.0),
    V_inter=0.25,
    sigma_inter=3.0,
):
    """Return only the total potential energy."""
    E, _ = total_energy_and_gradient(
        positions, labels, R_A, R_B, K_intra, V_inter, sigma_inter
    )
    return E



def total_energy_and_gradient(
    positions,
    labels,
    R_A,
    R_B,
    K_intra=(1.0, 1.0),
    V_inter=0.25,
    sigma_inter=3.0,
):
    """Return total potential energy and its analytic gradient."""
    x = np.asarray(positions, dtype=float)
    labels = np.asarray(labels)

    A = np.flatnonzero(labels == 0)
    B = np.flatnonzero(labels == 1)

    K_A, K_B = float(K_intra[0]), float(K_intra[1])
    grad = np.zeros_like(x)
    energy = 0.0

    for indices, R0, K in ((A, R_A, K_A), (B, R_B, K_B)):
        for i, j in zip(indices[:-1], indices[1:]):
            stretch = (x[j] - x[i]) - R0
            energy += 0.5 * K * stretch**2
            g = K * stretch
            grad[i] -= g
            grad[j] += g

    sigma2 = float(sigma_inter) ** 2
    delta = x[A][:, None] - x[B][None, :]
    phi = float(V_inter) * np.exp(-0.5 * delta**2 / sigma2)
    energy += np.sum(phi)

    dphi_ddelta = -(delta / sigma2) * phi
    grad[A] += np.sum(dphi_ddelta, axis=1)
    grad[B] -= np.sum(dphi_ddelta, axis=0)

    return float(energy), grad


def bulk_energy(
    positions,
    labels,
    R_A,
    R_B,
    K_intra=(1.0, 1.0),
    V_inter=0.25,
    sigma_inter=3.0,
    margin=None,
):
    """Estimate bulk energy by excluding edge regions."""
    x = np.asarray(positions, dtype=float)
    labels = np.asarray(labels)

    if margin is None:
        margin = 5.0 * float(sigma_inter)

    xmin, xmax = np.min(x), np.max(x)
    left, right = xmin + margin, xmax - margin
    if right <= left:
        raise ValueError("Bulk margin is too large for this system.")

    A = np.flatnonzero(labels == 0)
    B = np.flatnonzero(labels == 1)
    K_A, K_B = float(K_intra[0]), float(K_intra[1])

    energy = 0.0

    for indices, R0, K in ((A, R_A, K_A), (B, R_B, K_B)):
        for i, j in zip(indices[:-1], indices[1:]):
            midpoint = 0.5 * (x[i] + x[j])
            if left <= midpoint <= right:
                stretch = (x[j] - x[i]) - R0
                energy += 0.5 * K * stretch**2

    sigma2 = float(sigma_inter) ** 2
    for i in A:
        for j in B:
            midpoint = 0.5 * (x[i] + x[j])
            if left <= midpoint <= right:
                delta = x[i] - x[j]
                energy += float(V_inter) * np.exp(-0.5 * delta**2 / sigma2)

    return float(energy), float(right - left)


def relax_from_reference(
    x_ref,
    labels,
    R_A,
    R_B,
    K_intra=(1.0, 1.0),
    V_inter=0.25,
    sigma_inter=3.0,
    max_displacement=3.0,
    y0=None,
):
    """
    Relax at fixed relative offset using explicit symmetric constraints.

    Each site displacement is optimized independently. Zero mean displacement
    is imposed separately on A and B, so no single endpoint absorbs the
    constraint as system size increases.
    """
    x_ref = np.asarray(x_ref, dtype=float)
    labels = np.asarray(labels)

    A = np.flatnonzero(labels == 0)
    B = np.flatnonzero(labels == 1)
    n = len(x_ref)

    if y0 is None:
        y0 = np.zeros(n, dtype=float)
    else:
        y0 = np.asarray(y0, dtype=float).copy()
        if len(y0) != n:
            raise ValueError("y0 has the wrong size.")

    def objective(dx):
        E, _ = total_energy_and_gradient(
            x_ref + dx, labels, R_A, R_B, K_intra, V_inter, sigma_inter
        )
        return E

    def jacobian(dx):
        _, grad = total_energy_and_gradient(
            x_ref + dx, labels, R_A, R_B, K_intra, V_inter, sigma_inter
        )
        return grad

    jac_A = np.zeros(n, dtype=float)
    jac_B = np.zeros(n, dtype=float)
    jac_A[A] = 1.0 / len(A)
    jac_B[B] = 1.0 / len(B)

    constraints = [
        {"type": "eq", "fun": lambda dx: np.mean(dx[A]), "jac": lambda dx: jac_A},
        {"type": "eq", "fun": lambda dx: np.mean(dx[B]), "jac": lambda dx: jac_B},
    ]

    bounds = None
    if max_displacement is not None:
        md = float(max_displacement)
        bounds = [(-md, md)] * n

    result = minimize(
        objective,
        y0,
        method="SLSQP",
        jac=jacobian,
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 5000, "ftol": 1e-12, "disp": False},
    )

    dx_relaxed = result.x.copy()
    x_relaxed = x_ref + dx_relaxed
    return result, x_relaxed, dx_relaxed


def hessian_matrix(
    positions,
    labels,
    R_A,
    R_B,
    K_intra=(1.0, 1.0),
    V_inter=0.25,
    sigma_inter=3.0,
):
    """
    Analytic Hessian of total_energy with respect to all site positions.

    For an intra-sublattice harmonic bond the curvature is simply K_s.

    For the Gaussian A-B interaction
        V(delta) = V_inter * exp[-delta^2 / (2 sigma^2)],
    the curvature with respect to delta is
        V''(delta) = V(delta) * (delta^2/sigma^4 - 1/sigma^2).

    Each pair contributes the standard 2x2 Hessian block
        [[+c, -c],
         [-c, +c]],
    where c is the corresponding pair curvature.
    """
    x = np.asarray(positions, dtype=float)
    labels = np.asarray(labels)

    A = np.flatnonzero(labels == 0)
    B = np.flatnonzero(labels == 1)

    n = len(x)
    H = np.zeros((n, n), dtype=float)

    def add_pair_curvature(i, j, curvature):
        H[i, i] += curvature
        H[j, j] += curvature
        H[i, j] -= curvature
        H[j, i] -= curvature

    # Intra-sublattice harmonic bonds.
    for indices, K in ((A, float(K_intra[0])), (B, float(K_intra[1]))):
        for i, j in zip(indices[:-1], indices[1:]):
            add_pair_curvature(int(i), int(j), K)

    # Inter-sublattice Gaussian pair potential.
    sigma2 = sigma_inter ** 2
    sigma4 = sigma2 ** 2

    for i in A:
        for j in B:
            delta = x[i] - x[j]
            V = V_inter * np.exp(-0.5 * delta**2 / sigma2)
            curvature = V * (delta**2 / sigma4 - 1.0 / sigma2)
            add_pair_curvature(int(i), int(j), curvature)

    return H


def make_reference_family(base_positions, labels, offsets, base_offset):
    """
    Construct a family of reference configurations with fixed site indexing.

    Sublattice A is kept unchanged. Sublattice B is rigidly shifted by
    offset - base_offset. The atom count therefore remains identical for
    the entire offset scan.
    """
    base_positions = np.asarray(base_positions, dtype=float)
    labels = np.asarray(labels)
    B = labels == 1

    refs = []

    for offset in offsets:
        x_ref = base_positions.copy()
        x_ref[B] += float(offset) - float(base_offset)
        refs.append(x_ref)

    return np.asarray(refs)


def sweep_offsets(
    base_positions,
    labels,
    offsets,
    base_offset,
    R_A,
    R_B,
    K_intra=(1.0, 1.0),
    V_inter=0.25,
    sigma_inter=3.0,
    max_displacement=3.0,
    continuation=True,
    bulk_margin=None,
):
    """
    Relax the same finite set of sites for a sequence of global offsets.

    Returns a dictionary containing relaxed positions, energies, Hessians
    and optimizer diagnostics.
    """
    offsets = np.asarray(offsets, dtype=float)
    labels = np.asarray(labels)

    references = make_reference_family(
        base_positions,
        labels,
        offsets,
        base_offset,
    )

    n_offsets, n_sites = references.shape

    relaxed = np.zeros_like(references)
    displacements = np.zeros_like(references)
    energies = np.zeros(n_offsets)
    energy_density = np.zeros(n_offsets)
    bulk_energies = np.zeros(n_offsets)
    bulk_energy_density = np.zeros(n_offsets)
    bulk_lengths = np.zeros(n_offsets)
    hessians = np.zeros((n_offsets, n_sites, n_sites))
    success = np.zeros(n_offsets, dtype=bool)

    y_previous = None

    for iw, x_ref in enumerate(references):
        result, x_relaxed, dx_relaxed = relax_from_reference(
            x_ref,
            labels,
            R_A=R_A,
            R_B=R_B,
            K_intra=K_intra,
            V_inter=V_inter,
            sigma_inter=sigma_inter,
            max_displacement=max_displacement,
            y0=y_previous if continuation else None,
        )

        relaxed[iw] = x_relaxed
        displacements[iw] = dx_relaxed
        energies[iw] = total_energy(
            x_relaxed,
            labels,
            R_A=R_A,
            R_B=R_B,
            K_intra=K_intra,
            V_inter=V_inter,
            sigma_inter=sigma_inter,
        )

        physical_length = np.max(x_relaxed) - np.min(x_relaxed)
        energy_density[iw] = energies[iw] / physical_length

        E_bulk, L_bulk = bulk_energy(
            x_relaxed,
            labels,
            R_A=R_A,
            R_B=R_B,
            K_intra=K_intra,
            V_inter=V_inter,
            sigma_inter=sigma_inter,
            margin=bulk_margin,
        )
        bulk_energies[iw] = E_bulk
        bulk_lengths[iw] = L_bulk
        bulk_energy_density[iw] = E_bulk / L_bulk
        hessians[iw] = hessian_matrix(
            x_relaxed,
            labels,
            R_A=R_A,
            R_B=R_B,
            K_intra=K_intra,
            V_inter=V_inter,
            sigma_inter=sigma_inter,
        )
        success[iw] = result.success

        if continuation:
            y_previous = result.x.copy()

    return {
        "offsets": offsets,
        "references": references,
        "relaxed": relaxed,
        "displacements": displacements,
        "energies": energies,
        "energy_density": energy_density,
        "bulk_energies": bulk_energies,
        "bulk_energy_density": bulk_energy_density,
        "bulk_lengths": bulk_lengths,
        "hessians": hessians,
        "success": success,
    }


def phason_vector_from_family(relaxed_positions, offsets):
    """
    Compute v^(w) = d X^0 / d w0 from the relaxed family.

    np.gradient uses centered differences in the bulk and one-sided
    differences at the endpoints.
    """
    relaxed_positions = np.asarray(relaxed_positions, dtype=float)
    offsets = np.asarray(offsets, dtype=float)

    return np.gradient(
        relaxed_positions,
        offsets,
        axis=0,
        edge_order=2,
    )


def phason_diagnostics(hessians, phason_vectors):
    """
    Evaluate how close each phason tangent vector is to a Hessian zero mode.

    Returns
    -------
    rayleigh : ndarray
        v^T H v / v^T v.
    residual : ndarray
        ||H v|| / ||v||.
    lowest_eigenvalues : ndarray
        Lowest five eigenvalues of each Hessian.
    max_overlap_mode : ndarray
        Hessian eigenvalue of the eigenmode with the largest overlap with v.
    max_overlap : ndarray
        Corresponding normalized squared overlap.
    """
    hessians = np.asarray(hessians)
    phason_vectors = np.asarray(phason_vectors)

    n_offsets = len(hessians)
    n_low = min(5, hessians.shape[1])

    rayleigh = np.zeros(n_offsets)
    residual = np.zeros(n_offsets)
    lowest_eigenvalues = np.zeros((n_offsets, n_low))
    max_overlap_mode = np.zeros(n_offsets)
    max_overlap = np.zeros(n_offsets)

    for iw, (H, v) in enumerate(zip(hessians, phason_vectors)):
        norm = np.linalg.norm(v)

        if norm < 1e-14:
            rayleigh[iw] = np.nan
            residual[iw] = np.nan
            lowest_eigenvalues[iw] = np.nan
            max_overlap_mode[iw] = np.nan
            max_overlap[iw] = np.nan
            continue

        vn = v / norm

        Hv = H @ vn
        rayleigh[iw] = vn @ Hv
        residual[iw] = np.linalg.norm(Hv)

        evals, evecs = eigh(H)
        lowest_eigenvalues[iw] = evals[:n_low]

        overlaps = np.abs(evecs.T @ vn) ** 2
        imax = np.argmax(overlaps)

        max_overlap_mode[iw] = evals[imax]
        max_overlap[iw] = overlaps[imax]

    return {
        "rayleigh": rayleigh,
        "residual": residual,
        "lowest_eigenvalues": lowest_eigenvalues,
        "max_overlap_mode": max_overlap_mode,
        "max_overlap": max_overlap,
    }


def remove_sublattice_polynomial_trend(
    positions,
    labels,
    v,
    degree=2,
):
    positions = np.asarray(positions)
    labels = np.asarray(labels)
    v = np.asarray(v)

    residual = np.zeros_like(v)
    trend = np.zeros_like(v)
    coefficients = {}

    for s in [0, 1]:
        mask = labels == s

        x = positions[mask]
        y = v[mask]

        coeff = np.polyfit(x, y, deg=degree)
        fit = np.polyval(coeff, x)

        trend[mask] = fit
        residual[mask] = y - fit
        coefficients[s] = coeff

    return residual, trend, coefficients


def plot_offset_scan(scan, diagnostics, filename_prefix):
    """Create summary plots for the offset scan."""
    offsets = scan["offsets"]
    energies = scan["energies"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(offsets, energies, "o-")
    ax.set_xlabel(r"$w_0$ / offset")
    ax.set_ylabel(r"$E_{\min}(w_0)$")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(filename_prefix + "_energy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(offsets, scan["energy_density"], "o-", label="total energy density")
    ax.plot(offsets, scan["bulk_energy_density"], "s-", label="bulk energy density")
    ax.set_xlabel(r"$w_0$ / offset")
    ax.set_ylabel("energy / length")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(filename_prefix + "_energy_density.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(offsets, diagnostics["rayleigh"], "o-", label=r"$v^T H v / v^T v$")
    ax.plot(offsets, diagnostics["max_overlap_mode"], "s-", label="eigenvalue of max-overlap mode")
    ax.set_xlabel(r"$w_0$ / offset")
    ax.set_ylabel("curvature / eigenvalue")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(filename_prefix + "_phason_curvature.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(offsets, diagnostics["residual"], "o-")
    ax.set_xlabel(r"$w_0$ / offset")
    ax.set_ylabel(r"$\|Hv^{(w)}\|/\|v^{(w)}\|$")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(filename_prefix + "_phason_residual.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_phason_vector(
    positions,
    labels,
    phason_vector,
    filename=None,
):
    """Plot the microscopic phason tangent vector on both sublattices."""
    positions = np.asarray(positions)
    labels = np.asarray(labels)
    v = np.asarray(phason_vector)

    A = labels == 0
    B = labels == 1

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(positions[A], v[A], "o-", label="Sublattice A")
    ax.plot(positions[B], v[B], "o-", label="Sublattice B")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel("relaxed position")
    ax.set_ylabel(r"$v_i^{(w)} = \partial X_i^0/\partial w_0$")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename, dpi=200, bbox_inches="tight")

    return fig, ax



def hessian_spectral_maps(
    H,
    positions,
    labels,
    k_values,
    phason_vector=None,
    qp_vector=None,
    normalize_sublattices=True,
):
    """
    Compute Hessian eigensystem and several k-resolved spectral weights.

    Parameters
    ----------
    H : (N, N) ndarray
        Hessian / dynamical matrix.
    positions : (N,) ndarray
        Relaxed positions X_i^0.
    labels : (N,) ndarray
        Sublattice labels, 0 for A and 1 for B.
    k_values : (Nk,) ndarray
        Wavevectors at which Fourier probes are evaluated.
    phason_vector : (N,) ndarray, optional
        Full microscopic phason tangent v^(w).
    qp_vector : (N,) ndarray, optional
        Quasiperiodic dressing part of the phason tangent.
    normalize_sublattices : bool
        If True, normalize A/B Fourier amplitudes by sqrt(N_A/B).

    Returns
    -------
    dict with eigenvalues, eigenvectors and spectral intensities:
        I_common(k,n)
        I_relative(k,n)
        I_phason(k,n)      if phason_vector is provided
        I_qp(k,n)          if qp_vector is provided
    """
    H = np.asarray(H, dtype=float)
    x = np.asarray(positions, dtype=float)
    labels = np.asarray(labels)
    k_values = np.asarray(k_values, dtype=float)

    evals, evecs = eigh(H)

    A = labels == 0
    B = labels == 1

    phase_A = np.exp(-1j * np.outer(k_values, x[A]))
    phase_B = np.exp(-1j * np.outer(k_values, x[B]))

    F_A = phase_A @ evecs[A, :]
    F_B = phase_B @ evecs[B, :]

    if normalize_sublattices:
        F_A = F_A / np.sqrt(A.sum())
        F_B = F_B / np.sqrt(B.sum())

    F_common = F_A + F_B
    F_relative = F_A - F_B

    result = {
        "eigenvalues": evals,
        "eigenvectors": evecs,
        "F_A": F_A,
        "F_B": F_B,
        "I_A": np.abs(F_A) ** 2,
        "I_B": np.abs(F_B) ** 2,
        "I_common": np.abs(F_common) ** 2,
        "I_relative": np.abs(F_relative) ** 2,
    }

    def weighted_probe_intensity(weight_vector):
        w = np.asarray(weight_vector, dtype=float).copy()

        # Remove ordinary rigid translation from the probe.
        w = w - np.mean(w)

        norm = np.linalg.norm(w)
        if norm < 1e-14:
            raise ValueError("Probe vector has vanishing norm.")

        w = w / norm

        weighted_phase = np.exp(
            -1j * np.outer(k_values, x)
        ) * w[None, :]

        amplitude = weighted_phase.conj() @ evecs
        return np.abs(amplitude) ** 2

    if phason_vector is not None:
        result["I_phason"] = weighted_probe_intensity(
            phason_vector
        )

    if qp_vector is not None:
        result["I_qp"] = weighted_probe_intensity(
            qp_vector
        )

    return result


def plot_hessian_spectral_map(
    k_values,
    eigenvalues,
    intensity,
    filename,
    ylabel=r"$\omega^2$",
    title=None,
    threshold=1e-6,
    point_scale=20.0,
    log_color=True,
    ylim=None,
):
    """
    Scatter plot of Hessian eigenvalues versus k, colored by spectral weight.
    """
    import matplotlib.colors as colors

    k_values = np.asarray(k_values)
    eigenvalues = np.asarray(eigenvalues)
    intensity = np.asarray(intensity)

    fig, ax = plt.subplots(figsize=(8, 5))

    positive = intensity[intensity > threshold]

    if log_color and positive.size > 0:
        vmin = max(threshold, np.percentile(positive, 5))
        vmax = np.percentile(positive, 99.5)
        if vmax <= vmin:
            vmax = positive.max()
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None

    for ik, k in enumerate(k_values):
        mask = intensity[ik] > threshold

        if not np.any(mask):
            continue

        ax.scatter(
            np.full(mask.sum(), k),
            eigenvalues[mask],
            c=intensity[ik, mask],
            s=point_scale * intensity[ik, mask] / (
                np.max(intensity[ik, mask]) + 1e-15
            ) + 2.0,
            norm=norm,
        )

    ax.set_xlabel(r"$k$")
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def phason_mode_overlap(eigenvectors, phason_vector):
    """
    Compute the k-independent overlap of every Hessian eigenmode with the
    microscopic phason tangent.

    The ordinary translational component is removed from the phason probe.

        P_ph(n) = | <u_n | v_ph> |^2
    """
    evecs = np.asarray(eigenvectors, dtype=float)
    v = np.asarray(phason_vector, dtype=float).copy()

    v -= np.mean(v)

    norm = np.linalg.norm(v)
    if norm < 1e-14:
        raise ValueError("Phason probe has vanishing norm.")

    v /= norm

    return np.abs(evecs.T @ v) ** 2


def quasiperiodic_reciprocal_module(
    R_A,
    R_B,
    k_max,
    max_order=8,
    tolerance=1e-10,
):
    """
    Generate small positive reciprocal-module wavevectors

        G_mn = |m G_A + n G_B|,

    with
        G_A = 2 pi / R_A,
        G_B = 2 pi / R_B,

    restricted to 0 < G_mn <= k_max.

    Duplicate values are merged numerically. The returned labels contain
    the lowest-order integer pair found for each distinct wavevector.
    """
    G_A = 2.0 * np.pi / float(R_A)
    G_B = 2.0 * np.pi / float(R_B)

    candidates = []

    for m in range(-max_order, max_order + 1):
        for n in range(-max_order, max_order + 1):
            if m == 0 and n == 0:
                continue

            g = abs(m * G_A + n * G_B)

            if g <= tolerance or g > k_max + tolerance:
                continue

            order = abs(m) + abs(n)
            candidates.append((g, order, m, n))

    candidates.sort(key=lambda item: (item[0], item[1]))

    unique = []

    for g, order, m, n in candidates:
        if len(unique) == 0 or abs(g - unique[-1][0]) > tolerance:
            unique.append([g, order, m, n])
        elif order < unique[-1][1]:
            unique[-1] = [g, order, m, n]

    return [
        {
            "k": float(g),
            "m": int(m),
            "n": int(n),
            "order": int(order),
            "label": f"({m},{n})",
        }
        for g, order, m, n in unique
    ]


def plot_phason_mode_overlap(
    eigenvalues,
    overlaps,
    filename,
    max_modes=None,
    ylim=None,
):
    """
    Plot Hessian eigenvalues versus mode index with marker size proportional
    to the k-independent microscopic phason overlap.
    """
    eigenvalues = np.asarray(eigenvalues)
    overlaps = np.asarray(overlaps)

    if max_modes is None:
        max_modes = len(eigenvalues)

    idx = np.arange(min(max_modes, len(eigenvalues)))
    vals = eigenvalues[idx]
    ovs = overlaps[idx]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.scatter(
        idx,
        vals,
        s=20.0 + 350.0 * ovs,
        c=ovs,
    )

    ax.set_xlabel("Hessian eigenmode index")
    ax.set_ylabel(r"$\omega_n^2$")
    ax.set_title(r"$P_{\rm ph}(n)=|\langle u_n|v_{\rm ph}\rangle|^2$")

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def add_reciprocal_module_lines(
    ax,
    reciprocal_points,
    max_lines=12,
):
    """
    Add vertical guides at the lowest-order quasiperiodic reciprocal-module
    wavevectors.
    """
    points = sorted(
        reciprocal_points,
        key=lambda item: (item["order"], item["k"]),
    )[:max_lines]

    for point in points:
        k = point["k"]
        label = point["label"]

        ax.axvline(
            k,
            linestyle="--",
            linewidth=0.8,
            alpha=0.55,
        )

        ymin, ymax = ax.get_ylim()
        ax.text(
            k,
            ymax,
            label,
            rotation=90,
            va="top",
            ha="right",
            fontsize=7,
            alpha=0.75,
        )

    return points
