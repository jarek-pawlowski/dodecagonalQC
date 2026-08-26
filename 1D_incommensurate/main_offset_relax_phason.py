
import os
import numpy as np
import matplotlib.pyplot as plt

import utils
from utils_offset_relax import (
    sweep_offsets,
    phason_vector_from_family,
    phason_diagnostics,
    plot_offset_scan,
    plot_phason_vector,
    remove_sublattice_polynomial_trend,
    hessian_spectral_maps,
    plot_hessian_spectral_map,
    phason_mode_overlap,
    quasiperiodic_reciprocal_module,
    plot_phason_mode_overlap,
    add_reciprocal_module_lines,
)


results_path = "./results_offset_scan/"
os.makedirs(results_path, exist_ok=True)

# ---------------------------------------------------------------------
# Fixed finite set of sites
# ---------------------------------------------------------------------

system_size = 2200.0
q = np.sqrt(2.0)

R_A = 20.0
R_B = q * R_A

# Build the site set once. During the scan we keep the same site indexing
# and shift only the reference position of sublattice B.
base_offset = 3.0

lattice = utils.RandomLattice1D(supercell_size=system_size)

lattice.generate_sublattices_with_bonds(
    periods=[R_A, R_B],
    offsets=[0.0, base_offset],
    inter_cutoff=R_B,
)

base_positions = lattice.positions.copy()
labels = lattice.labels.copy()

# ---------------------------------------------------------------------
# Potential parameters
# ---------------------------------------------------------------------

K_intra = [1.0, 1.0]
V_inter = 0.5
sigma_inter = 4.0

# Start with a relatively narrow interval around the chosen reference
# offset. Once this works robustly, the interval can be enlarged.
offsets = np.linspace(1.0, 20.0, 41)

scan = sweep_offsets(
    base_positions=base_positions,
    labels=labels,
    offsets=offsets,
    base_offset=base_offset,
    R_A=R_A,
    R_B=R_B,
    K_intra=K_intra,
    V_inter=V_inter,
    sigma_inter=sigma_inter,
    max_displacement=3.0,
    continuation=True,
    bulk_margin=5.0 * sigma_inter,
)

print("All optimizations successful:", np.all(scan["success"]))

delta_E = np.max(scan["energies"]) - np.min(scan["energies"])
delta_e = np.max(scan["energy_density"]) - np.min(scan["energy_density"])
delta_e_bulk = np.max(scan["bulk_energy_density"]) - np.min(scan["bulk_energy_density"])

print("Delta E =", delta_E)
print("Delta (E/L) =", delta_e)
print("Delta bulk energy density =", delta_e_bulk)

# ---------------------------------------------------------------------
# Microscopic phason tangent v^(w) = dX^0/dw0
# ---------------------------------------------------------------------

v_w = phason_vector_from_family(
    scan["relaxed"],
    scan["offsets"],
)

diagnostics = phason_diagnostics(
    scan["hessians"],
    v_w,
)

# ---------------------------------------------------------------------
# Summary output
# ---------------------------------------------------------------------

mid = len(offsets) // 2

print()
print("Reference offset:", offsets[mid])
print("E_min =", scan["energies"][mid])
print("E_min/L =", scan["energy_density"][mid])
print("Bulk E/L =", scan["bulk_energy_density"][mid])
print("Rayleigh quotient =", diagnostics["rayleigh"][mid])
print("Residual ||Hv||/||v|| =", diagnostics["residual"][mid])
print("Eigenvalue of max-overlap mode =", diagnostics["max_overlap_mode"][mid])
print("Maximum overlap =", diagnostics["max_overlap"][mid])
print("Lowest Hessian eigenvalues =", diagnostics["lowest_eigenvalues"][mid])

A = labels == 0
B = labels == 1

print("Mean v on A =", v_w[mid, A].mean())
print("Mean v on B =", v_w[mid, B].mean())
print("Norm v =", np.linalg.norm(v_w[mid]))

# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

plot_offset_scan(
    scan,
    diagnostics,
    results_path + "offset_scan",
)

v_residual, v_trend, coeff = (
    remove_sublattice_polynomial_trend(
        scan["relaxed"][mid],
        labels,
        v_w[mid],
        degree=2,
    )
)

fig, ax = plot_phason_vector(
    scan["relaxed"][mid],
    labels,
    v_w[mid],
    filename=results_path + "phason_vector_midpoint.png",
)
plt.close(fig)

fig, ax = plot_phason_vector(
    scan["relaxed"][mid],
    labels,
    v_residual,
    filename=results_path + "phason_vector_midpoint_trend.png",
)
plt.close(fig)


# ---------------------------------------------------------------------
# Hessian eigensystem and k-resolved spectral maps
# ---------------------------------------------------------------------

H_mid = scan["hessians"][mid]
x_mid = scan["relaxed"][mid]

# Full microscopic phason tangent at the selected offset.
v_full = v_w[mid].copy()

# Remove only the ordinary translation from the full phason probe.
v_full = v_full - np.mean(v_full)

# Quasiperiodic dressing obtained from polynomial detrending.
v_qp = v_residual.copy()

k_values_spec = np.linspace(
    0.0,
    np.pi / min(R_A, R_B),
    500,
)

spectra = hessian_spectral_maps(
    H_mid,
    x_mid,
    labels,
    k_values_spec,
    phason_vector=v_full,
    qp_vector=v_qp,
    normalize_sublattices=True,
)

# Relative A-B Fourier content.
plot_hessian_spectral_map(
    k_values_spec,
    spectra["eigenvalues"],
    spectra["I_relative"],
    results_path + "hessian_relative_fourier.png",
    title="Relative Fourier weight of Hessian eigenmodes",
    threshold=1e-5,
    point_scale=30.0,
)

# Common A+B Fourier content.
plot_hessian_spectral_map(
    k_values_spec,
    spectra["eigenvalues"],
    spectra["I_common"],
    results_path + "hessian_common_fourier.png",
    title="Common Fourier weight of Hessian eigenmodes",
    threshold=1e-5,
    point_scale=30.0,
)

# Full microscopic phason-adapted probe.
plot_hessian_spectral_map(
    k_values_spec,
    spectra["eigenvalues"],
    spectra["I_phason"],
    results_path + "hessian_phason_adapted.png",
    title="Microscopic phason spectral weight",
    threshold=1e-6,
    point_scale=30.0,
)

# Quasiperiodic dressing probe.
plot_hessian_spectral_map(
    k_values_spec,
    spectra["eigenvalues"],
    spectra["I_qp"],
    results_path + "hessian_qp_dressing.png",
    title="Quasiperiodic dressing spectral weight",
    threshold=1e-6,
    point_scale=30.0,
)

# Save the k-resolved spectral data.
np.savez(
    results_path + "hessian_spectral_maps.npz",
    k_values=k_values_spec,
    eigenvalues=spectra["eigenvalues"],
    I_A=spectra["I_A"],
    I_B=spectra["I_B"],
    I_common=spectra["I_common"],
    I_relative=spectra["I_relative"],
    I_phason=spectra["I_phason"],
    I_qp=spectra["I_qp"],
)

# Save all numerical data for later finite-size scaling.
np.savez(
    results_path + "offset_scan_data.npz",
    offsets=scan["offsets"],
    relaxed=scan["relaxed"],
    displacements=scan["displacements"],
    energies=scan["energies"],
    energy_density=scan["energy_density"],
    bulk_energies=scan["bulk_energies"],
    bulk_energy_density=scan["bulk_energy_density"],
    bulk_lengths=scan["bulk_lengths"],
    hessians=scan["hessians"],
    phason_vectors=v_w,
    rayleigh=diagnostics["rayleigh"],
    residual=diagnostics["residual"],
    lowest_eigenvalues=diagnostics["lowest_eigenvalues"],
    max_overlap_mode=diagnostics["max_overlap_mode"],
    max_overlap=diagnostics["max_overlap"],
    labels=labels,
    R_A=R_A,
    R_B=R_B,
    q=q,
    system_size=system_size,
    delta_E=delta_E,
    delta_e=delta_e,
    delta_e_bulk=delta_e_bulk,
)


# ---------------------------------------------------------------------
# Three complementary phason diagnostics
# ---------------------------------------------------------------------

# 1) Relative Fourier map:
#    I_rel(k,n) = |F_A(k,n) - F_B(k,n)|^2
# This is the most direct visualization of relative A/B motion.
plot_hessian_spectral_map(
    k_values_spec,
    spectra["eigenvalues"],
    spectra["I_relative"],
    results_path + "01_relative_fourier_map.png",
    title="Relative A-B Fourier weight",
    threshold=1e-5,
    point_scale=30.0,
)

# 2) Pure microscopic-phason overlap:
#    P_ph(n) = |<u_n | v_ph>|^2
# No extra k dependence is introduced here.
P_ph = phason_mode_overlap(
    spectra["eigenvectors"],
    v_full,
)

plot_phason_mode_overlap(
    spectra["eigenvalues"],
    P_ph,
    results_path + "02_phason_mode_overlap.png",
    max_modes=min(80, len(P_ph)),
)

imax_ph = int(np.argmax(P_ph))

print()
print("Pure microscopic phason overlap")
print("--------------------------------")
print("Mode with largest P_ph =", imax_ph)
print("Eigenvalue =", spectra["eigenvalues"][imax_ph])
print("P_ph =", P_ph[imax_ph])

# 3) Quasiperiodic dressing spectral map with reciprocal-module guides.
reciprocal_points = quasiperiodic_reciprocal_module(
    R_A,
    R_B,
    k_max=np.max(k_values_spec),
    max_order=10,
)

import matplotlib.colors as colors

fig, ax = plt.subplots(figsize=(8, 5))

I_qp = spectra["I_qp"]
positive = I_qp[I_qp > 1e-6]

if positive.size > 0:
    vmin = max(1e-6, np.percentile(positive, 5))
    vmax = np.percentile(positive, 99.5)
    norm_qp = colors.LogNorm(vmin=vmin, vmax=vmax)
else:
    norm_qp = None

for ik, k in enumerate(k_values_spec):
    mask = I_qp[ik] > 1e-6

    if not np.any(mask):
        continue

    local = I_qp[ik, mask]

    ax.scatter(
        np.full(mask.sum(), k),
        spectra["eigenvalues"][mask],
        c=local,
        s=2.0 + 28.0 * local / (np.max(local) + 1e-15),
        norm=norm_qp,
    )

selected_reciprocal_points = add_reciprocal_module_lines(
    ax,
    reciprocal_points,
    max_lines=14,
)

ax.set_xlabel(r"$k$")
ax.set_ylabel(r"$\omega^2$")
ax.set_title("Quasiperiodic dressing with reciprocal-module wavevectors")
ax.grid(alpha=0.15)

fig.tight_layout()
fig.savefig(
    results_path + "03_qp_dressing_with_Gmn.png",
    dpi=220,
    bbox_inches="tight",
)
plt.close(fig)

print()
print("Lowest-order reciprocal-module points shown:")
for point in selected_reciprocal_points:
    print(
        f"  k={point['k']:.8f}, "
        f"(m,n)=({point['m']},{point['n']}), "
        f"order={point['order']}"
    )

np.savez(
    results_path + "phason_three_diagnostics.npz",
    k_values=k_values_spec,
    eigenvalues=spectra["eigenvalues"],
    P_ph=P_ph,
    I_relative=spectra["I_relative"],
    I_qp=spectra["I_qp"],
    reciprocal_k=np.array(
        [p["k"] for p in reciprocal_points]
    ),
    reciprocal_m=np.array(
        [p["m"] for p in reciprocal_points]
    ),
    reciprocal_n=np.array(
        [p["n"] for p in reciprocal_points]
    ),
)
