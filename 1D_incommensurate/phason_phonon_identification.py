#!/usr/bin/env python3
"""
phason_q_vint_hessian_map.py

Numerical checks for the two-incommensurate-chain model:

  (1) Direct microscopic block check:
          ||H_AB||_F / sqrt(||H_AA||_F ||H_BB||_F) -> 0 as V_inter -> 0.

  (2) Full (q, V_inter) map of EXACT Hessian eigenmodes.
      For every V_inter we relax the offset family, build the full Hessian,
      diagonalize it ONCE, and for every q identify/track the two exact modes
      carrying the long-wavelength A/B (= P/W) sector.

  (3) Maps of raw overlaps with A_q, B_q, P_q, W_q and lambda/q^2.

  (4) P/W-coordinate mixing-angle maps (kept as a representation diagnostic).

  (5) Basis-independent exact-mode rotation
          Theta_ref(q,V)=acos(|<e(q,0)|e(q,V)>|)
      plus a signed microscopic A/B hybridization rotation.  These are the
      preferred diagnostics for interaction-induced rotation of the two
      acoustic branches.

  (6) Explicit phenomenological inter-sublattice damping test.  We integrate
      u_ddot + gamma_AB |R_q><R_q| u_dot + H u = 0 for each tracked exact
      branch at q_min, and fit the decay of its modal-amplitude envelope to
      exp(-Gamma t).  R_q=(Ahat_q-Bhat_q)/sqrt(2), so only relative A/B
      velocity is damped while common translation remains undamped.

Important:
  * The projected probes are diagnostics only.  We NEVER diagonalize a 2x2
    projected Hessian to define the branches.
  * At V_inter=0 the branches are initialized as the exact modes with maximal
    A_q and B_q overlap, respectively.  For each fixed q they are then tracked
    continuously in V_inter by exact-eigenvector overlap.
  * P_q is the common displacement envelope. W_q is constructed from the
    relaxed-family tangent dX0/dw0 with its global translation removed.

Put this file next to utils.py and utils_offset_relax.py.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.linalg import expm_multiply
from scipy.signal import find_peaks

import utils
from utils_offset_relax import sweep_offsets, phason_vector_from_family


# =====================================================================
# Configuration
# =====================================================================

OUT = Path("./results_phason_q_vint_map")
OUT.mkdir(parents=True, exist_ok=True)

system_size = 2200.0
R_A = 20.0
R_B = np.sqrt(2.0) * R_A
base_offset = 3.0
K_intra = [1.0, 1.0]
V0 = 0.5
sigma_inter = 4.0

# Denser close to the decoupled limit.
G_RATIOS = np.array([
    0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00
])

# q_n = 2*pi*n/L.  q indices 1..16 reproduce roughly the range used before.
Q_INDICES = np.arange(1, 17, dtype=int)

# Relaxed family used to obtain the true phason tangent.
offsets = np.linspace(1.0, 20.0, 41)
mid = len(offsets) // 2

# Number of low exact Hessian modes considered when matching a q sector.
# Increase if the highest q starts hitting this ceiling.
N_EXACT_KEEP = 80
N_CANDIDATES = 16
S_TOL = 1e-11


# =====================================================================
# Fixed microscopic site set
# =====================================================================

lattice = utils.RandomLattice1D(supercell_size=system_size)
lattice.generate_sublattices_with_bonds(
    periods=[R_A, R_B],
    offsets=[0.0, base_offset],
    inter_cutoff=R_B,
)

base_positions = lattice.positions.copy()
labels = lattice.labels.copy()
A_mask = labels == 0
B_mask = labels == 1
N = len(labels)

print(f"N sites = {N} (A={A_mask.sum()}, B={B_mask.sum()})")


# =====================================================================
# Linear-algebra helpers
# =====================================================================

def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-30:
        raise RuntimeError("Attempt to normalize a zero vector")
    return v / n


def remove_translation(v):
    v = np.asarray(v, dtype=float).copy()
    t = np.ones_like(v)
    return v - t * (t @ v) / (t @ t)


def orthonormal_basis(*vectors, tol=S_TOL):
    """Lowdin orthonormal basis spanning the supplied vectors."""
    B = np.column_stack(vectors)
    S = 0.5 * (B.T @ B + (B.T @ B).T)
    se, U = np.linalg.eigh(S)
    keep = se > tol * max(float(se.max()), 1e-30)
    if not np.any(keep):
        raise RuntimeError("Probe subspace has zero numerical rank")
    return B @ U[:, keep] @ np.diag(1.0 / np.sqrt(se[keep]))


def projector_weights(evecs, Q):
    return np.sum(np.abs(Q.T @ evecs) ** 2, axis=0)


def raw_overlap(evecs, probe):
    p = normalize(probe)
    return np.abs(evecs.T @ p) ** 2


def pw_orthonormal_frame(Pq, Wq, tol=S_TOL):
    """
    Build an interpretable orthonormal P/W frame.

    The first axis is exactly the normalized phonon probe Pq.
    The second axis is the component of Wq orthogonal to Pq.
    This avoids treating raw P/W overlaps as probabilities when Pq and Wq
    are not orthogonal.
    """
    p = normalize(Pq)
    w = np.asarray(Wq, dtype=float)
    w_perp = w - p * (p @ w)
    nw = np.linalg.norm(w_perp)
    if nw < tol * max(np.linalg.norm(w), 1.0):
        raise RuntimeError("Pq and Wq are numerically linearly dependent")
    return p, w_perp / nw


def pw_mixing_diagnostics(evecs, Pq, Wq, chosen, eps=1e-15):
    """
    Return orthogonal P/W weights and a label-independent mixing angle.

    theta_mix = atan2(sqrt(min(wP,wW)), sqrt(max(wP,wW))) in [0,45] deg.
    Therefore theta=0 means a pure axis of the P/W frame and theta=45 deg
    means equal P/W mixing.
    """
    p, w = pw_orthonormal_frame(Pq, Wq)
    E = evecs[:, np.asarray(chosen, dtype=int)]
    wP = np.abs(p @ E) ** 2
    wW = np.abs(w @ E) ** 2
    hi = np.maximum(wP, wW)
    lo = np.minimum(wP, wW)
    theta = np.full_like(hi, np.nan, dtype=float)
    good = hi > eps
    theta[good] = np.degrees(np.arctan2(np.sqrt(lo[good]), np.sqrt(hi[good])))
    return wP, wW, theta


def build_probes(X0, v_ph, sector, q):
    xc = X0 - 0.5 * (X0.max() + X0.min())
    if sector == "cos":
        env = np.cos(q * xc)
    elif sector == "sin":
        env = np.sin(q * xc)
    else:
        raise ValueError(sector)

    Aq = np.zeros_like(X0)
    Bq = np.zeros_like(X0)
    Aq[A_mask] = env[A_mask]
    Bq[B_mask] = env[B_mask]

    Pq = env.copy()
    Wq = v_ph * env
    return Aq, Bq, Pq, Wq


def select_initial_AB(evecs, Aq, Bq, nkeep):
    """At V=0 initialize branch 0=A phonon and branch 1=B phonon."""
    oa = raw_overlap(evecs[:, :nkeep], Aq)
    ob = raw_overlap(evecs[:, :nkeep], Bq)
    score = np.vstack([oa, ob])
    rows, cols = linear_sum_assignment(-score)
    chosen = np.empty(2, dtype=int)
    chosen[rows] = cols
    return chosen


def hydro_candidates(evecs, Aq, Bq, Pq, Wq, nkeep, ncand):
    """Candidate exact modes ranked by long-wave projector weight."""
    QAB = orthonormal_basis(Aq, Bq)
    QPW = orthonormal_basis(Pq, Wq)
    wab = projector_weights(evecs[:, :nkeep], QAB)
    wpw = projector_weights(evecs[:, :nkeep], QPW)
    score = 0.5 * (wab + wpw)
    order = np.argsort(score)[::-1]
    return order[:min(ncand, len(order))]


def track_two(prev_vecs, evecs, candidates):
    C = evecs[:, candidates]
    score = np.abs(prev_vecs.T @ C) ** 2
    rows, cols = linear_sum_assignment(-score)
    chosen = np.empty(2, dtype=int)
    chosen[rows] = np.asarray(candidates, dtype=int)[cols]
    continuity = np.empty(2, dtype=float)
    continuity[rows] = score[rows, cols]
    return chosen, continuity


def hessian_block_diagnostic(H):
    """Direct matrix-level A/B block norm diagnostic."""
    HAA = H[np.ix_(A_mask, A_mask)]
    HBB = H[np.ix_(B_mask, B_mask)]
    HAB = H[np.ix_(A_mask, B_mask)]
    HBA = H[np.ix_(B_mask, A_mask)]

    nAA = np.linalg.norm(HAA, ord="fro")
    nBB = np.linalg.norm(HBB, ord="fro")
    nAB = np.linalg.norm(HAB, ord="fro")
    nBA = np.linalg.norm(HBA, ord="fro")
    denom = np.sqrt(nAA * nBB)
    ratio = nAB / denom if denom > 0 else np.nan
    symmetry = np.linalg.norm(HAB - HBA.T, ord="fro")
    return nAA, nBB, nAB, nBA, ratio, symmetry


# =====================================================================
# Storage
# =====================================================================

sectors = ("cos", "sin")
ng = len(G_RATIOS)
nq = len(Q_INDICES)

# q varies weakly with V because L changes after relaxation.
q_grid = np.full((ng, nq), np.nan)

# Direct H block diagnostics.
HAA_norm = np.full(ng, np.nan)
HBB_norm = np.full(ng, np.nan)
HAB_norm = np.full(ng, np.nan)
HBA_norm = np.full(ng, np.nan)
HAB_ratio = np.full(ng, np.nan)
HAB_symmetry_error = np.full(ng, np.nan)

# [sector][quantity] arrays have shape (ng,nq,2), except mode index.
# Cache exact physical-coupling modes for the final time-domain damping test.
# Filled only at the grid point closest to V_inter/V0 = 1.
physical_mode_cache = {"cos": {}, "sin": {}}

data = {}
for sec in sectors:
    data[sec] = {
        "mode_idx": np.full((ng, nq, 2), -1, dtype=int),
        "lambda": np.full((ng, nq, 2), np.nan),
        "lambda_over_q2": np.full((ng, nq, 2), np.nan),
        "continuity": np.full((ng, nq, 2), np.nan),
        "A": np.full((ng, nq, 2), np.nan),
        "B": np.full((ng, nq, 2), np.nan),
        "P": np.full((ng, nq, 2), np.nan),
        "W": np.full((ng, nq, 2), np.nan),
        "ABspan": np.full((ng, nq, 2), np.nan),
        "PWspan": np.full((ng, nq, 2), np.nan),
        "Porth": np.full((ng, nq, 2), np.nan),
        "Worth": np.full((ng, nq, 2), np.nan),
        "theta_mix_deg": np.full((ng, nq, 2), np.nan),

        # New, basis-independent diagnostics:
        # theta_ref_deg: angle of the exact tracked eigenvector relative to
        # its own V_inter=0 exact eigenvector.  Thus theta_ref_deg=0 at V=0.
        #
        # theta_AB_deg: signed angle of the projection of the exact mode
        # onto the orthonormal microscopic (A_q,B_q) frame:
        # atan2(<Bhat|e>, <Ahat|e>).  The sign/gauge is fixed continuously
        # along the V sweep.
        "theta_ref_deg": np.full((ng, nq, 2), np.nan),
        "theta_AB_deg": np.full((ng, nq, 2), np.nan),
        "AB_proj_norm": np.full((ng, nq, 2), np.nan),
        "damping_factor_AB": np.full((ng, nq, 2), np.nan),
    }

# Previous exact vectors for branch tracking, independently for each q and sector.
previous = {
    sec: [[None, None] for _ in range(nq)]
    for sec in sectors
}

# Exact V_inter=0 reference eigenvectors for every (sector,q,branch).
reference = {
    sec: [[None, None] for _ in range(nq)]
    for sec in sectors
}


def sign_align(v, ref):
    """Fix the arbitrary real-eigenvector sign by positive overlap with ref."""
    v = np.asarray(v, dtype=float).copy()
    if float(ref @ v) < 0.0:
        v *= -1.0
    return v


def signed_AB_angle(e, Aq, Bq, eps=1e-14):
    """
    Signed coordinate angle of e in the orthonormal A/B frame.

    Since Aq and Bq have disjoint support, they are already orthogonal;
    normalizing them gives a particularly transparent microscopic frame.

    angle = atan2(<Bhat|e>, <Ahat|e>) in degrees.
    Also return the norm of the A/B projection, which tells us whether
    this angle is meaningful for the exact eigenmode.
    """
    ahat = normalize(Aq)
    bhat = normalize(Bq)
    a = float(ahat @ e)
    b = float(bhat @ e)
    rho = np.hypot(a, b)
    if rho < eps:
        return np.nan, rho
    return np.degrees(np.arctan2(b, a)), rho


# =====================================================================
# Main V_inter sweep
# =====================================================================

for ig, gratio in enumerate(G_RATIOS):
    V_inter = float(V0 * gratio)
    print("\n" + "=" * 90)
    print(f"V/V0={gratio:.6g}   V_inter={V_inter:.10g}")
    print("=" * 90)

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

    if not np.all(scan["success"]):
        print("WARNING failed relaxation indices:", np.flatnonzero(~scan["success"]))

    v_family = phason_vector_from_family(scan["relaxed"], scan["offsets"])
    X0 = np.asarray(scan["relaxed"][mid], dtype=float)
    H = np.asarray(scan["hessians"][mid], dtype=float)
    H = 0.5 * (H + H.T)
    v_ph = remove_translation(np.asarray(v_family[mid], dtype=float))

    # ----- direct microscopic block check
    vals = hessian_block_diagnostic(H)
    (HAA_norm[ig], HBB_norm[ig], HAB_norm[ig], HBA_norm[ig],
     HAB_ratio[ig], HAB_symmetry_error[ig]) = vals

    print(
        f"||HAB||F={HAB_norm[ig]:.8e}  "
        f"R_AB={HAB_ratio[ig]:.8e}  "
        f"||HAB-HBA^T||F={HAB_symmetry_error[ig]:.3e}"
    )

    # Full exact eigensystem -- diagonalized once for this V.
    evals, evecs = np.linalg.eigh(H)
    nkeep = min(N_EXACT_KEEP, len(evals))

    L = float(X0.max() - X0.min())
    qs = 2.0 * np.pi * Q_INDICES / L
    q_grid[ig] = qs

    for iq, q in enumerate(qs):
        for sec in sectors:
            r = data[sec]
            Aq, Bq, Pq, Wq = build_probes(X0, v_ph, sec, q)
            QAB = orthonormal_basis(Aq, Bq)
            QPW = orthonormal_basis(Pq, Wq)

            if ig == 0:
                # Decoupled limit: exact physical labels A and B.
                chosen = select_initial_AB(evecs, Aq, Bq, nkeep)
                cont = np.ones(2)
            else:
                cand = hydro_candidates(
                    evecs, Aq, Bq, Pq, Wq,
                    nkeep=nkeep, ncand=N_CANDIDATES
                )
                # Keep previous mode indices in the candidate pool too.
                prev_idx = r["mode_idx"][ig - 1, iq]
                cand = np.unique(np.concatenate([cand, prev_idx[prev_idx >= 0]]))
                prev_vecs = np.column_stack(previous[sec][iq])
                chosen, cont = track_two(prev_vecs, evecs, cand)

            r["mode_idx"][ig, iq] = chosen
            r["lambda"][ig, iq] = evals[chosen]
            r["lambda_over_q2"][ig, iq] = evals[chosen] / (q * q)
            r["continuity"][ig, iq] = cont

            r["A"][ig, iq] = raw_overlap(evecs, Aq)[chosen]
            r["B"][ig, iq] = raw_overlap(evecs, Bq)[chosen]
            r["P"][ig, iq] = raw_overlap(evecs, Pq)[chosen]
            r["W"][ig, iq] = raw_overlap(evecs, Wq)[chosen]
            r["ABspan"][ig, iq] = projector_weights(evecs, QAB)[chosen]
            r["PWspan"][ig, iq] = projector_weights(evecs, QPW)[chosen]

            # Mixing angle in the old orthonormal P/W frame.
            porth, worth, theta = pw_mixing_diagnostics(
                evecs, Pq, Wq, chosen
            )
            r["Porth"][ig, iq] = porth
            r["Worth"][ig, iq] = worth
            r["theta_mix_deg"][ig, iq] = theta

            # -------------------------------------------------------------
            # NEW: exact-mode rotation relative to V_inter=0 and signed A/B
            # hybridization angle.  First remove the arbitrary eigenvector
            # sign by enforcing continuity along V_inter.
            # -------------------------------------------------------------
            current_vecs = []
            for b in range(2):
                e = evecs[:, chosen[b]].copy()

                if ig == 0:
                    # Choose a physically transparent gauge:
                    # branch 0 has positive A amplitude, branch 1 positive B.
                    gauge_probe = Aq if b == 0 else Bq
                    if float(normalize(gauge_probe) @ e) < 0.0:
                        e *= -1.0
                    reference[sec][iq][b] = e.copy()
                else:
                    e = sign_align(e, previous[sec][iq][b])

                eref = reference[sec][iq][b]

                # Principal angle between one-dimensional exact eigenspaces.
                # abs makes it fully invariant to the arbitrary sign.
                ov = np.clip(abs(float(eref @ e)), 0.0, 1.0)
                r["theta_ref_deg"][ig, iq, b] = np.degrees(np.arccos(ov))

                # Signed microscopic A/B coordinate angle.
                ang, rho = signed_AB_angle(e, Aq, Bq)
                r["theta_AB_deg"][ig, iq, b] = ang
                r["AB_proj_norm"][ig, iq, b] = rho

                # Minimal phenomenological damping channel for relative A/B
                # motion. Aq and Bq have disjoint support, hence Ahat and Bhat
                # are orthogonal and Rq is normalized by construction.
                ahat = normalize(Aq)
                bhat = normalize(Bq)
                Rq = (ahat - bhat) / np.sqrt(2.0)
                r["damping_factor_AB"][ig, iq, b] = abs(float(Rq @ e))**2

                current_vecs.append(e)

            previous[sec][iq] = current_vecs

            # Save the exact modes and probes at physical coupling for the
            # final damped time evolution.  Copies avoid any later gauge/mutation.
            if np.isclose(gratio, 1.0):
                physical_mode_cache[sec][iq] = {
                    "H": H.copy(),
                    "q": float(q),
                    "modes": [v.copy() for v in current_vecs],
                    "Aq": Aq.copy(),
                    "Bq": Bq.copy(),
                }

    # concise diagnostic at q_min
    for sec in sectors:
        r = data[sec]
        print(f"  {sec}, q_min={qs[0]:.7e}")
        for b in range(2):
            print(
                f"    b{b}: e{r['mode_idx'][ig,0,b]} "
                f"lam/q2={r['lambda_over_q2'][ig,0,b]:.5e} "
                f"A={r['A'][ig,0,b]:.3f} B={r['B'][ig,0,b]:.3f} "
                f"P={r['P'][ig,0,b]:.3f} W={r['W'][ig,0,b]:.3f} "
                f"cont={r['continuity'][ig,0,b]:.4f}"
            )



# =====================================================================
# Final-note figures only
# =====================================================================

def savefig(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=230)
    plt.close()


def heatmap(Z, title, cbar, filename, vmin=None, vmax=None):
    qmean = np.nanmean(q_grid, axis=0)
    X, Y = np.meshgrid(qmean, G_RATIOS)
    plt.figure(figsize=(8.0, 5.8))
    pcm = plt.pcolormesh(X, Y, Z, shading="nearest", vmin=vmin, vmax=vmax)
    plt.colorbar(pcm, label=cbar)
    plt.xlabel(r"$q$")
    plt.ylabel(r"$V_{\rm inter}/V_0$")
    plt.title(title)
    savefig(filename)


# Fig. 2 in the note: projected microscopic A-B coupling at q_min.
# At q_min, Aq and Bq have disjoint support, so <Aq|H|Bq>/sqrt(...) is
# evaluated directly from the relaxed Hessian for cosine and sine probes.
eta_qmin = {sec: np.full(ng, np.nan) for sec in sectors}
for ig, gratio in enumerate(G_RATIOS):
    V_inter = float(V0 * gratio)
    scan = sweep_offsets(
        base_positions=base_positions, labels=labels, offsets=offsets,
        base_offset=base_offset, R_A=R_A, R_B=R_B, K_intra=K_intra,
        V_inter=V_inter, sigma_inter=sigma_inter, max_displacement=3.0,
        continuation=True, bulk_margin=5.0 * sigma_inter,
    )
    X0 = np.asarray(scan["relaxed"][mid], float)
    H = np.asarray(scan["hessians"][mid], float)
    H = 0.5 * (H + H.T)
    v_family = phason_vector_from_family(scan["relaxed"], scan["offsets"])
    v_ph = remove_translation(np.asarray(v_family[mid], float))
    q = q_grid[ig, 0]
    for sec in sectors:
        Aq, Bq, _, _ = build_probes(X0, v_ph, sec, q)
        kaa = float(Aq @ H @ Aq)
        kbb = float(Bq @ H @ Bq)
        kab = float(Aq @ H @ Bq)
        den = np.sqrt(abs(kaa * kbb))
        eta_qmin[sec][ig] = kab / den if den > 0 else np.nan

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
for a, sec in zip(ax, sectors):
    a.plot(G_RATIOS, eta_qmin[sec], "o-")
    a.axhline(0.0, lw=0.8)
    a.set_xlabel(r"$V_{\rm inter}/V_0$")
    a.set_ylabel(r"$K_{AB}/\sqrt{K_{AA}K_{BB}}$")
    a.set_title(sec)
fig.tight_layout()
fig.savefig(OUT / "final_fig02_projected_AB_qmin.png", dpi=230)
plt.close(fig)


# Fig. 3: direct microscopic Hessian block coupling.
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
ax[0].plot(G_RATIOS, HAB_ratio, "o-")
ax[0].axhline(0.0, lw=0.8)
ax[0].set_xlabel(r"$V_{\rm inter}/V_0$")
ax[0].set_ylabel(r"$\|H_{AB}\|_F/\sqrt{\|H_{AA}\|_F\|H_{BB}\|_F}$")
ax[0].set_title("Direct microscopic Hessian block coupling")
ax[1].semilogy(G_RATIOS, np.maximum(HAB_norm, 1e-18), "o-", label=r"$\|H_{AB}\|_F$")
ax[1].semilogy(G_RATIOS, np.maximum(HAB_symmetry_error, 1e-18), "s--",
               label=r"$\|H_{AB}-H_{BA}^T\|_F$")
ax[1].set_xlabel(r"$V_{\rm inter}/V_0$")
ax[1].set_ylabel("Frobenius norm")
ax[1].set_title("Microscopic A-B Hessian block")
ax[1].legend()
fig.tight_layout()
fig.savefig(OUT / "final_fig03_microscopic_HAB.png", dpi=230)
plt.close(fig)


# Fig. 4: raw A/B/P/W overlaps of the two tracked exact modes at q_min.
fig, ax = plt.subplots(2, 2, figsize=(10.5, 8.0), sharex=True, sharey=True)
for j, sec in enumerate(sectors):
    r = data[sec]
    for b in range(2):
        a = ax[b, j]
        for key, marker in zip(("A", "B", "P", "W"), ("o", "s", "^", "v")):
            a.plot(G_RATIOS, r[key][:, 0, b], marker + "-", label=rf"$|\langle {key}_q|e\rangle|^2$")
        a.set_title(f"{sec}: tracked exact branch {b}")
        a.set_xlabel(r"$V_{\rm inter}/V_0$")
        a.set_ylabel("raw squared overlap")
        a.set_ylim(-0.03, 1.03)
        a.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "final_fig04_qmin_overlaps.png", dpi=230)
plt.close(fig)


# Fig. 5: hydrodynamic span{P,W} weight of exact eigenmodes during the sweep.
# Reconstruct the full low-mode PW weights only at q_min, matching the diagnostic
# used in the note. This is intentionally the only remaining mode-index heat map.
pw_heat = {sec: np.full((ng, N_EXACT_KEEP), np.nan) for sec in sectors}
for ig, gratio in enumerate(G_RATIOS):
    V_inter = float(V0 * gratio)
    scan = sweep_offsets(
        base_positions=base_positions, labels=labels, offsets=offsets,
        base_offset=base_offset, R_A=R_A, R_B=R_B, K_intra=K_intra,
        V_inter=V_inter, sigma_inter=sigma_inter, max_displacement=3.0,
        continuation=True, bulk_margin=5.0 * sigma_inter,
    )
    X0 = np.asarray(scan["relaxed"][mid], float)
    H = np.asarray(scan["hessians"][mid], float)
    H = 0.5 * (H + H.T)
    evals, evecs = np.linalg.eigh(H)
    nkeep = min(N_EXACT_KEEP, len(evals))
    v_family = phason_vector_from_family(scan["relaxed"], scan["offsets"])
    v_ph = remove_translation(np.asarray(v_family[mid], float))
    q = q_grid[ig, 0]
    for sec in sectors:
        _, _, Pq, Wq = build_probes(X0, v_ph, sec, q)
        QPW = orthonormal_basis(Pq, Wq)
        pw_heat[sec][ig, :nkeep] = projector_weights(evecs[:, :nkeep], QPW)

fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
for a, sec in zip(ax, sectors):
    im = a.imshow(pw_heat[sec].T, origin="lower", aspect="auto", vmin=0, vmax=1,
                  extent=[G_RATIOS[0], G_RATIOS[-1], 0, N_EXACT_KEEP-1])
    r = data[sec]
    for b, marker in zip(range(2), ("o", "s")):
        a.plot(G_RATIOS, r["mode_idx"][:, 0, b], marker + "-", ms=3, label=f"tracked branch {b}")
    a.set_xlabel(r"$V_{\rm inter}/V_0$")
    a.set_title(sec)
    a.legend(fontsize=8)
ax[0].set_ylabel("exact Hessian mode index")
fig.colorbar(im, ax=ax.ravel().tolist(), label=r"$\|\Pi_{\mathrm{span}\{P,W\}}e_n\|^2$")
fig.savefig(OUT / "final_fig05_hydrodynamic_weight.png", dpi=230, bbox_inches="tight")
plt.close(fig)


# Fig. 6: representative small-q quadratic behavior at physical coupling g=1.
ig_phys = int(np.argmin(np.abs(G_RATIOS - 1.0)))
q = q_grid[ig_phys]
q2 = q*q
fit_n = min(6, nq)  # strict long-wavelength window
plt.figure(figsize=(7.6, 5.4))
fit_lines = []
for sec, marker in zip(sectors, ("o", "s")):
    r = data[sec]
    for b in range(2):
        y = r["lambda"][ig_phys, :, b]
        plt.plot(q2, y, marker, ms=5, label=f"{sec} branch {b}")
        coef = np.polyfit(q2[:fit_n], y[:fit_n], 1)
        xfit = np.linspace(0, q2[fit_n-1], 100)
        plt.plot(xfit, np.polyval(coef, xfit), "--", lw=1)
        fit_lines.append((sec, b, coef[0], coef[1]))
plt.xlabel(r"$q^2$")
plt.ylabel(r"$\lambda(q)$")
plt.title("Small-q fits at physical coupling")
plt.legend()
savefig("final_fig06_small_q_fits.png")


# Fig. 7: stiffness maps lambda/q^2 for the two branches (cosine sector;
# sine gives nearly identical stiffnesses and is summarized by the fits above).
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
qmean = np.nanmean(q_grid, axis=0)
X, Y = np.meshgrid(qmean, G_RATIOS)
for b in range(2):
    im = ax[b].pcolormesh(X, Y, data["cos"]["lambda_over_q2"][:, :, b], shading="nearest")
    fig.colorbar(im, ax=ax[b], label=r"$\lambda(q,V)/q^2$")
    ax[b].set_xlabel(r"$q$")
    ax[b].set_ylabel(r"$V_{\rm inter}/V_0$")
    ax[b].set_title(f"cos: branch {b}")
fig.tight_layout()
fig.savefig(OUT / "final_fig07_stiffness_maps.png", dpi=230)
plt.close(fig)


# Fig. 8: the four final q-cuts added to the note.
V_CUT_TARGETS = (0.1, 0.5, 1.0)
vcut_ids = [int(np.argmin(np.abs(G_RATIOS-v))) for v in V_CUT_TARGETS]
qmean = np.nanmean(q_grid, axis=0)
for sec in sectors:
    r = data[sec]
    for b in range(2):
        plt.figure(figsize=(7.4, 5.2))
        for ig in vcut_ids:
            plt.plot(qmean, r["theta_ref_deg"][ig, :, b], "o-",
                     label=fr"$V_{{\rm inter}}/V_0={G_RATIOS[ig]:.2g}$")
        plt.xlabel(r"$q$")
        plt.ylabel(r"$\Theta_{\rm ref}$ [deg]")
        plt.title(f"{sec}: branch {b}, exact-mode rotation vs q")
        plt.grid(alpha=0.25)
        plt.legend()
        savefig(f"15_theta_ref_vs_q_{sec}_branch{b}.png")


# Fig. 9: explicit damped time evolution at physical coupling g=1.
#
# Equation of motion:
#     u_ddot + gamma_AB D_AB u_dot + H u = 0,
# where D_AB = |R_q><R_q| and R_q=(Ahat_q-Bhat_q)/sqrt(2).
# Thus only relative A/B velocity is damped; common translation is untouched.
#
# For a deliberately minimal final test we use the longest-wavelength q=q_min
# in the cosine sector and launch each tracked exact branch separately with
# u(0)=e_b, u_dot(0)=0.  We project u(t) back onto e_b, extract peak amplitudes,
# and fit log(envelope)=const-Gamma*t.
DAMP_SECTOR = "cos"
DAMP_IQ = 0

# Weak relative A/B friction.  The final test is performed ONLY in the
# two-mode low-energy subspace, so no large 2N x 2N diagonalization is needed.
GAMMA_AB = 0.005
GAMMA_SCAN = np.array([0.0, 0.001, 0.002, 0.005, 0.01, 0.02])

from scipy.linalg import eig


def relative_AB_vector(Aq, Bq):
    """
    Unit relative A/B displacement direction

        |R_q> = (|A_q> - |B_q>)/sqrt(2)

    after separately normalizing the A and B sector vectors.
    """
    a = normalize(Aq)
    b = normalize(Bq)
    return normalize(a - b)


def reduced_damped_modes(H, modes, Aq, Bq, gamma_ab):
    """
    Exact damping inside the 2-mode subspace spanned by `modes`.

    We project
        u'' + D u' + H u = 0
    onto E=(e0,e1), with
        D = gamma_AB |R_q><R_q|.

    This gives only a 4x4 first-order generator, so it is essentially
    instantaneous even when the full microscopic Hessian is large.

    Returns one result dictionary for each undamped branch.
    """
    E = np.column_stack([normalize(modes[0]), normalize(modes[1])])

    # Re-orthogonalize very gently in case accumulated numerical tracking
    # has produced tiny loss of orthogonality.
    E, _ = np.linalg.qr(E)

    Rq = relative_AB_vector(Aq, Bq)

    H_eff = E.T @ H @ E
    H_eff = 0.5 * (H_eff + H_eff.T)

    r = E.T @ Rq
    D_eff = gamma_ab * np.outer(r, r)

    # Undamped reference modes of the projected H_eff.
    lam_ref, Uref = np.linalg.eigh(H_eff)
    order = np.argsort(lam_ref)
    lam_ref = lam_ref[order]
    Uref = Uref[:, order]

    # Match projected reference eigenvectors to the original tracked
    # branches, rather than assuming eigenvalue order is always identical.
    tracked_in_eff = E.T @ np.column_stack(
        [normalize(modes[0]), normalize(modes[1])]
    )
    overlap_map = np.abs(Uref.T @ tracked_in_eff)

    # For 2 modes there are only two possible assignments.
    score_identity = overlap_map[0, 0] + overlap_map[1, 1]
    score_swap = overlap_map[0, 1] + overlap_map[1, 0]
    if score_identity >= score_swap:
        ref_for_branch = [0, 1]
    else:
        ref_for_branch = [1, 0]

    Z = np.zeros((2, 2))
    I2 = np.eye(2)
    L_eff = np.block([
        [Z,       I2],
        [-H_eff, -D_eff],
    ])

    vals, vecs = eig(L_eff)

    positive = np.where(vals.imag > 0.0)[0]
    if len(positive) < 2:
        positive = np.argsort(vals.imag)[-2:]

    results = []

    for b in range(2):
        ir = ref_for_branch[b]
        lam0 = max(float(lam_ref[ir]), 1e-14)
        omega0 = np.sqrt(lam0)
        v0 = Uref[:, ir]

        z0 = np.concatenate([
            v0.astype(complex),
            1j * omega0 * v0.astype(complex),
        ])
        z0 /= np.linalg.norm(z0)

        ovs = []
        for j in positive:
            z = vecs[:, j].copy()
            z /= np.linalg.norm(z)
            ovs.append(abs(np.vdot(z0, z)))
        ovs = np.asarray(ovs)

        jbest = positive[np.argmax(ovs)]
        sval = vals[jbest]

        # Relative A/B content of the tracked branch itself.
        eb = normalize(modes[b])
        f_ab = abs(float(Rq @ eb))**2

        results.append({
            "branch": b,
            "f_AB": f_ab,
            "Gamma_pert": 0.5 * gamma_ab * f_ab,
            "Gamma_exact": max(0.0, -float(sval.real)),
            "omega0": omega0,
            "omega_exact": abs(float(sval.imag)),
            "overlap": float(ovs.max()),
            "eig": sval,
        })

    return results


cache = physical_mode_cache[DAMP_SECTOR][DAMP_IQ]
H_damp = cache["H"]
q_damp = cache["q"]
Aq_damp = cache["Aq"]
Bq_damp = cache["Bq"]
modes_damp = cache["modes"]

print("\n" + "=" * 90)
print("WEAK RELATIVE A/B FRICTION: FAST TWO-MODE COMPLEX-EIGENVALUE TEST")
print("=" * 90)
print(
    f"sector={DAMP_SECTOR}, q={q_damp:.8g}, "
    f"gamma_AB={GAMMA_AB:g}"
)

damp_results = reduced_damped_modes(
    H_damp, modes_damp, Aq_damp, Bq_damp, GAMMA_AB
)

for rr in damp_results:
    ratio = (
        rr["Gamma_exact"] / rr["Gamma_pert"]
        if rr["Gamma_pert"] > 0 else np.nan
    )
    print(
        f"  branch {rr['branch']}: "
        f"f_AB={rr['f_AB']:.6f}  "
        f"Gamma_pert={rr['Gamma_pert']:.8e}  "
        f"Gamma_exact={rr['Gamma_exact']:.8e}  "
        f"exact/pert={ratio:.5f}  "
        f"omega0={rr['omega0']:.6e}  "
        f"omega={rr['omega_exact']:.6e}  "
        f"track={rr['overlap']:.6f}"
    )

if damp_results[0]["Gamma_exact"] > 0:
    print(
        "\n  exact damping ratio Gamma_1/Gamma_0 = "
        f"{damp_results[1]['Gamma_exact']/damp_results[0]['Gamma_exact']:.6f}"
    )
if damp_results[0]["f_AB"] > 0:
    print(
        "  relative-weight ratio f_AB,1/f_AB,0 = "
        f"{damp_results[1]['f_AB']/damp_results[0]['f_AB']:.6f}"
    )


# -------------------------------------------------------------------------
# gamma_AB scan: demonstrate the weak-damping linear regime
# -------------------------------------------------------------------------
scan_exact = np.zeros((2, len(GAMMA_SCAN)))
scan_pert = np.zeros_like(scan_exact)

for igam, gam in enumerate(GAMMA_SCAN):
    rr_list = reduced_damped_modes(
        H_damp, modes_damp, Aq_damp, Bq_damp, gam
    )
    for rr in rr_list:
        b = rr["branch"]
        scan_exact[b, igam] = rr["Gamma_exact"]
        scan_pert[b, igam] = rr["Gamma_pert"]


fig, ax = plt.subplots(figsize=(7.4, 5.0))
for b in range(2):
    ax.plot(
        GAMMA_SCAN, scan_exact[b], "o-", lw=1.8,
        label=fr"branch {b}: exact $-\mathrm{{Re}}\,s$"
    )
    ax.plot(
        GAMMA_SCAN, scan_pert[b], "--", lw=1.6,
        label=fr"branch {b}: $(\gamma_{{AB}}/2)f_{{AB}}$"
    )

ax.set_xlabel(r"$\gamma_{AB}$")
ax.set_ylabel(r"modal damping rate $\Gamma$")
ax.set_title(
    fr"Relative A/B friction: {DAMP_SECTOR}, "
    fr"$q={q_damp:.4g}$, $V_{{\rm inter}}/V_0=1$"
)
ax.grid(alpha=0.25)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(
    OUT / "final_fig09_damped_complex_eigenvalues.png",
    dpi=230, bbox_inches="tight"
)
plt.close(fig)


# -------------------------------------------------------------------------
# Direct check of the central physical statement:
#
#       2 Gamma / gamma_AB  ~=  f_AB
#
# Thus the mode with larger relative A/B character is more strongly damped.
# -------------------------------------------------------------------------
fab = np.array([rr["f_AB"] for rr in damp_results])
scaled_gamma = np.array([
    2.0 * rr["Gamma_exact"] / GAMMA_AB
    for rr in damp_results
])

x = np.arange(2)
w = 0.36

fig, ax = plt.subplots(figsize=(6.3, 4.7))
ax.bar(
    x - w/2, fab, width=w,
    label=r"$f_{AB}=|\langle R_q|e_b\rangle|^2$"
)
ax.bar(
    x + w/2, scaled_gamma, width=w,
    label=r"$2\Gamma_{\rm exact}/\gamma_{AB}$"
)

ax.set_xticks(x, ["branch 0", "branch 1"])
ax.set_ylabel("relative A/B damping weight")
ax.set_title(
    fr"Weak-friction check: {DAMP_SECTOR}, "
    fr"$q={q_damp:.4g}$, $\gamma_{{AB}}={GAMMA_AB:g}$"
)
ax.grid(axis="y", alpha=0.25)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(
    OUT / "final_fig10_damping_vs_AB_weight.png",
    dpi=230, bbox_inches="tight"
)
plt.close(fig)


# -------------------------------------------------------------------------
# Compact output used by the note
# -------------------------------------------------------------------------
np.savez(
    OUT / "final_note_data.npz",
    g_ratios=G_RATIOS,
    q_grid=q_grid,
    HAB_ratio=HAB_ratio,
    HAB_norm=HAB_norm,
    HAB_symmetry_error=HAB_symmetry_error,
    gamma_scan=GAMMA_SCAN,
    damping_exact=scan_exact,
    damping_pert=scan_pert,
    damping_fAB=fab,
    **{
        f"{key}_{sec}": val
        for sec in sectors
        for key, val in data[sec].items()
    }
)

with open(OUT / "final_note_summary.txt", "w") as f:
    f.write("# Final-note diagnostics only\n")
    f.write("# small-q fits: lambda = intercept + C q^2 at g=1\n")

    for sec, b, C, intercept in fit_lines:
        f.write(
            f"{sec} branch {b}: "
            f"C={C:.10e}, intercept={intercept:.10e}\n"
        )

    f.write(
        "\n# weak relative A/B friction: "
        "fast two-mode complex-eigenvalue test\n"
    )
    f.write(
        f"# sector={DAMP_SECTOR}, gamma_AB={GAMMA_AB:.10e}, "
        f"q={q_damp:.10e}\n"
    )

    for rr in damp_results:
        f.write(
            f"branch {rr['branch']}: "
            f"f_AB={rr['f_AB']:.10e}, "
            f"Gamma_exact={rr['Gamma_exact']:.10e}, "
            f"Gamma_pert={rr['Gamma_pert']:.10e}, "
            f"omega0={rr['omega0']:.10e}, "
            f"omega_exact={rr['omega_exact']:.10e}, "
            f"track_overlap={rr['overlap']:.10e}\n"
        )

    f.write("\n# q_min endpoints\n")
    for sec in sectors:
        r = data[sec]
        for ig in (0, ng - 1):
            for b in range(2):
                f.write(
                    f"{sec} g={G_RATIOS[ig]:.6g} b={b} "
                    f"mode={r['mode_idx'][ig,0,b]} "
                    f"A={r['A'][ig,0,b]:.6f} "
                    f"B={r['B'][ig,0,b]:.6f} "
                    f"P={r['P'][ig,0,b]:.6f} "
                    f"W={r['W'][ig,0,b]:.6f} "
                    f"thetaRef={r['theta_ref_deg'][ig,0,b]:.4f}deg\n"
                )

print("\nDONE -- final-note outputs only")
print("Results:", OUT)
print("  final_fig02_projected_AB_qmin.png")
print("  final_fig03_microscopic_HAB.png")
print("  final_fig04_qmin_overlaps.png")
print("  final_fig05_hydrodynamic_weight.png")
print("  final_fig06_small_q_fits.png")
print("  final_fig07_stiffness_maps.png")
print("  15_theta_ref_vs_q_{cos,sin}_branch{0,1}.png")
print("  final_fig09_damped_complex_eigenvalues.png")
print("  final_fig10_damping_vs_AB_weight.png")
print("  final_note_data.npz")
print("  final_note_summary.txt")
