import numpy as np
from scipy.linalg import eigh
from scipy.fft import fft 
from scipy.signal import find_peaks
from skimage import filters
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import utils


results_path = './results/'

lattice = utils.RandomLattice1D(supercell_size=200)

q = 1.4
R_A = 20.0
R_B = R_A * q

lattice.generate_sublattices_with_bonds(
    periods=[R_A, R_B],
    offsets=[0.0, 1.0],
    inter_cutoff=R_B,
)

# Visualize geometry and connectivity
fig, ax = lattice.plot_sublattices(
    filename=results_path+'sublattices.png',
    vertical_spacing=1.2,
    node_size=110,
    show_indices=True,
)
plt.close(fig)

# Build the dynamical matrix
D = lattice.dynamical_matrix(
    alpha_c=1.0,
    intra_coupling=[1.0, 0.8],
    inter_coupling=0.25,
)

lattice = utils.RandomLattice1D(supercell_size=16000)

eks = []
ommax = 0.15
no_q = 201
qmin = 1.333
qmax = 1.4142
qs = np.linspace(qmin, qmax, num = int(no_q)+1, endpoint = True)

for ic, inter_c in enumerate([0.0, 0.01, 0.1, 0.2]):
    c_str = ['0.00', '0.01', '0.10', '0.20']
    for qi, q in enumerate([4./3., 1.4, 1.4142]):
        q_str = ['1.3333', '1.4000', '1.4142']
        R_A = 20.
        R_B = R_A * q
        lattice.generate_sublattices_with_bonds(
            periods=[R_A, R_B],
            offsets=[0.0, 1.0],
            inter_cutoff=R_B)
        D = lattice.dynamical_matrix(alpha_c=1.0,
            intra_coupling=[1.0, 1.0],
            inter_coupling=inter_c)
        omega2, uns = eigh(D, eigvals_only=False)
        #
        L = lattice.positions.max() - lattice.positions.min()
        k_values = np.linspace(0.0, np.pi / min(R_A, R_B), 300)
        I_phonon, I_phason, participation_balance = lattice.mode_character_vs_k(uns, k_values)
        common_F, relative_F = lattice.sublattice_fourier_modes(uns, k_values)
        
        fig, ax = plt.subplots()
        for ik, k in enumerate(k_values):
            mask = I_phason[ik] > 1e-4
            ax.scatter(
                np.full(mask.sum(), k),
                omega2[mask],
                c=I_phason[ik, mask]*participation_balance[mask],
                s=20 * I_phason[ik, mask],
                norm=colors.LogNorm(vmin=1.e-4,
                vmax=1.e-1)
            )
        ax.set_xlabel(r'$k$')
        ax.set_ylabel(r'$\omega^2$')
        ax.set_ylim(0.0, ommax)
        fig.savefig(results_path+'phasons'+q_str[qi]+'_'+c_str[ic]+'.png', dpi=200)
        plt.close()
        
        fig, ax = plt.subplots()
        for ik, k in enumerate(k_values):
            mask = common_F[ik] > 1e-2
            ax.scatter(
                np.full(mask.sum(), k),
                omega2[mask],
                c=common_F[ik, mask],
                s=0.001*common_F[ik, mask],
                vmin=0.,
                vmax=1.
            )
        ax.set_xlabel(r'$k$')
        ax.set_ylabel(r'$\omega^2$')
        ax.set_ylim(0.0, ommax)
        fig.savefig(results_path+'common_F'+q_str[qi]+'_'+c_str[ic]+'.png', dpi=200)
        plt.close()
        
inter_c = 0.2
ommax = 0.12
ommin = 0.08
for qi, q in enumerate([1.4, 1.4142]):
    q_str = ['1.4000', '1.4142']
    R_A = 20.
    R_B = R_A * q
    lattice.generate_sublattices_with_bonds(
        periods=[R_A, R_B],
        offsets=[0.0, 1.0],
        inter_cutoff=R_B)
    D = lattice.dynamical_matrix(alpha_c=1.0,
        intra_coupling=[1.0, 1.0],
        inter_coupling=inter_c)
    omega2, uns = eigh(D, eigvals_only=False)
    #
    L = lattice.positions.max() - lattice.positions.min()
    k_values = np.linspace(0.0, np.pi / min(R_A, R_B) / 4, 300)
    I_phonon, I_phason, participation_balance = lattice.mode_character_vs_k(uns, k_values)
    common_F, relative_F = lattice.sublattice_fourier_modes(uns, k_values)
    
    fig, ax = plt.subplots()
    for ik, k in enumerate(k_values):
        mask = I_phason[ik] > 1e-4
        ax.scatter(
            np.full(mask.sum(), k),
            omega2[mask],
            c=I_phason[ik, mask]*participation_balance[mask],
            s=20 * I_phason[ik, mask],
            norm=colors.LogNorm(vmin=1.e-4,
            vmax=1.e-1)
        )
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'$\omega^2$')
    ax.set_ylim(ommin, ommax)
    fig.savefig(results_path+'phasons'+q_str[qi]+'_'+c_str[ic]+'_s.png', dpi=200)
    plt.close()
