import utils

N = 80
crystal = utils.CrystalUtils(a=0.246, d=0.335)
sublattices = crystal.make_dodecagonal_graphene(n=N)
Rc = (N-20)*crystal.a
sublattices_cut = crystal.cut_sublattices_to_circle(
    sublattices,
    radius=Rc,
    center=(0.0, 0.0)
)

utils.plot_top_view(sublattices_cut, s=1., filename='dodecagonal_graphene_top_view.png')

connections = crystal.find_connections(sublattices_cut, cutoff=crystal.a * 5.)
#utils.plot_connections(connections, sublattices_cut, l=.1, filename='dodecagonal_graphene_connections.png')

print(connections[:10])
print("number of connections:", len(connections))

hamiltonian = utils.HamUtils(crystal, Vpp_pi0=-2.7, Vpp_sigma0=0.48, r0=0.0453)
#H, global_indices = hamiltonian.build_sparse_hamiltonian(sublattices_cut, connections, Rb=0.75*Rc)
H, global_indices = hamiltonian.build_sparse_hamiltonian(sublattices_cut, connections)
print(H.shape)
print("nonzero elements:", H.nnz)
crystal.save(sublattices_cut, global_indices)

evals, evecs = hamiltonian.diagonalize_hamiltonian(H, sigma=0., k=10000, return_eigenvectors=True)
#evals, evecs = hamiltonian.diagonalize_hamiltonian(H, sigma=-2., k=10000, return_eigenvectors=True)
hamiltonian.save(evals, evecs)
utils.plot_histogram_dos(evals, bins=200, filename='dodecagonal_graphene_dos.png')

