import numpy as np
import matplotlib.pyplot as plt
import utils

prefix = './0.0/'  # or wherever they are stored
evals = np.load(prefix+'evals.npy')
evecs = np.load(prefix+'evecs.npy')
sublattices_cut = np.load(prefix+'sublattices.npy')
global_indices = np.load(prefix+'indices.npy')

utils.plot_density(evecs[:,3976], sublattices_cut, global_indices, filename='dodecagonal_graphene_eigenstate_density.png')

es = np.linspace(evals.min(), evals.max(), 500)
dos = np.zeros(500)
for eval in evals:
    dos += np.exp(-(es-eval)**2/(0.02**2))/1000.
    
R2=np.amax(sublattices_cut[0][:,0]**2+sublattices_cut[0][:,1]**2)  
rs = []
gs = []
for sj, s in enumerate(sublattices_cut):
    for si, xs in enumerate(s):
        gs.append(global_indices[sj][si])
        rs.append(xs[0]**2+xs[1]**2)
rs = np.array(rs)
gs = np.array(gs)
rse = []
lre = 1.
ilre = -1
for ie, evec in enumerate(evecs.T):
    re=(rs*np.abs(evec[gs])**2).sum()/R2
    rse.append(re)
    if re < lre: 
        lre = re
        ilre = ie
print(ilre, lre)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(es, dos, label='DOS', color='blue')
ax.scatter(evals, rse, color='red', s=.1, label='<r2>/R2')
ax.set_xlabel("Energy [eV]")
ax.set_ylabel("DOS [arb. units]")
ax.set_title("Density of States")
ax.legend()
plt.tight_layout()
plt.savefig('dodecagonal_graphene_doss.png', dpi=300)