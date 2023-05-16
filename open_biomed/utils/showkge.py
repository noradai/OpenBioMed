import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import matplotlib as mpl
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Descriptors import ExactMolWt
from sklearn.manifold import TSNE

transe_params = torch.load("../../ckpts/kge/epoch699.pth", map_location="cpu")
kge_all = transe_params['ent_emb.weight'].numpy()
print(transe_params['rel_emb.weight'].norm(p=2, dim=1))

id2smiles = {}
with open("../../datasets/mtr/momu_pretrain/pair.txt", "r") as f:
    for line in f.readlines():
        line = line.rstrip("\n").split("\t")
        idx, smi = line[0], line[1]
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            id2smiles[idx] = smi
print(len(id2smiles))

kg = pickle.load(open("/share/project/molpretrain/data/kg/kg.pkl", "rb"))
ents = kg["ent_dict"]
print(len(ents))
print(kg["rel_dict"])
mol_weights = np.zeros(len(ents))
cnt = 0
selected, unselected = [], []
for k in ents:
    if k in id2smiles:
        mol = Chem.MolFromSmiles(id2smiles[k])
        mol_weights[ents[k]] = ExactMolWt(mol)
        #mol_weights[ents[k]] = 0.8
        selected.append(ents[k])
    elif not k.startswith("DRUGBANK"):
        unselected.append(ents[k])

#print(selected, unselected)
selected = selected[:5000]
unselected = np.array(unselected)
perm = np.random.permutation(len(unselected))
idxs = selected + list(unselected[perm[:len(selected) // 4]])
#idxs = selected
#print(idxs)
kge = kge_all[idxs]
tsne = TSNE(n_components=2, random_state=42)
kge = tsne.fit_transform(kge)

cnt = 0
N = 0
sim_dist = 0
unsim_dist = 0
with open("/share/project/molpretrain/data/kg/sim_kg.txt") as f:
    for line in f.readlines():
        line = line.rstrip("\n").split("\t")
        h = ents[line[0]]
        t = ents[line[2]]
        cur = 0
        curn = 0
        for i in range(N % 20, len(ents) - N % 20, len(ents) // 20):
            cur += np.linalg.norm(kge_all[h] - kge_all[i])
            curn += 1
        sim_dist += np.linalg.norm(kge_all[h] - kge_all[t])
        unsim_dist += cur / curn
        N += 1
        if h in selected and t in selected and cnt <= 10 and np.linalg.norm(kge[selected.index(h)] - kge[selected.index(t)]) < 0.1 and np.linalg.norm(kge[selected.index(h)] - kge[selected.index(t)]) > 0:
            print(id2smiles[line[0]], id2smiles[line[2]], kge[selected.index(h)], kge[selected.index(t)], np.linalg.norm(kge[selected.index(h)] - kge[selected.index(t)]))
print(sim_dist / N, unsim_dist / N)

indexes = [[], [], [], [], [], []]
for i in idxs:
    if mol_weights[i] == 0:
        indexes[0].append(i)
    elif mol_weights[i] <= 120:
        indexes[1].append(i)
        mol_weights[i] = 100
    elif mol_weights[i] <= 180:
        indexes[2].append(i)
        mol_weights[i] = 200
    elif mol_weights[i] <= 260:
        indexes[3].append(i)
        mol_weights[i] = 300
    elif mol_weights[i] <= 360:
        indexes[4].append(i)
        mol_weights[i] = 400
    else:
        indexes[5].append(i)
        mol_weights[i] = 500
print([len(x) for x in indexes])

print(mol_weights[idxs])
sc = plt.scatter(kge[:len(selected), 0], kge[:len(selected), 1], s=1, c=mol_weights[selected], cmap=mpl.colormaps['RdYlBu'])
plt.scatter(kge[len(selected):, 0], kge[len(selected):, 1], s=1, c=[(0, 0.5, 0)] * (len(selected) // 4))
plt.colorbar(sc)
plt.axis('off')
plt.show()