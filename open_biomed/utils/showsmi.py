from rdkit import DataStructs, Chem
from rdkit.Chem import AllChem, MACCSkeys

with open("../assets/text2smi/molfm-smi.txt", "r") as f:
    for i, line in enumerate(f.readlines()):
        if i == 0:
            continue
        line = line.rstrip("\n").split("\t")
        if i % 1 == 0:
            gt = line[1]
            pred = line[2]
            try:
                gt_mol = Chem.MolFromSmiles(gt)
                pred_mol = Chem.MolFromSmiles(pred)
                print(gt, pred, DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(gt_mol, 2), AllChem.GetMorganFingerprint(pred_mol, 2)))
            except:
                print(gt, pred)