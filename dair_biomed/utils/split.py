import logging
logger = logging.getLogger(__name__)

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

def random_split(n, r_val, r_test):
    r_train = 1 - r_val - r_test
    perm = np.random.permutation(n)
    train_cutoff = r_train * n
    val_cutoff = (r_train + r_val) * n
    return perm[:train_cutoff], perm[train_cutoff : val_cutoff], perm[val_cutoff:]

def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold

def generate_scaffolds(dataset, log_every_n=1000):
    scaffolds = {}
    data_len = len(dataset)

    logger.info("About to generate scaffolds")
    for ind, smiles in enumerate(dataset.smiles):
        if ind % log_every_n == 0:
            logger.info("Generating scaffold %d/%d" % (ind, data_len))
        scaffold = _generate_scaffold(smiles)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), 
            key=lambda x: (len(x[1]), x[1][0]), 
            reverse=True
        )
    ]
    return scaffold_sets

def scaffold_split(dataset, r_val, r_test, log_every_n=1000):
    r_train = 1.0 - r_val - r_test
    scaffold_sets = generate_scaffolds(dataset, log_every_n)

    train_cutoff = r_train * len(dataset)
    valid_cutoff = (r_train + r_val) * len(dataset)
    train_inds = []
    valid_inds = []
    test_inds = []

    logger.info("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    return train_inds, valid_inds, test_inds

