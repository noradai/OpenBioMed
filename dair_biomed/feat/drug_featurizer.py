import numpy as np
import torch

import rdkit.Chem as Chem
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import Data

from feat.base_featurizer import BaseFeaturizer
from feat.kg_featurizer import SUPPORTED_KG_FEATURIZER
from feat.text_featurizer import SUPPORTED_TEXT_FEATURIZER

def one_hot_encoding(x, allowable_set, encode_unknown=False):
    """One-hot encoding.
    """
    if encode_unknown and (allowable_set[-1] is not None):
        allowable_set.append(None)

    if encode_unknown and (x not in allowable_set):
        x = None

    return list(map(lambda s: x == s, allowable_set))

# Atom featurization: Borrowed from dgllife.utils.featurizers.py

def atom_type_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the type of an atom.
    """
    if allowable_set is None:
        allowable_set = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                         'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn',
                         'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au',
                         'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb']
    return one_hot_encoding(atom.GetSymbol(), allowable_set, encode_unknown)

def atom_degree_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the degree of an atom.
    """
    if allowable_set is None:
        allowable_set = list(range(11))
    return one_hot_encoding(atom.GetDegree(), allowable_set, encode_unknown)

def atom_implicit_valence_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the implicit valence of an atom.
    """
    if allowable_set is None:
        allowable_set = list(range(7))
    return one_hot_encoding(atom.GetImplicitValence(), allowable_set, encode_unknown)

def atom_formal_charge(atom):
    """Get formal charge for an atom.
    """
    return [atom.GetFormalCharge()]

def atom_num_radical_electrons(atom):
    return [atom.GetNumRadicalElectrons()]

def atom_hybridization_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the hybridization of an atom.
    """
    if allowable_set is None:
        allowable_set = [Chem.rdchem.HybridizationType.SP,
                         Chem.rdchem.HybridizationType.SP2,
                         Chem.rdchem.HybridizationType.SP3,
                         Chem.rdchem.HybridizationType.SP3D,
                         Chem.rdchem.HybridizationType.SP3D2]
    return one_hot_encoding(atom.GetHybridization(), allowable_set, encode_unknown)

def atom_is_aromatic(atom):
    """Get whether the atom is aromatic.
    """
    return [atom.GetIsAromatic()]

def atom_total_num_H_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the total number of Hs of an atom.
    """
    if allowable_set is None:
        allowable_set = list(range(5))
    return one_hot_encoding(atom.GetTotalNumHs(), allowable_set, encode_unknown)

def atom_is_in_ring(atom):
    """Get whether the atom is in ring.
    """
    return [atom.IsInRing()]

def atom_chirality_type_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the chirality type of an atom.
    """
    if not atom.HasProp('_CIPCode'):
        return [False, False]

    if allowable_set is None:
        allowable_set = ['R', 'S']
    return one_hot_encoding(atom.GetProp('_CIPCode'), allowable_set, encode_unknown)

# Atom featurization: Borrowed from dgllife.utils.featurizers.py

def bond_type_one_hot(bond, allowable_set=None, encode_unknown=False):
    """One hot encoding for the type of a bond.
    """
    if allowable_set is None:
        allowable_set = [Chem.rdchem.BondType.SINGLE,
                         Chem.rdchem.BondType.DOUBLE,
                         Chem.rdchem.BondType.TRIPLE,
                         Chem.rdchem.BondType.AROMATIC]
    return one_hot_encoding(bond.GetBondType(), allowable_set, encode_unknown)

class DrugOneHotFeaturizer(BaseFeaturizer):
    smiles_char = ['?', '#', '%', ')', '(', '+', '-', '.', '1', '0', '3', '2', '5', '4',
       '7', '6', '9', '8', '=', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I',
       'H', 'K', 'M', 'L', 'O', 'N', 'P', 'S', 'R', 'U', 'T', 'W', 'V',
       'Y', '[', 'Z', ']', '_', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i',
       'h', 'm', 'l', 'o', 'n', 's', 'r', 'u', 't', 'y']

    def __init__(self, max_len=256):
        super(DrugOneHotFeaturizer, self).__init__()
        self.max_len = max_len
        self.enc = OneHotEncoder().fit(np.array(self.smiles_char).reshape(-1, 1))

    def __call__(self, data):
        temp = [c if c in self.smiles_char else '?' for c in data]
        if len(temp) < self.max_len:
            temp = temp + ['?'] * (self.max_len - len(temp))
        else:
            temp = temp [:self.max_len]
        return self.enc.transform(np.array(temp).reshape(-1, 1)).toarray().T

"""
class MolCLRFeaturizer(BaseFeaturizer):
    ATOM_LIST = list(range(1,119))
    CHIRALITY_LIST = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ]
    BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
    BONDDIR_LIST = [
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
    
    def __init__(self):
        super(MolCLRFeaturizer, self).__init__()

    def featurize(self, data):
        mol = Chem.MolFromSmiles(data)
        mol = Chem.AddHs(mol)

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
        return x, edge_index, edge_attr
"""

class DrugTGSAFeaturizer(BaseFeaturizer):
    def __init__(self, name):
        super(DrugTGSAFeaturizer, self).__init__()

    def atom_feature(self, atom):
        """
        Converts rdkit atom object to feature list of indices
        :param mol: rdkit atom object
        :return: list
        8 features are canonical, 2 features are from OGB
        """
        featurizer_funcs = [
            atom_type_one_hot,
            atom_degree_one_hot,
            atom_implicit_valence_one_hot,
            atom_formal_charge,
            atom_num_radical_electrons,
            atom_hybridization_one_hot,
            atom_is_aromatic,
            atom_total_num_H_one_hot,
            atom_is_in_ring,
            atom_chirality_type_one_hot,
        ]
        atom_feature = np.concatenate([func(atom) for func in featurizer_funcs], axis=0)
        return atom_feature

    def bond_feature(self, bond):
        """
        Converts rdkit bond object to feature list of indices
        :param mol: rdkit bond object
        :return: list
        """
        featurizer_funcs = [bond_type_one_hot]
        bond_feature = np.concatenate([func(bond) for func in featurizer_funcs], axis=0)

        return bond_feature
    
    def __call__(self, data):
        mol = Chem.MolFromSmiles(data)
        """
        Converts SMILES string to graph Data object without remove salt
        :input: SMILES string (str)
        :return: pyg Data object
        """
        # atoms
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_features_list.append(self.atom_feature(atom))
        x = np.array(atom_features_list, dtype=np.int64)

        # bonds
        num_bond_features = 3  # bond type, bond stereo, is_conjugated
        if len(mol.GetBonds()) > 0:  # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_feature = self.bond_feature(bond)
                # add edges in both directions
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            edge_index = np.array(edges_list, dtype=np.int64).T
            edge_attr = np.array(edge_features_list, dtype=np.int64)

        else:  # mol has no bonds
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

        graph = Data(
            x=torch.tensor(x, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        )
        return graph
    
class DrugMultiModalFeaturizer(BaseFeaturizer):
    def __init__(self, config):
        super(DrugMultiModalFeaturizer, self).__init__()
        self.modality = config["modality"]
        self.featurizers = {}
        if "structure" in config["modality"]:
            self.featurizers["structure"] = SUPPORTED_DRUG_FEATURIZER[config["featurizer"]["structure"]]()
        if "KG" in config["modality"]:
            self.featurizers["KG"] = SUPPORTED_KG_FEATURIZER[config["featurizer"]["KG"]]
        if "text" in config["modality"]:
            self.featurizers["text"] = SUPPORTED_TEXT_FEATURIZER[config["featurizer"]["text"]]

    def __call__(self, data):
        feat = self.featurizers["structure"](data)
        for modality in self.modality:
            if modality != "structure":
                cur_feat = self.featurizers[modality](data)
                for key in cur_feat:
                    feat.__setitem__(key, cur_feat[key])
        return feat

SUPPORTED_DRUG_FEATURIZER = {
    "OneHot": DrugOneHotFeaturizer,
    #"molclr": DrugMolCLRFeaturizer,
    "TGSA": DrugTGSAFeaturizer,
    "MultiModal": DrugMultiModalFeaturizer,
}