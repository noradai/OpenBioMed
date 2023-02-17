import os
import csv
import math
import time
import random
import json
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

from torch_scatter import scatter
from torch_geometric.data import Data, Dataset, DataLoader

from transformers import BertModel, BertTokenizer

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import RDLogger                                                                                                                                                               
RDLogger.DisableLog('rdApp.*')  


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


def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def generate_scaffolds(dataset, log_every_n=1000):
    scaffolds = {}
    data_len = len(dataset)
    print(data_len)

    print("About to generate scaffolds")
    for ind, smiles in enumerate(dataset.smiles_data):
        if ind % log_every_n == 0:
            print("Generating scaffold %d/%d" % (ind, data_len))
        scaffold = _generate_scaffold(smiles)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    return scaffold_sets


def scaffold_split(dataset, valid_size, test_size, seed=None, log_every_n=1000):
    train_size = 1.0 - valid_size - test_size
    scaffold_sets = generate_scaffolds(dataset)

    train_cutoff = train_size * len(dataset)
    valid_cutoff = (train_size + valid_size) * len(dataset)
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []

    print("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    return train_inds, valid_inds, test_inds


def read_smiles(data_path, target, task):
    smiles_data, labels = [], []
    with open(data_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i != -1:
                smiles = row['smiles']
                label = row[target]
                mol = Chem.MolFromSmiles(smiles)
                if mol != None and label != '':
                    smiles_data.append(smiles)
                    if task == 'classification':
                        labels.append(int(label))
                    elif task == 'regression':
                        labels.append(float(label))
                    else:
                        ValueError('task must be either regression or classification')
                if i % 100 == 0:
                    print(i, smiles)
    print(len(smiles_data))
    return smiles_data, labels

class PubMedBERT(nn.Module):
    def __init__(self, model_name_or_path, hidden_size=256, dropout=0.1, dim_reduction=True):
        super(PubMedBERT, self).__init__()
        self.encoder = BertModel.from_pretrained(model_name_or_path)
        self.dim_reduction = dim_reduction
        # unfreeze_layers = ['layer.10', 'layer.11', 'bert.pooler']
        unfreeze_layers = []
        for name, param in self.encoder.named_parameters():
            if not any(nd in name for nd in unfreeze_layers):
                param.requires_grad = False
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(768, hidden_size)

    def forward(self, x):
        tok, att = x[0], x[1]
        typ = torch.zeros(tok.shape).long().to(tok)
        result = self.encoder(tok, token_type_ids=typ, attention_mask=att)
        h = self.dropout(result[1])
        if self.dim_reduction:
            h = self.fc(h)
        return h


class MolTestDataset(Dataset):
    def __init__(self, data_path, target, task):
        super(Dataset, self).__init__()
        self.smiles_data, self.labels = read_smiles(data_path, target, task)
        # load kg encoding & text encoding
        self.kg_enc = []
        self.text_enc = []
        smi2id_path = data_path.split(".")[0] + "_smi2id.pkl"
        smi2id = pickle.load(open(smi2id_path, "rb"))
        kg_emb = pickle.load(open("/share/project/biomed/hk/hk/Open_DAIR_BioMed/origin-data/BMKG-DP/kg_embed_ace2.pkl", "rb"))
        # print(kg_emb.keys())
        drug_info = json.load(open("/share/project/biomed/hk/hk/Open_DAIR_BioMed/origin-data/BMKG-DP/bmkg-dp_drug.json", "r"))
        tokenizer = BertTokenizer.from_pretrained("/share/project/biomed/hk/hk/Open_DAIR_BioMed/pretrained_lm/pubmedbert_uncased/")
        model = PubMedBERT("/share/project/biomed/hk/hk/Open_DAIR_BioMed/pretrained_lm/pubmedbert_uncased/", dim_reduction=False, dropout=0).to(0)
        cnt = 0
        for smi in self.smiles_data:
            if smi in smi2id:
                if smi2id[smi] in kg_emb:
                    cnt += 1
                    self.kg_enc.append(kg_emb[smi2id[smi]])
                else:
                    self.kg_enc.append(np.zeros(256))
                seq = drug_info[smi2id[smi]]["text"]
                tok_result = tokenizer(seq, max_length=512, truncation=True, return_tensors='pt')
                enc = model((tok_result['input_ids'].to(0), tok_result['attention_mask'].to(0))).detach().cpu()
                self.text_enc.append(enc.squeeze(0))
            else:
                self.kg_enc.append(np.zeros(256))
                self.text_enc.append(torch.zeros(768))
        print("matched to kg: %d / %d" % (cnt, len(self.smiles_data)))

        self.task = task

        self.conversion = 1
        if 'qm9' in data_path and target in ['homo', 'lumo', 'gap', 'zpve', 'u0']:
            self.conversion = 27.211386246
            print(target, 'Unit conversion needed!')

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.smiles_data[index])
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
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
        fp = torch.tensor(fp, dtype=torch.float)
        if self.task == 'classification':
            y = torch.tensor(self.labels[index], dtype=torch.long).view(1,-1)
        elif self.task == 'regression':
            y = torch.tensor(self.labels[index] * self.conversion, dtype=torch.float).view(1,-1)
        kg_x = torch.tensor(self.kg_enc[index], dtype=torch.float)
        text_x = self.text_enc[index]
        data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, fp_x=fp, kg_x=kg_x, text_x=text_x)
        return data

    def __len__(self):
        return len(self.smiles_data)


class MolTestDatasetWrapper(object):
    
    def __init__(self, 
        batch_size, num_workers, valid_size, test_size, 
        data_path, target, task, splitting
    ):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.target = target
        self.task = task
        self.splitting = splitting
        assert splitting in ['random', 'scaffold']

    def get_data_loaders(self):
        train_dataset = MolTestDataset(data_path=self.data_path, target=self.target, task=self.task)
        train_loader, valid_loader, test_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader, test_loader

    def get_train_validation_data_loaders(self, train_dataset):
        if self.splitting == 'random':
            # obtain training indices that will be used for validation
            num_train = len(train_dataset)
            indices = list(range(num_train))
            np.random.shuffle(indices)

            # split = int(np.floor(self.valid_size * num_train))
            # split2 = int(np.floor(self.test_size * num_train))
            # valid_idx, test_idx, train_idx = indices[:split], indices[split:split+split2], indices[split+split2:]
            data = json.load(open("/share/project/biomed/hk/hk/Open_DAIR_BioMed/origin-data/DDI/sider/split.json", "r"))
            train_idx, valid_idx, test_idx = np.array(data["train"]), np.array(data["val"]), np.array(data["test"])
            print(" ".join([str(train_dataset.labels[i]) for i in valid_idx]))
        
        elif self.splitting == 'scaffold':
            train_idx, valid_idx, test_idx = scaffold_split(train_dataset, self.valid_size, self.test_size)

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=train_sampler,
            num_workers=self.num_workers, drop_last=False
        )
        valid_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
            num_workers=self.num_workers, drop_last=False
        )
        test_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=test_sampler,
            num_workers=self.num_workers, drop_last=False
        )

        return train_loader, valid_loader, test_loader
