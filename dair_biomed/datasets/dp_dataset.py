from abc import ABC, abstractmethod
import logging
logger = logging.getLogger(__name__)

import copy
from enum import Enum
import numpy as np
import pandas as pd
import os
from rdkit import Chem

import torch
from torch.utils.data import Dataset

from feat.drug_featurizer import SUPPORTED_DRUG_FEATURIZER, DrugMultiModalFeaturizer
from utils.split import random_split, scaffold_split
from utils.utils import Normalizer

class Task(Enum):
    CLASSFICATION = 0
    REGRESSION = 1

class DPDataset(Dataset, ABC):
    def __init__(self, path, config):
        super(DPDataset, self).__init__()
        self.path = path
        self.config = config
        self._load_data()
        self._featurize()

    @abstractmethod
    def _load_data(self, path):
        raise NotImplementedError

    def _featurize(self):
        if len(self.config["drug"]["modality"]) > 1:
            featurizer = DrugMultiModalFeaturizer(self.config["drug"])
        else:
            feat_config = self.config["drug"]["featurizer"]["structure"]
            featurizer = SUPPORTED_DRUG_FEATURIZER[feat_config["name"]](feat_config)

        if "kg" in self.config["drug"]["modality"]:
            featurizer.set_drug2kgid_dict(self.drug2kg)
        if "text" in self.config["drug"]["modality"]:
            featurizer.set_drug2text_dict(self.drug2text)
        self.drugs = [featurizer(drug) for drug in self.drugs]
        self.labels = [torch.tensor(label) for label in self.labels]

    def index_select(self, indexes):
        new_dataset = copy.copy(self)
        new_dataset.drugs = [new_dataset.drugs[i] for i in indexes]
        new_dataset.labels = [new_dataset.labels[i] for i in indexes]
        return new_dataset

    def __getitem__(self, index):
        return self.drugs[index], self.labels[index]

    def __len__(self):
        return len(self.drugs)

class MoleculeNetDataset(DPDataset):
    name2target = {
        "BBBP":     ["p_np"],
        "Tox21":    ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", 
                     "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"],
        "ClinTox":  ["CT_TOX", "FDA_APPROVED"],
        "HIV":      ["HIV_active"],
        "Bace":     ["class"],
        "SIDER":    ["Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues", 
                     "Eye disorders", "Investigations", "Musculoskeletal and connective tissue disorders", 
                     "Gastrointestinal disorders", "Social circumstances", "Immune system disorders", 
                     "Reproductive system and breast disorders", 
                     "Neoplasms benign, malignant and unspecified (incl cysts and polyps)", 
                     "General disorders and administration site conditions", "Endocrine disorders", 
                     "Surgical and medical procedures", "Vascular disorders", 
                     "Blood and lymphatic system disorders", "Skin and subcutaneous tissue disorders", 
                     "Congenital, familial and genetic disorders", "Infections and infestations", 
                     "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders", 
                     "Renal and urinary disorders", "Pregnancy, puerperium and perinatal conditions", 
                     "Ear and labyrinth disorders", "Cardiac disorders", 
                     "Nervous system disorders", "Injury, poisoning and procedural complications"],
        "MUV":      ['MUV-692', 'MUV-689', 'MUV-846', 'MUV-859', 'MUV-644', 'MUV-548', 'MUV-852',
                     'MUV-600', 'MUV-810', 'MUV-712', 'MUV-737', 'MUV-858', 'MUV-713', 'MUV-733',
                     'MUV-652', 'MUV-466', 'MUV-832'],
        "FreeSolv": ["expt"],
        "ESOL":     ["measured log solubility in mols per litre"],
        "Lipo":     ["exp"],
        "qm7":      ["u0_atom"],
        "qm8":      ["E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0", 
                     "f1-PBE0", "f2-PBE0", "E1-CAM", "E2-CAM", "f1-CAM","f2-CAM"],
        "qm9":      ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'cv']
    }
    name2task = {
        "BBBP":     Task.CLASSFICATION,
        "Tox21":    Task.CLASSFICATION,
        "ClinTox":  Task.CLASSFICATION,
        "HIV":      Task.CLASSFICATION,
        "Bace":     Task.CLASSFICATION,
        "SIDER":    Task.CLASSFICATION,
        "MUV":      Task.CLASSFICATION,
        "FreeSolv": Task.REGRESSION,
        "ESOL":     Task.REGRESSION,
        "Lipo":     Task.REGRESSION,
        "qm7":      Task.REGRESSION,
        "qm8":      Task.REGRESSION,
        "qm9":      Task.REGRESSION
    }

    def __init__(self, path, config, name="BBBP"):
        if name not in self.name2target:
            raise ValueError("%s is not a valid moleculenet task!" % name)
        path = os.path.join(path, name.lower(), name.lower() + ".csv")
        self.name = name
        self.targets = self.name2target[name]
        self.task = self.name2task[name]
        super(MoleculeNetDataset, self).__init__(path, config)
        self._train_test_split()
        self._normalize()

    def _load_data(self):
        data = pd.read_csv(self.path)
        smiles = data['smiles'].to_numpy()
        labels = data[self.targets].to_numpy()
        self.smiles, self.drugs, self.labels = [], [], []
        for i, drug in enumerate(smiles):
            mol = Chem.MolFromSmiles(drug)
            if mol is not None:
                self.smiles.append(drug)
                self.drugs.append(drug)
                self.labels.append(labels[i])

        if self.name == "qm9":
            for target in ['homo', 'lumo', 'gap', 'zpve', 'u0']:
                self.labels[:, self.targets.index(target)] *= 27.211386246

    def _train_test_split(self, strategy="scaffold"):
        if strategy == "random":
            self.train_index, self.val_index, self.test_index = random_split(len(self), 0.1, 0.1)
        elif strategy == "scaffold":
            self.train_index, self.val_index, self.test_index = scaffold_split(self, 0.1, 0.1)

    def _normalize(self):
        if self.name in ["qm7", "qm9"]:
            self.normalizer = []
            for i in range(len(self.targets)):
                self.normalizer.append(Normalizer(self.labels[:, i]))
                self.labels[:, i] = self.normalizer[i].norm(self.labels[:, i])
        else:
            self.normalizer = [None] * len(self.targets)

SUPPORTED_DP_DATASETS = {
    "MoleculeNet": MoleculeNetDataset
}