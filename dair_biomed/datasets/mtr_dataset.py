"""
Dataset for Molecule-Text Retrieval
"""
from abc import ABC, abstractmethod
import logging
logger = logging.getLogger(__name__)

import os.path as osp
import copy

import rdkit.Chem as Chem
import numpy as np
import torch
from torch.utils.data import Dataset

from feat.drug_featurizer import SUPPORTED_DRUG_FEATURIZER, DrugMultiModalFeaturizer
from feat.text_featurizer import SUPPORTED_TEXT_FEATURIZER

class MTRDataset(Dataset, ABC):
    def __init__(self, path, config):
        super(MTRDataset, self).__init__()
        self.path = path
        self.config = config
        self._load_data()
        self._featurize()

    @abstractmethod
    def _load_data(self):
        raise NotImplementedError

    def _featurize(self):
        # featurize drug with paired text
        featurizer = DrugMultiModalFeaturizer(self.config["drug"])
        featurizer.set_drug2text_dict(self.drug2text)
        self.drugs = [featurizer(drug) for drug in self.drugs]

    def index_select(self, indexes):
        new_dataset = copy.copy(self)
        new_dataset.drugs = [new_dataset.drugs[i] for i in indexes]
        return new_dataset

    def __getitem__(self, index):
        return self.drugs[index]

    def __len__(self):
        return len(self.drugs)

class PCdes(MTRDataset):
    def __init__(self, path, config):
        super(PCdes, self).__init__(path, config)
        self._train_test_split()

    def _load_data(self):
        with open(osp.join(self.path, "align_smiles.txt"), "r") as f:
            drugs = f.readlines()

        with open(osp.join(self.path, "align_des_filt3.txt"), "r") as f:
            texts = f.readlines()[:len(drugs)]

        self.drugs = []
        self.texts = []
        for i, drug in enumerate(drugs):
            try:
                mol = Chem.MolFromSmiles(drug)
                if mol is not None:
                    self.drugs.append(drug)
                    self.texts.append(texts[i])
            except:
                print("2D graph generating error")

        self.drug2text = dict(zip(self.drugs, self.texts))
        print(len(self))

    def _train_test_split(self):
        self.train_index = np.arange(0, 10500)
        self.val_index = np.arange(10500, 12000)
        self.test_index = np.arange(12000, len(self.drugs))

SUPPORTED_MTR_DATASETS = {
    "PCdes": PCdes,
}