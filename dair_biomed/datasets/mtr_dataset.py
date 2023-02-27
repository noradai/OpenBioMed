"""
Dataset for Molecule-Text Retrieval
"""
from abc import ABC, abstractmethod
import logging
logger = logging.getLogger(__name__)

import os.path as osp
import copy

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

    def __call__(self, index):
        return self.drugs[index]

    def __len__(self):
        return len(self.drugs)

class PCdes(MTRDataset):
    def __init__(self, path):
        super().__init__(path)
        self._train_test_split()

    def _load_data(self):
        with open(osp.join(self.path, "align_smiles.txt"), "r") as f:
            self.drugs = f.readlines()
        
        with open(osp.join(self.path, "align_des_filt3.txt"), "r") as f:
            self.texts = f.readlines()[:len(self.drugs)]

        self.drug2text = dict(zip(self.drugs, self.texts))

    def _train_test_split(self):
        self.train_index = np.arange(0, 10500)
        self.val_index = np.arange(10500, 12000)
        self.test_index = np.arange(12000, 15000)

SUPPORTED_MTR_DATASETS = {
    "PCdes": PCdes,
}