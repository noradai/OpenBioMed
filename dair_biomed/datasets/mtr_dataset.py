"""
Dataset for Molecule-Text Retrieval
"""
from abc import ABC, abstractmethod
import logging
logger = logging.getLogger(__name__)

import os.path as osp
import copy

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
        # featurize drug
        if len(self.config["drug"]["modality"]) > 1:
            featurizer = DrugMultiModalFeaturizer(self.config["drug"])
        else:
            featurizer = SUPPORTED_DRUG_FEATURIZER[self.config["drug"]["featurizer"]["structure"]["name"]](**self.config["drug"]["featurizer"]["structure"])
        self.drugs = [featurizer(drug) for drug in self.drugs]

        # featurize text
        featurizer = SUPPORTED_TEXT_FEATURIZER[self.config["text"]["featurizer"]["name"]](**self.config["text"]["featurizer"])
        self.texts = [featurizer(text) for text in self.texts]

    def index_select(self, indexes):
        new_dataset = copy.copy(self)
        new_dataset.drugs = [new_dataset.drugs[i] for i in indexes]
        new_dataset.texts = [new_dataset.texts[i] for i in indexes]
        return new_dataset

    def __call__(self, index):
        return self.drugs[index], self.texts[index]

    def __len__(self):
        return len(self.drugs)

class PCdes(MTRDataset):
    def __init__(self, path):
        super().__init__(path)

    def _load_data(self):
        with open(osp.join(self.path, "align_smiles.txt"), "r") as f:
            self.drugs = f.readlines()
        
        with open(osp.join(self.path, "align_des_filt3.txt"), "r") as f:
            self.texts = f.readlines()[:len(self.drugs)]