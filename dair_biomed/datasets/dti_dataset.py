from abc import ABC, abstractmethod
import logging
logger = logging.getLogger(__name__)

import numpy as np
import os.path as osp

import torch
from torch_geometric.data import Data, Dataset

from data.drug import Drug
from data.protein import Protein
from data.kg import SUPPORTED_KG
from feat.drug_featurizer import SUPPORTED_DRUG_FEATUREIZER

class DTIDataset(Dataset, ABC):
    def __init__(self, path, config):
        super(DTIDataset, self).__init__()
        self.path = path
        self.config = config
        self.load_data(path)

    @abstractmethod
    def load_data(self, path):
        raise NotImplementedError

    def compile_data(self):
        """
        Obtain KG and text data
        """
        if "external" in self.config:
            self.kg = SUPPORTED_KG[self.config["external"]]()
            for i in range(len(self.drugs)):
                self.drugs[i] = kg.get_drug(self.drugs[i])
            for i in range(len(self.proteins)):
                self.proteins[i] = kg.get_protein(self.proteins[i])

    def featurize(self):
        for entity in ["drug", "protein"]:
            for modality in self.config[entity]:
                conf = self.config[entity][modality]
                featurizer = SUPPORTED_DRUG_FEATUREIZER[conf["type"]][conf]
                for i in range(len(self.drugs)):
                    self.drugs[i].featurize(modality, featurizer)

    def split(self, splitter='random'):
        pass

    def get_data_loaders(self):
        # return train_loader, valid_loader, test_loader
        pass

    def __getitem__(self, index):
        # pyg Data(x=, y=, text_feat_drug, text_feat_protein, ...)
        pass

    def __len__(self):
        pass

class Yamanishi08(DTIDataset):
    def __init__(self, path, config):
        super(Yamanishi08, self).__init__(path, config)

    def load_data(self, path):
        """
        # Drugs are firstly SMILES strings, and the Drug object could be compiled with external data
        self.drugs = 
        self.proteins =
        self.pairs = 
        self.labels = 
        """

class BMKG(DTIDataset):

class Davis(DTIDataset):

class KIBA(DTIDataset):