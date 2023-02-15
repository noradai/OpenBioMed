from abc import ABC, abstractmethod
import logging
logger = logging.getLogger(__name__)

import pickle
import numpy as np
import pandas as pd
import os.path as osp

import torch
from torch.utils.data import Dataset
from torch_geometric import Batch

from feat.drug_featurizer import SUPPORTED_DRUG_FEATURIZER, DrugMultiModalFeaturizer
from feat.cell_featurizer import SUPPORTED_CELL_FEATURIZER
from utils.gene_select import SUPPORTED_GENE_SELECTOR

def _collate_TGSA(samples):
    drugs, cells, labels = map(list, zip(*samples))
    batched_drug = Batch.from_data_list(drugs)
    batched_cell = Batch.from_data_list(cells)
    return batched_drug, batched_cell, torch.tensor(labels)

class DRPDataset(Dataset, ABC):
    def __init__(self, path, config, split="train"):
        super(DRPDataset, self).__init__()
        self.path = path
        self.config = config
        self.gene_selector = SUPPORTED_GENE_SELECTOR[config["cell"]["gene_selector"]](path)
        self._load_data()
        self._featurize()

    @abstractmethod
    def _load_data(self):
        raise NotImplementedError

    def _featurize(self):
        # featurize drug
        if len(self.config["drug"]["modality"]) > 1:
            featurizer = DrugMultiModalFeaturizer(self.config["drug"])
        else:
            featurizer = SUPPORTED_DRUG_FEATURIZER[self.config["drug"]["featurizer"]["structure"]["name"]](**self.config["drug"]["featurizer"]["structure"])
        for key in self.drug_dict:
            smi = self.drug_dict[key]
            self.drug_dict[key] = featurizer(smi)
        
        # featurize cell
        featurizer = SUPPORTED_CELL_FEATURIZER[self.config["cell"]["featurizer"]["name"]](**self.config["cell"]["featureizer"])
        self.cell_dict = featurizer(self.cell_dict)
        if self.config["cell"]["featurizer"] == "TGSA":
            self.predifined_cluster = featurizer.predifined_cluster

        # convert labels to tensor
        self.IC = torch.FloatTensor(self.IC)
        self.response = torch.FloatTensor(self.response)

    def __getitem__(self, index):
        return self.drug_dict[self.drug_index[index]], self.cell_dict[self.cell_index[index]], self.IC[index] if self.config["task"] == "classification" else self.response[index]

    def __len__(self):
        return len(self.IC50)

class GDSC(DRPDataset):
    def __init__(self, path, config, split="train"):
        super(GDSC, self).__init__(path, config)
        self._train_test_split(split)
        
    def _load_data(self):
        # load drug information
        data_drug = np.loadtxt(osp.join(self.path, "GDSC_DrugAnnotation.csv"), delimiter=',', skiprows=1)
        self.drug_dict = dict([
            (data_drug[i][0], data_drug[i][2]) for i in data_drug.shape[0]
        ])
        
        # load cell-line information
        self.cell_dict = {}
        for feat in self.config["cell_line"]["gene_feature"]:
            data_cell = np.loadtxt(osp.join(self.path, "GDSC_" + feat + ".csv"), delimiter=',')
            gene_names = data_cell[0][1:]
            # select genes strongly related to tumor expression
            selected_idx = self.gene_selector(gene_names)
            data_cell = data_cell[1:][selected_idx]
            for cell in data_cell:
                if cell[0] not in self.cell_dict:
                    self.cell_dict[cell[0]] = cell[1:].reshape(-1, 1)
                else:
                    self.cell_dict[cell[0]] = np.concatenate((self.cell_dict[cell[0]], cell[1:].reshape(-1, 1)), axis=1)

        # load drug-cell response information
        data_IC50 = np.loadtxt(osp.join(self.path, "GDSC_DR" + feat + ".csv"), delimeter=',')
        self.drug_index = data_IC50[:, 0]
        self.cell_index = data_IC50[:, 1]
        self.IC = data_IC50[:, 2]
        resp2val = {'R': 1, 'S': 0}
        self.response = np.array([resp2val[x] for x in data_IC50[:, 4]])

    def _train_test_split(self, split):
        if self.config["data"]["split"]["type"] == "random":
            N = len(self.IC)
            train_ratio, val_ratio = self.config["data"]["split"]["train"], self.config["data"]["split"]["val"]
            indexes = np.random.permutation(len(self.IC))
            if split == "train":
                indexes = indexes[:N * train_ratio]
            elif split == "val":
                indexes = indexes[N * train_ratio: N * (train_ratio + val_ratio)]
            else:
                indexes = indexes[N * (train_ratio + val_ratio):]
        self.drug_index = self.drug_index[indexes]
        self.cell_index = self.cell_index[indexes]
        self.IC = self.IC[indexes]
        self.response = self.response[indexes]

class TCGA(DRPDataset):
    def __init__(self, path, config, tumor_type="BRCA_28"):
        super(TCGA, self).__init__(path, config)
        self.tumor_type = tumor_type
        all = ["BRCA_28", "CESC_24", "COAD_8", "GBM_69", "HNSC_45", "KIRC_47", "LUAD_23", "LUSC_20", "PAAD_55", "READ_8", "SARC_30", "SKCM_56", "STAD_30"]

    def _load_data(self):
        # load cell-line data
        feat2file = {"EXP": "xena_gex", "MUT": "xena_mutation"}
        save_path = osp.join(self.path, "celldict_%s.pkl" % (self.config["cell"]["gene_selector"]))
        if osp.exists(save_path):
            self.cell_dict = pickle.load(open(save_path, "rb"))
        else:
            self.cell_dict = {}
            for feat in self.config["cell_line"]["gene_feature"]:
                data_cell = np.loadtxt(osp.join(self.path, feat2file[feat] + ".csv"), delimiter=',')
                gene_names = data_cell[0][1:]
                selected_idx = self.gene_selector(gene_names)
                data_cell = data_cell[1:][selected_idx]
                for cell in data_cell:
                    cell_name = cell[0][:12]
                    if cell_name not in self.cell_dict:
                        self.cell_dict[cell_name] = cell[1:].reshape(-1, 1)
                    else:
                        self.cell_dict[cell_name] = np.concatenate((self.cell_dict[cell_name], cell[1:].reshape(-1, 1)), axis=1)
            pickle.dump(self.cell_dict, open(save_path, "wb"))

        # load drug and its response data
        df = pd.read_csv(osp.join(self.path, "tcga_clinical_data", self.tumor_type + ".csv"))
        drugs = df["smiles"].unique()
        self.drug_dict = dict(zip(drugs, drugs))
        self.drug_index = df["smiles"].to_numpy()
        self.cell_index = df["bcr_patient_barcode"].to_numpy()
        self.response = df["label"].to_numpy()

SUPPORTED_DRP_COLLATE_FN = {
    "TGSA": _collate_TGSA
}

SUPPORTED_DRP_DATASET = {
    "GDSC": GDSC,
    "TCGA": TCGA
}