from abc import ABC, abstractmethod
import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import os
import json

import torch
from torch.utils.data import Dataset

from feat.drug_featurizer import SUPPORTED_DRUG_FEATURIZER, DrugMultiModalFeaturizer
from feat.protein_featurizer import SUPPORTED_PROTEIN_FEATURIZER, ProteinMultiModalFeaturizer
from utils.kg_utils import SUPPORTED_KG
from utils.split import kfold_split

class DTIDataset(Dataset, ABC):
    def __init__(self, path, config, split_strategy):
        super(DTIDataset, self).__init__()
        self.path = path
        self.config = config
        self.split_strategy = split_strategy
        self._load_data()
        if len(self.config["drug"]["modality"]) > 1:
            kg = SUPPORTED_KG[self.config["kg"]](self.config["kg_path"])
            self.drug2kg, self.drug2text, self.protein2kg, self.protein2text = kg.link(self)
            self.concat_text_first = self.config["concat_text_first"]
        else:
            self.concat_text_first = False
        self._configure_featurizer()
        self._train_test_split()

    @abstractmethod
    def _load_data(self):
        raise NotImplementedError

    @abstractmethod
    def _train_test_split(self):
        raise NotImplementedError

    def _configure_featurizer(self):
        if len(self.config["drug"]["modality"]) > 1:
            self.drug_featurizer = DrugMultiModalFeaturizer(self.config["drug"]["featurizer"])
            self.protein_featurizer = ProteinMultiModalFeaturizer(self.config["protein"]["featurizer"])
            self.drug_featurizer.set_drug2kgid_dict(self.drug2kg)
            self.drug_featurizer.set_drug2text_dict(self.drug2text)
            self.protein_featurizer.set_protein2kgid_dict(self.protein2kg)
            self.protein_featurizer.set_protein2text_dict(self.protein2text)
        else:
            drug_feat_config = self.config["drug"]["featurizer"]["structure"]
            self.drug_featurizer = SUPPORTED_DRUG_FEATURIZER[drug_feat_config["name"]](drug_feat_config)
            protein_feat_config = self.config["protein"]["featurizer"]["structure"]
            self.protein_featurizer = SUPPORTED_PROTEIN_FEATURIZER[protein_feat_config["name"]](protein_feat_config)

    def __getitem__(self, index):
        # since the drug-protein pairs are much larger than the number of drugs and proteins, we featurize them at this step
        drug, protein, label = self.smiles[self.pair_index[index][0]], self.proteins[self.pair_index[index][1]], self.labels[index]
        processed_drug = self.drug_featurizer(drug)
        processed_protein = self.protein_featurizer(protein)
        if self.concat_text_first:
            processed_drug["text"] = processed_drug["text"] + self.drug_featurizer["text"].tokenizer.sep_token + processed_protein.pop("text")
        return processed_drug, processed_protein, label

    def __len__(self):
        return len(self.pair_index)

class DTI4Classification(DTIDataset):
    def __init__(self, path, config, split_strategy):
        super(DTI4Classification, self).__init__(path, config, split_strategy)

    def _load_data(self):
        data = json.load(open(os.path.join(self.path, "drug.json")))
        self.smiles = [item["SMILES"] for item in data]
        drugid2index = dict(zip(data.keys(), range(len(self.smiles))))

        data = json.load(open(os.path.join(self.path, "protein.json")))
        self.proteins = [item["sequence"] for item in data]
        proteinid2index = dict(zip(data.keys(), range(len(self.proteins))))

        df = pd.read_csv(os.path.join(self.path, "data.csv"))
        self.pair_index, self.labels = [], []
        for row in df.iterrows():
            self.pair_index.append((drugid2index[row["drug_id"]], proteinid2index[row["protein_id"]]))
            self.labels.append(row["affinity"])

    def _train_test_split(self):
        if self.split_strategy == "random":
            self.nfolds = 5
            folds = kfold_split(len(self), 5)
            self.folds = []
            for i in range(5):
                self.folds.append({
                    "train": np.concatenate(folds[:i] + folds[i + 1:], axis=0).tolist(), 
                    "test": folds[i]
                })
        # TODO: Add cold-start splittings

class Davis(DTIDataset):
    def __init__(self, path, config):
        super(Davis, self).__init__(path, config)

class KIBA(DTIDataset):
    def __init__(self, path, config):
        super(KIBA, self).__init__(path, config)

SUPPORTED_DTI_DATASETS = {
    "yamanishi08": DTI4Classification,
    "bmkg-dti": DTI4Classification,
    "davis": Davis,
    "kiba": KIBA
}