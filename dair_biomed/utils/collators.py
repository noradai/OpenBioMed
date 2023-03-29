import torch
from torch_geometric.data import Data, Batch
from transformers import BatchEncoding, DataCollatorWithPadding, BertTokenizer, T5Tokenizer

name2tokenizer = {
    "bert": BertTokenizer,
    "t5": T5Tokenizer
}

def ToDevice(obj, device):
    if isinstance(obj, dict):
        for k in obj:
            obj[k] = ToDevice(obj[k], device)
        return obj
    elif isinstance(obj, tuple) or isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = ToDevice(obj[i], device)
        return obj
    else:
        return obj.to(device)

class BaseCollator(object):
    def __init__(self, config):
        self.config = config
        self._build(config)

    def _collate_single(self, data, config):
        if isinstance(data[0], Data):
            return Batch.from_data_list(data)
        elif torch.is_tensor(data[0]):
            return torch.stack([x.squeeze() for x in data])
        elif isinstance(data[0], BatchEncoding):
            return config["collator"](data)
        elif isinstance(data[0], dict):
            result = {}
            for key in data[0]:
                result[key] = self._collate_single([x[key] for x in data], config[key])
            return result

    def _collate_multiple(self, data, config):
        cor = []
        flatten_data = []
        for x in data:
            cor.append(len(flatten_data))
            flatten_data += x
        cor.append(len(flatten_data))
        return (cor, self._collate_single(flatten_data, config),)

    def _build(self, config):
        if not isinstance(config, dict):
            return
        if "model_name_or_path" in config:
            config["collator"] = DataCollatorWithPadding(
                tokenizer=name2tokenizer[config["transformer_type"]].from_pretrained(config["model_name_or_path"]),
                padding=True
            )
            return
        for key in config:
            self._build(config[key])

class DrugCollator(BaseCollator):
    def __init__(self, config):
        self.config = config
        self._build(config)

    def __call__(self, drugs):
        if len(self.config["modality"]) > 1:
            batch = {}
            for modality in self.config["modality"]:
                batch[modality] = self._collate_single([drug[modality] for drug in drugs], self.config["featurizer"][modality])
        else:
            batch = self._collate_single(drugs)
        return batch

class ProteinCollator(BaseCollator):
    def __init__(self, config):
        super(ProteinCollator, self).__init__(config)
        self.config = config

    def __call__(self, proteins):
        if len(self.config["modality"]) > 1:
            batch = {}
            for modality in self.config["modality"]:
                if isinstance(proteins[0][modality], list):
                    batch[modality] = self._collate_multiple([drug[modality] for drug in proteins], self.config["featurizer"][modality])
                else:
                    batch[modality] = self._collate_single([drug[modality] for drug in proteins], self.config["featurizer"][modality])
        else:
            batch = self._collate_single(proteins)
        return batch

class DPCollator(object):
    def __init__(self, config):
        self.config = config
        self.drug_collator = DrugCollator(config)

    def __call__(self, data):
        drugs, labels = map(list, zip(*data))
        return self.drug_collator(drugs), torch.stack(labels)

class DTICollator(object):
    def __init__(self, config):
        self.config = config
        self.drug_collator = DrugCollator(config["drug"])
        self.protein_collator = ProteinCollator(config["protein"])

    def __call__(self, data):
        drugs, prots, labels = map(list, zip(*data))
        return self.drug_collator(drugs), self.protein_collator(prots), torch.stack(labels)