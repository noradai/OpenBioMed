import torch
from torch_geometric.data import Data, Batch
from transformers import BatchEncoding

def ToDevice(obj, device):
    if isinstance(obj, dict):
        for k in obj:
            obj[k] = ToDevice(obj[k], device)
        return obj
    else:
        return obj.to(device)

class DrugCollator(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, drugs):
        batch = {}
        if len(self.config["modality"]) > 1:
            for modality in self.config["modality"]:
                if isinstance(drugs[0][modality], Data):
                    batch[modality] = Batch.from_data_list([drug[modality] for drug in drugs])
                elif torch.is_tensor(drugs[0][modality]):
                    batch[modality] = torch.stack([drug[modality].squeeze() for drug in drugs])
                elif isinstance(drugs[0][modality], dict):
                    batch[modality] = {}
                    for key in drugs[0][modality]:
                        batch[modality][key] = torch.stack([drug[modality][key].squeeze() for drug in drugs])
        else:
            if isinstance(drugs[0], Data):
                batch = Batch.from_data_list([drug for drug in drugs])
            elif torch.is_tensor(drugs[0]):
                batch = torch.stack([drug.squeeze() for drug in drugs])
        return batch

class DPCollator(object):
    def __init__(self, config):
        self.config = config
        self.drug_collator = DrugCollator(config)

    def __call__(self, data):
        drugs, labels = map(list, zip(*data))
        return self.drug_collator(drugs), torch.stack(labels)