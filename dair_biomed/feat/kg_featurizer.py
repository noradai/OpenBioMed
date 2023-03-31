from abc import ABC, abstractmethod

import torch

from feat.base_featurizer import BaseFeaturizer
from utils.kg_utils import SUPPORTED_KG, embed

class KGFeaturizer(BaseFeaturizer, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.kg = SUPPORTED_KG[self.config["kg_name"]](self.config["kg_path"])
        self.transform = None

    def set_transform(self, transform):
        self.transform = transform

    @abstractmethod
    def __call__(self, data):
        raise NotImplementedError

class KGIDFeaturizer(KGFeaturizer):
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, data):
        return self.transform(data)

# ugly, redesign later
class KGEFeaturizer(KGFeaturizer):
    def __init__(self, config):
        super().__init__(config)
        self.kge = config["kge"]
        self.embed_dim = config["embed_dim"]

    def __call__(self, data):
        if self.transform is not None:
            data = self.transform[data]
        if data is None or data not in self.kge:
            return torch.zeros(self.embed_dim)
        else:
            return torch.FloatTensor(self.kge[data])

SUPPORTED_KG_FEATURIZER = {
    "id": KGIDFeaturizer, 
    "KGE": KGEFeaturizer
}