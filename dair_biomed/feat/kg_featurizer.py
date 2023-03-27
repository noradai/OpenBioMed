from abc import ABC, abstractmethod

from feat.base_featurizer import BaseFeaturizer
from utils.kg_utils import SUPPORTED_KG

class KGFeaturizer(BaseFeaturizer, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.kg = SUPPORTED_KG[config["kg_name"]](config["kg_path"])
        self.transform = None

    def set_transform(self, transform):
        self.transform = transform

    @abstractmethod
    def __call__(self, data):
        raise NotImplementedError

class KGIDFeaturizer(KGFeaturizer):
    def __init__(self, config):
        super().__init__(config)
        if not hasattr(self.kg, "ent_dict") or not hasattr(self.kg, "rel_dict"):
            raise AttributeError
        else:
            self.set_transform(self.kg["ent_dict"])

    def __call__(self, data):
        if self.transform is not None:
            data = self.transform[data]
        return self.kg[data]

class KGEFeaturizer(KGFeaturizer):
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, data):
        pass

SUPPORTED_KG_FEATURIZER = {KGIDFeaturizer, KGEFeaturizer}