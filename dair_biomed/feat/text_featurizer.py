from abc import ABC, abstractmethod
from transformers import BertModel, BertTokenizer

from feat.base_featurizer import BaseFeaturizer

class TextFeaturizer(BaseFeaturizer, ABC):
    def __init__(self):
        super(TextFeaturizer).__init__()
        self.transform = None

    def set_transform(self, transform):
        self.transform = transform

    @abstractmethod
    def __call__(self, data):
        raise NotImplementedError

class TextBertTokFeaturizer(TextFeaturizer):
    def __init__(self, config):
        super(TextBertTokFeaturizer, self).__init__()
        self.max_length = config["max_length"]
        self.tokenizer = BertTokenizer.from_pretrained(config["model_name_or_path"])

    def __call__(self, data):
        if self.transform is not None:
            data = self.transform[data]
        return self.tokenizer(data, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt').data

class TextBertEncFeaturizer(TextFeaturizer):
    def __init__(self, config):
        super(TextBertEncFeaturizer, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(config["model_name_or_path"])
        self.encoder = BertModel.from_pretrained(config["model_name_or_path"])
        self.max_length = config["max_length"]

    def __call__(self, data):
        if self.transform is not None:
            data = self.transform[data]
        data = self.tokenizer(data, max_length=self.max_length, truncation=True, return_tensors='pt')
        return self.encoder(**data)[1]

SUPPORTED_TEXT_FEATURIZER = {
    "BertTokenizer": TextBertTokFeaturizer,
    "BertEncoder": TextBertEncFeaturizer,
}