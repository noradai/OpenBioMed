from abc import ABC

class BaseFeaturizer(ABC):
    def __init__(self):
        super(BaseFeaturizer, self).__init__()
    
    @abstractmethod
    def featurize(self, data):
        raise NotImplementedError