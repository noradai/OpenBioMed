import logging
logger = logging.getLogger(__name__)

class Protein(object):
    def __init__(self, seq, kg_id=None, text=None):
        super(Protein, self).__init__()
        self.seq = seq
        self.kg_id = kg_id
        self.text = text

    def featurize(self, modality, featurizer):
        if modality == "structure":
            self.seq_feat = featurizer(self.seq)
        if modality == "kg":
            self.kg_feat = featurizer(self.kg)
        if modality == "text":
            self.text_feat = featurizer(self.text)