import logging
import rdkit.Chem as Chem

logger = logging.getLogger(__name__)

class Drug(object):
    def __init__(self, smi, kg_id=None, text=None):
        super(Molecule, self).__init__()
        try:
            self.smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True)
        except Exception:
            logger.warn("Invalid SMILES string, fail to generate molecule")
        self.kg_id = kg_id
        self.text = text

    def featurize(self, modality, featurizer):
        if modality == "structure":
            self.seq_feat = featurizer(self.seq)
        if modality == "kg":
            self.kg_feat = featurizer(self.kg)
        if modality == "text":
            self.text_feat = featurizer(self.text)
