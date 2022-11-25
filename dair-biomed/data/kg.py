import os.path as osp

class KG(object):
    def  __init__(self):
        super(KG, self).__init__()

    def load_drugs(self, path):

    def load_proteins(self, path):

    def load_edges(self, path):

    def get_drug(self, smi):

    def get_protein(self, seq):

    def __str__(self):

class BMKG_DP(KG):
    def __init__(self, path):
        super(BMKG_DP, self).__init__()
        self.load_drugs(open(osp.join(path, "drug.json")))
        self.load_proteins(open(osp.join(path, "protein.json")))
        self.load_edges(open(osp.join(path, "edge.csv")))

class BMKG(KG):
    def __init__(self, path):
        super(BMKG, self).__init__()

SUPPORTED_KG = {"BMKG_DP": BMKG_DP, "BMKG": BMKG}