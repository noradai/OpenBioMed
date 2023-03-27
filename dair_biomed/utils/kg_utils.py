from abc import ABC, abstractmethod
import os.path as osp
import pandas as pd
import numpy as np
import pickle

from utils.gene_select import hugo2ncbi

class KG(object):
    def  __init__(self):
        super(KG, self).__init__()

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

class BMKG(KG):
    def __init__(self, path):
        super(BMKG, self).__init__()
        self.load_drugs(open(osp.join(path, "drug.json")))
        self.load_proteins(open(osp.join(path, "protein.json")))
        self.load_edges(open(osp.join(path, "edge.csv")))

class BMKGv2(KG):
    def __init__(self, path):
        super(BMKGv2, self).__init__()
        self.kg = pickle.load(open(path, "rb"))
        self.adj = {}
        for triplet in self.kg["triplets"]:
            if triplet[0] not in self.adj:
                self.adj[triplet[0]] = [triplet]
            else:
                self.adj[triplet[0]].append(triplet)
            if triplet[2] not in self.adj:
                self.adj[triplet[2]] = [triplet]
            else:
                self.adj[triplet[2]].append(triplet)

class STRING(KG):
    def __init__(self, path, thresh=0.95):
        super(STRING, self).__init__()
        self.thresh = thresh
        self._load_proteins(path)
        self._load_edges(path)

    def _load_proteins(self, path):
        # self.proteins: Dict
        # Key: ensp id
        # Value: kg_id    - index in the knowledge graph
        #        name     - preferred name in HUGO
        #        sequence - amino acid sequence
        #        text     - description
        self.proteins = {}
        self.ncbi2ensp = {}
        df = pd.read_csv(osp.join(path, "9606.protein.info.v11.0.txt"), sep='\t')
        for index, protein in df.iterrows():
            self.proteins[protein['protein_external_id']] = {
                "kg_id": index,
                "name": protein['preferred_name'],
                "text": protein['annotation']
            }
            self.ncbi2ensp[protein['preferred_name']] = protein['protein_external_id']
        # protein sequence
        with open(osp.join(path, "9606.protein.sequences.v11.0.fa"), 'r') as f:
            id, buf = None, ''
            for line in f.readlines():
                if line.startswith('>'):
                    if id is not None:
                        self.proteins[id]["sequence"] = buf
                    id = line.lstrip('>').rstrip("\n")
                    buf = ''
                else:
                    buf = buf + line.rstrip("\n")
            
    def _load_edges(self, path):
        edges = pd.read_csv(osp.join(path, "9606.protein.links.v11.0.txt"), sep=' ')
        selected_edges = edges['combined_score'] > (self.thresh * 1000)
        self.edges = edges[selected_edges][["protein1", "protein2"]].values.tolist()
        for i in range(len(self.edges)):
            self.edges[i][0] = self.proteins[self.edges[i][0]]["kg_id"]
            self.edges[i][1] = self.proteins[self.edges[i][1]]["kg_id"]

    def node_subgraph(self, node_idx, format="hugo"):
        if format == "hugo":
            node_idx = [hugo2ncbi[x] for x in node_idx]
        node_idx = [self.ncbi2ensp[x] if x in self.ncbi2ensp else x for x in node_idx]
        ensp2subgraphid = dict(zip(node_idx, range(len(node_idx))))
        names_ensp = list(self.proteins.keys())
        edge_index = []
        for i in self.edges:
            p0, p1 = names_ensp[i[0]], names_ensp[i[1]]
            if p0 in node_idx and p1 in node_idx:
                edge_index.append((ensp2subgraphid[p0], ensp2subgraphid[p1]))
                edge_index.append((ensp2subgraphid[p1], ensp2subgraphid[p0]))
        edge_index = list(set(edge_index))
        return np.array(edge_index, dtype=np.int64).T

    def __str__(self):
        return "Collected from string v11.0 database, totally %d proteins and %d edges" % (len(self.proteins), len(self.edges))

SUPPORTED_KG = {"BMKG": BMKG, "STRING": STRING}

def sample(graph, node_id, sampler):
    ### Inputs:
    # G: object of KG
    # node_id: the id of the center node
    # sampler: sampling strategy, e.g. ego-net
    ### Outputs:
    # G': graph in pyg Data(x, y, edge_index)
    pass

def embed(graph, model='ProNE', dim=256):
    ### Inputs:
    # G: object of KG
    # model: network embedding model, e.g. ProNE
    ### Outputs:
    # emb: numpy array, |G| * dim
    pass

def bfs(graph, node_id, max_depth):
    ### Inputs:
    # G: object of KG
    # node_id: the id of the starting node
    # max_depth: the max number of steps to go
    ### Outputs:
    # dist: a list, dist[i] is the list of i-hop neighbors
    pass