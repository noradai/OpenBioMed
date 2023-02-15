import logging
logger = logging.getLogger(__name__)

import torch
import numpy as np

from torch_geometric.data import Data, Batch
from torch_geometric.nn import graclus, max_pool

from base_featurizer import BaseFeaturizer
from utils.kg_utils import STRING

class CellTGSAFeaturizer(BaseFeaturizer):
    def __init__(self, name, edge_threshold):
        super(CellTGSAFeaturizer, self).__init__()
        self.selected_index_hugo = []
        with open("../datasets/drp/census706.txt", "r") as f:
            line = f.readline().strip("\n").split(",")
            for index in line:
                self.selected_index_hugo.append(index.lstrip('(').rstrip(')'))
        self.ppi_graph = STRING("../datasets/kg/STRING", edge_threshold).node_subgraph(self.selected_index_hugo)
        self.cluster_predefine = self._gen_predefined_cluster()

    def _gen_predefined_cluster(self):
        g = Data(edge_index=torch.tensor(self.ppi_graph, dtype=torch.long), x=torch.zeros(len(self.selected_index_hugo), 1))
        g = Batch.from_data_list([g])
        cluster_predefine = {}
        for i in range(5):
            cluster = graclus(g.edge_index, None, g.x.size(0))
            logger.info("%d nodes at cluster level #%d" % (i, len(cluster.unique())))
            g = max_pool(cluster, g, transform=None)
            cluster_predefine[i] = cluster
        return cluster_predefine

    def __call__(self, data):
        feat = {}
        for cell_name in data:
            feat[cell_name] = Data(
                x=torch.tensor(data[cell_name]),
                edge_index=torch.tensor(self.ppi_graph)
            )

SUPPORTED_CELL_FEATURIZER = {
    "TGSA": CellTGSAFeaturizer,
}
    