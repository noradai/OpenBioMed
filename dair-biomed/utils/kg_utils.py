def sample(G, node_id, sampler):
    ### Inputs:
    # G: object of KG
    # node_id: the id of the center node
    # sampler: sampling strategy, e.g. ego-net
    ### Outputs:
    # G': graph in pyg form (x, y, edge_index)

def embed(G, model='ProNE', dim=256):
    ### Inputs:
    # G: object of KG
    # model: network embedding model, e.g. ProNE
    ### Outputs:
    # emb: numpy array, |G| * dim

def bfs(G, node_id, max_depth):
    ### Inputs:
    # G: object of KG
    # node_id: the id of the starting node
    # max_depth: the max number of steps to go
    ### Outputs:
    # dist: a list, dist[i] is the list of i-hop neighbors