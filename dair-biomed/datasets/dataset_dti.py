import os
import pickle
import json
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from rdkit import Chem
from rdkit.Chem import AllChem, ChemicalFeatures
from rdkit import RDLogger,RDConfig                                                                                                                                                               
RDLogger.DisableLog('rdApp.*')  

import networkx as nx
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

from sklearn.metrics import jaccard_score
from PyBioMed.PyProtein import CTD
from sklearn.preprocessing import OneHotEncoder
from transformers import BertTokenizer

class UFS(object):
    def __init__(self, n):
        self.fa = list(range(n))
    
    def merge(self, x, y):
        self.fa[x] = self.find(y)

    def find(self, x):
        self.fa[x] = self.find(self.fa[x]) if self.fa[x] != x else x
        return self.fa[x]

smiles_char = ['?', '#', '%', ')', '(', '+', '-', '.', '1', '0', '3', '2', '5', '4',
       '7', '6', '9', '8', '=', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I',
       'H', 'K', 'M', 'L', 'O', 'N', 'P', 'S', 'R', 'U', 'T', 'W', 'V',
       'Y', '[', 'Z', ']', '_', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i',
       'h', 'm', 'l', 'o', 'n', 's', 'r', 'u', 't', 'y']
amino_char = ['?', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'O',
       'N', 'Q', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', 'X', 'Z']
VOCAB_PROTEIN = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
				"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
				"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
				"U": 19, "T": 20, "W": 21, 
				"V": 22, "Y": 23, "X": 24, 
				"Z": 25 }

MAX_SEQ_DRUG = 357
MAX_SEQ_PROTEIN = 1024
SEQ_LEN = 512

onehot_drug = OneHotEncoder().fit(np.array(smiles_char).reshape(-1, 1))
onehot_prot = OneHotEncoder().fit(np.array(amino_char).reshape(-1, 1))

def trunc(s):
    return " ".join(s.split(" ")[:256])

def seqs2int(target):
    return [VOCAB_PROTEIN[s] for s in target] 

def tok_protein(x, tokenizer):
    temp = list(x.upper())
    if tokenizer == "cnn":
        temp = [i if i in amino_char else '?' for i in temp]
        if len(temp) < MAX_SEQ_PROTEIN:
            temp = temp + ['?'] * (MAX_SEQ_PROTEIN - len(temp))
        else:
            temp = temp[:MAX_SEQ_PROTEIN]
        return onehot_prot.transform(np.array(temp).reshape(-1, 1)).toarray().T
    elif tokenizer == "mcnn":
        temp = [VOCAB_PROTEIN[i] if i in VOCAB_PROTEIN else 0 for i in x]
        if len(temp) < MAX_SEQ_PROTEIN:
            temp = temp + [0] * (MAX_SEQ_PROTEIN - len(temp))
        else:
            temp = temp [:MAX_SEQ_PROTEIN]
        return temp
    elif tokenizer == "transformer":
        if len(temp) <= SEQ_LEN - 2:
            seq1 = temp
            seq1 = add_padding(transformer_prot.encode(seq1), SEQ_LEN)
            seq2 = add_padding([], SEQ_LEN)
        else:
            seq1, seq2 = temp[:SEQ_LEN - 2], temp[SEQ_LEN - 2: min(len(temp), 2 * (SEQ_LEN - 2))]
            seq1 = add_padding(transformer_prot.encode(seq1), SEQ_LEN)
            seq2 = add_padding(transformer_prot.encode(seq2), SEQ_LEN)
        return [seq1, seq2]

class KG(object):
    def __init__(self):
        super(KG, self).__init__()
        self.drugs = None
        self.proteins = None
        self.edges = None
        self.drugs_dict = {}
        self.proteins_dict = {}
        self.G = nx.Graph()
        self.kg_embedding = None

    # TODO: 现在默认json里边存的是个dict，key是id，value是一个dict，里面包含各种信息
    def load_node(self, path):
        with open(path, "r") as f:
            node_dict = json.load(f)
        return node_dict

    def load_drugs(self, path, save=False, save_path=None):
        self.drugs = self.load_node(path)
        for key, value in self.drugs.items():
            smile = value["SMILES"]
            self.drugs_dict[smile] = {"bmkg_id": str(key), "text": value["text"], "fingerprint": value["fingerprint"]}
        if save:
            if not save_path:
                save_path = osp.join(osp.dirname(path), "SMILES_dict.json")
            with open(save_path, 'w') as f:
                json.dump(self.drugs_dict, f)

    def load_proteins(self, path, save=False, save_path=None):
        self.proteins = self.load_node(path)
        for key, value in self.proteins.items():
            seq = value["sequence"]
            self.proteins_dict[seq] = {"bmkg_id": str(key), "text": value["text"], "descriptor": value["descriptor"]}
        if save:
            if not save_path:
                save_path = osp.join(osp.dirname(path), "seqs_dict.json")
            with open(save_path, 'w') as f:
                json.dump(self.proteins_dict, f)

    # TODO: 发现有些id不在self.drugs和self.proteins里
    def get_node_info(self, id):
        node, text, feature = None, None, None
        try:
            if isinstance(id, str) and id.startswith("DB"):
                node = self.drugs[id]["SMILES"]
                text = self.drugs[id]["text"]
                feature = self.drugs[id]["fingerprint"]
            else:
                node = self.proteins[id]["sequence"]
                text = self.proteins[id]["text"]
                feature = self.proteins[id]["descriptor"]
            return node, text, feature
        except Exception as e:
            # print(e)
            return node, text, feature

    def load_edges(self, path):
        edge_data = pd.read_csv(path, delimiter=',')
        print(f"The shape of KG is {edge_data.shape}")
        # edges = edge_data.values.tolist()
        for index, edge in tqdm(edge_data.iterrows()):
            head, tail = str(edge['x_id']), str(edge['end_id'])
            self.G.add_edge(head, tail)

    def save_graph(self, path):
        pass

    def get_drug(self, smi, radius=2):
        try:
            drug = self.drugs_dict[smi]
            drug_id = drug["bmkg_id"]
            drug_embedding = self.kg_embedding[drug_id]
            # drug_graph = nx.ego_graph(self.G, drug_id, radius)
            drug_graph = None
            return (drug, drug_graph, drug_embedding)
        except Exception as e:
            # print(e)
            return (None, None, None)

    def get_protein(self, seq, radius=2):
        try:
            protein = self.proteins_dict[seq]
            protein_id = protein["bmkg_id"]
            protein_embedding = self.kg_embedding[protein_id]
            # protein_graph = nx.ego_graph(self.G, protein, radius)
            protein_graph = None
            return (protein, protein_graph, protein_embedding)
        except Exception as e:
            # print(e) 
            return (None, None, None)

    def model_train():
        pass

    def __str__(self):
        pass


class BMKG_DP(KG):

    def __init__(self, path):
        super(BMKG_DP, self).__init__()
        self.kg_embedding = pickle.load(open(path + "/" + "kg_embed_ace2.pickle", "rb"))
        self.load_drugs(osp.join(path, "bmkg-dp_drug.json"), save=True, save_path="")
        self.load_proteins(osp.join(path, "bmkg-dp_protein.json"), save=True, save_path="")
        # self.load_edges(osp.join(path, "kg_data.csv"))


class BMKG(KG):
    def __init__(self, path):
        super(BMKG, self).__init__()
        pass
SUPPORTED_KG = {"BMKG_DP": BMKG_DP, "BMKG": BMKG}



class PubMedBERT(nn.Module):
    def __init__(self, model_name_or_path, hidden_size=256, dropout=0.1, dim_reduction=True):
        super(PubMedBERT, self).__init__()
        self.encoder = BertModel.from_pretrained(model_name_or_path)
        self.dim_reduction = dim_reduction
        # unfreeze_layers = ['layer.10', 'layer.11', 'bert.pooler']
        unfreeze_layers = []
        for name, param in self.encoder.named_parameters():
            if not any(nd in name for nd in unfreeze_layers):
                param.requires_grad = False
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(768, hidden_size)

    def forward(self, x):
        tok, att = x[0], x[1]
        typ = torch.zeros(tok.shape).long().to(tok)
        result = self.encoder(tok, token_type_ids=typ, attention_mask=att)
        h = self.dropout(result[1])
        if self.dim_reduction:
            h = self.fc(h)
        return h

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    encoding = one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown'])
    encoding += one_of_k_encoding(atom.GetDegree(), [0,1,2,3,4,5,6,7,8,9,10]) + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0,1,2,3,4,5,6,7,8,9,10]) 
    encoding += one_of_k_encoding_unk(atom.GetImplicitValence(), [0,1,2,3,4,5,6,7,8,9,10]) 
    encoding += one_of_k_encoding_unk(atom.GetHybridization(), [
                      Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                      Chem.rdchem.HybridizationType.SP3D2, 'other']) 
    encoding += [atom.GetIsAromatic()]

    try:
        encoding += one_of_k_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
    except:
        encoding += [0, 0] + [atom.HasProp('_ChiralityPossible')]
    
    return np.array(encoding)
    
def mol_to_graph(mol):
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature/np.sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

    if len(edges) == 0:
        return features, [[0, 0]]

    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return features, edge_index

class Yamanishi08(Dataset):
    def __init__(self, config, type):
        super(Yamanishi08, self).__init__()
        self.csv_name = type
        self.config = config
        self.standard_fold_path = config['data']['standard_fold_path']
        self.task = "classification"
        self.split = config['data']['split']
        self.origin_datapath = config['data']['origin_datapath']
        self.cold = config['data']['type']
        self.fold_num = config['data']['fold_num']
        self.split_num = config['data']['n_fold']
        self.prot_tokenizer = 'mcnn'

        # drug info:
        self.drug_ids, self.drug_smiles = self.load_smiles(osp.join(self.origin_datapath, '791drug_struc.csv'))
        self.fp = np.loadtxt(osp.join(self.origin_datapath, 'morganfp.txt'), delimiter=',')
        # drug map:
        self.id2smiles = dict(zip(self.drug_ids, self.drug_smiles))
        self.smiles2id = dict(zip(self.drug_smiles, self.drug_ids))

        self.graph_dict = dict()
        for smile in tqdm(self.drug_smiles, total=len(self.drug_smiles)):
            mol = Chem.MolFromSmiles(smile)
            if mol == None:
                print("Unable to process: ", smile)
                continue
            self.graph_dict[smile] = mol_to_graph(mol)
            
        # target info:
        self.target_ids, target_seq, self.seq2id, self.id2seq = self.load_protein_seq(osp.join(self.origin_datapath, '989proseq.csv'), self.prot_tokenizer)
        self.id2tokseq = dict(zip(self.target_ids, target_seq))
        self.ctd = np.loadtxt(osp.join(self.origin_datapath, 'pro_ctd.txt'), delimiter=',')
        

        # gen standard data
        if not osp.exists(self.standard_fold_path):
                os.mkdir(self.standard_fold_path)
                self.standard_split_08(self.origin_datapath)
        self.cold_path = self.standard_fold_path +'/' + self.cold
        if not osp.exists(self.cold_path):
                os.mkdir(self.cold_path)
                self.gen_08_cold(self.standard_fold_path + '/' + 'data.csv', self.cold)

        # data used:
        if self.cold == "std":
            self.drug_id, self.target_id, self.labels = self.load_data(osp.join(self.standard_fold_path, self.cold, 'data_' + self.split + '.csv'))
        else:
            self.drug_id, self.target_id, self.labels = self.load_cold_data(self.standard_fold_path)

        # kg info:
        #kg
        kg_path = '/share/project/biomed/hk/hk/Open_DAIR_BioMed/origin-data/BMKG-DP'
        tokenizer = BertTokenizer.from_pretrained("/share/project/biomed/hk/hk/Open_DAIR_BioMed/pretrained_lm/pubmedbert_uncased/")
        model = PubMedBERT("/share/project/biomed/hk/hk/Open_DAIR_BioMed/pretrained_lm/pubmedbert_uncased/", dropout=0, dim_reduction=False).to(0)
        bmkg = BMKG_DP(kg_path)
        self.kg_enc = []
        self.text_enc = []
        cnt_d = 0
        cnt_p = 0
        for i, d_id in enumerate(self.drug_id):
            # print(i,d_id)
            smi = self.id2smiles[d_id]
            seq = self.id2seq[self.target_id[i]]
            
            drug, drug_graph, drug_embedding = bmkg.get_drug(smi)
            protein, protein_graph, protein_embedding = bmkg.get_protein(seq)

            if drug == None:
                text_d = ''
                kg_d = np.zeros(256)
            else:
                cnt_d+=1
                kg_d =drug_embedding
                text_d = drug['text']

            if protein == None:
                kg_p = np.zeros(256)
                text_p = ''
            else:
                cnt_p +=1
                kg_p =protein_embedding
                text_p = protein['text']
                
            # print('kg_p=============', kg_p.shape)
            # print("kg_d==============", kg_d.shape)
            

            kg = np.concatenate((kg_d, kg_p), axis=0)
            # print('kg===========',kg.shape)
            self.kg_enc.append(kg)
            text = trunc(text_d) + " [SEP] " + trunc(text_p)
            text = tokenizer(text, max_length=512, truncation=True, return_tensors='pt')
            text = model((text['input_ids'].to(0), text['attention_mask'].to(0))).detach().cpu()
            self.text_enc.append(text)
        print("matched to kg: %d / %d drug; %d / %d protein" % (cnt_d, len(self.drug_id), cnt_p, len(self.target_id)))
            

    def __getitem__(self, index):
        drug_id = self.drug_id[index]
        smi = self.id2smiles[drug_id]
        target_id = self.target_id[index]
        seq = self.id2seq[target_id]

        x, edge_index = self.graph_dict[smi]
        drug_fp = self.get_finger(smi)
        prot_desc = self.get_ctd(seq)
        
        target = seqs2int(seq)
        target_len = 1200
        if len(target) < target_len:
            target = np.pad(target, (0, target_len- len(target)))
        else:
            target = target[:target_len]
        

        x=torch.FloatTensor(np.array(x))
        edge_index=torch.LongTensor(edge_index).transpose(1, 0)
        target=torch.LongTensor(np.array([target]))
        
        drug_fp = torch.FloatTensor(np.array([drug_fp]))
        prot_desc = torch.FloatTensor(np.array([prot_desc]))
        
        kg_x = torch.FloatTensor(np.array([self.kg_enc[index]]))
        text_x = self.text_enc[index]

        if self.task == 'classification':
            y = torch.tensor(self.labels[index], dtype=torch.float)
        elif self.task == 'regression':
            y = torch.tensor(self.labels[index] * self.conversion, dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, target=target,  y=y, drug_fp=drug_fp, prot_desc=prot_desc, kg=kg_x, text=text_x)     
        return data

    def __len__(self):
        return len(self.drug_id)

    def get_finger(self, smile):
        mols =Chem.MolFromSmiles(smile)
        fp = AllChem.GetMorganFingerprintAsBitVect(mols,2,nBits=1024,)
        fp = list(fp)
        return fp

    def get_ctd(self, seq):
        ctd = CTD.CalculateCTD(seq).values()
        protein_descriptor = list(ctd)
        return protein_descriptor
    
    def cluster(self, mat, threshold):
        n = len(mat)
        e = []
        f = UFS(n)
        for i in range(n):
            for j in range(n):
                x, y = f.find(i), f.find(j)
                if x != y and mat[x][y] > threshold:
                    f.merge(x, y)
                    for k in range(n):
                        mat[y][k] = min(mat[y][k], mat[x][k])
        clusters = [[] for i in range(n)]
        for i in range(n):
            clusters[f.find(i)].append(i)
        merged_clusters = [[] for i in range(3)]
        perm = np.random.permutation(n)
        cur = 0
        for i in perm:
            if len(clusters[i]) > 0:
                print(len(clusters[i]))
            merged_clusters[cur].extend(clusters[i])
            if cur < 2 and len(merged_clusters[cur]) >= n // 3:
                cur += 1
        for i in range(3):
            print(len(merged_clusters[i]))
        return merged_clusters

    def standard_split_08(self, path):

        df1 = pd.read_csv(osp.join(path, 'data_folds/warm_start_1_1/test_fold_4.csv'))
        df2 = pd.read_csv(osp.join(path, 'data_folds/warm_start_1_1/train_fold_4.csv'))
        df_all = pd.concat([df1,df2],axis=0,ignore_index=True)  #将df2数据与df1合并

        df_all = df_all.drop_duplicates()   #去重
        df_all = df_all.reset_index(drop=True) #重新生成index
        df_all.to_csv(self.standard_fold_path + '/' + 'data.csv') #将结果保存为新的csv文件

    def gen_08_cold(self, path, split):
        # path = "/home/hk/Open_DAIR_BioMed/origin-data/yamanishi_08/standard_split/data.csv"
        drug_ids = np.loadtxt(path, dtype=str, skiprows=1, usecols=(1), comments="!", delimiter=',')
        target_ids = np.loadtxt(path, dtype=str, skiprows=1, usecols=(3), comments="!", delimiter=',')
        label = np.loadtxt(path, dtype=str, skiprows=1, usecols=(4), comments="!", delimiter=',')
        drugid2num = dict(zip(self.drug_ids, range(len(self.drug_ids))))
        targids2num = dict(zip(self.target_ids, range(len(self.target_ids))))

        if split == 'cold_drug':
            perm = np.random.permutation(len(self.drug_ids))
            drug_fold = [[] for i in range(5)]
            for i, x in enumerate(perm):
                drug_fold[i % 5].append(x)
            for fold in range(5):
                cold_drug_train = []
                cold_drug_test = []
                for i in range(len(drug_ids)):
                    triple = [drug_ids[i], target_ids[i], label[i]]
                    if drugid2num[drug_ids[i]] in drug_fold[fold]:
                    # if drugid2num[i] in drug_fold[fold]:
                        cold_drug_test.append(triple)
                    else:
                        cold_drug_train.append(triple)
                print("cold drug fold ", fold, "train samples:", len(cold_drug_train), "test_samples:", len(cold_drug_test))
                with open(self.cold_path + '/' +'train_fold' + str(fold) + ".csv", "w") as f:
                    for x in cold_drug_train:
                        f.write(",".join(map(str, x)) + "\n")
                with open(self.cold_path + '/' +'test_fold' + str(fold) + ".csv", "w") as f:
                    for x in cold_drug_test:
                        f.write(",".join(map(str, x)) + "\n")

        # cold_protein
        if split == 'cold_protein':
            perm = np.random.permutation(len(self.target_ids))
            prot_fold = [[] for i in range(5)]
            for i, x in enumerate(perm):
                prot_fold[i % 5].append(x)
            for fold in range(5):
                cold_prot_train = []
                cold_prot_test = []
                for i in range(len(drug_ids)):
                    triple = [drug_ids[i], target_ids[i], label[i]]
                    if targids2num[target_ids[i]] in prot_fold[fold]:
                        cold_prot_test.append(triple)
                    else:
                        cold_prot_train.append(triple)
                print("cold protein fold ", fold, "train samples:", len(cold_prot_train), "test_samples:", len(cold_prot_test))
                with open(self.cold_path + '/' +'train_fold' + str(fold) + ".csv", "w") as f:
                    for x in cold_prot_train:
                        f.write(",".join(map(str, x)) + "\n")
                with open(self.cold_path + '/' +'test_fold' + str(fold) + ".csv", "w") as f:
                    for x in cold_prot_test:
                        f.write(",".join(map(str, x)) + "\n")

        # cold_cluster
        if split == 'cold_cluster':
            fp = self.fp
            drug_sim = np.zeros((len(fp), len(fp)))
            for i in tqdm(range(len(fp))):
                for j in range(len(fp)):
                    drug_sim[i][j] = jaccard_score(fp[i], fp[j])
            """
            import swalign
            scoring = swalign.NucleotideScoringMatrix(2, -1)
            sw = swalign.LocalAlignment(scoring)
            score = np.zeros(len(seq))
            for i in tqdm(range(len(seq))):
                score[i] = sw.align(seq[i], seq[i]).score
            prot_sim = np.zeros((len(seq), len(seq)))
            for i in tqdm(range(len(seq))):
                for j in range(len(seq)):
                    prot_sim[i][j] = sw.align(seq[i], seq[j]).score / np.sqrt(score[i]) / np.sqrt(score[j])
            """
            pt = self.ctd
            mean = np.mean(pt, axis=0)
            var = np.var(pt, axis=0)
            pt = (pt - mean) / np.sqrt(var)
            prot_sim = np.zeros((len(self.target_ids), len(self.target_ids)))
            for i in tqdm(range(len(self.target_ids))):
                for j in range(len(self.target_ids)):
                    prot_sim[i][j] = np.dot(pt[i], pt[j]) / np.sqrt(np.dot(pt[i], pt[i])) / np.sqrt(np.dot(pt[j], pt[j]))
            drug_cluster = self.cluster(drug_sim, 0.2)
            prot_cluster = self.cluster(prot_sim, 0.3)
            for c1 in range(3):
                for c2 in range(3):
                    cold_cluster_train = []
                    cold_cluster_test = []
                    for i in range(len(drug_ids)):
                        triple = [drug_ids[i], target_ids[i], label[i]]
                        if drugid2num[drug_ids[i]] in drug_cluster[c1] and targids2num[target_ids[i]] in prot_cluster[c1]:
                            cold_cluster_test.append(triple)
                        elif drugid2num[drug_ids[i]] not in drug_cluster[c1] and targids2num[target_ids[i]] not in drug_cluster[c2]:
                            cold_cluster_train.append(triple)
                    fold = c1 * 3 + c2
                    print("cluster fold ", fold, "train samples:", len(cold_cluster_train), "test_samples:", len(cold_cluster_test))
                    with open(self.cold_path + '/' +'train_fold' + str(fold) + ".csv", "w") as f:
                        for x in cold_cluster_train:
                            f.write(",".join(map(str, x)) + "\n")
                    with open(self.cold_path + '/' +'test_fold' + str(fold) + ".csv", "w") as f:
                        for x in cold_cluster_test:
                            f.write(",".join(map(str, x)) + "\n")

    def can_smiles(self, smiles):
        smiles=smiles.tolist()
        cnt =0
        can_smiles = []
        for origin_sm in smiles:
            try:
                mol = Chem.MolFromSmiles(origin_sm)
                canonical_smi_PUBCHEM = Chem.MolToSmiles(mol)
                can_smiles.append(canonical_smi_PUBCHEM)
            except:
                can_smiles.append(origin_sm)
                cnt+=1
                print(f'{cnt} smiles have no mol')
        return np.array(smiles)

    def load_smiles(self, path):
        drug_id = np.loadtxt(path, dtype=str, skiprows=1, usecols=(0), comments="!", delimiter=',')
        smiles = np.loadtxt(path, dtype=str, skiprows=1, usecols=(1), comments="!", delimiter=',')
        smiles = self.can_smiles(smiles)
        return drug_id, smiles

    def load_protein_seq(self, path, tokenizer="cnn"):
        target_id = np.loadtxt(path, dtype=str, skiprows=1, usecols=(0), delimiter=',')
        target_seq = np.loadtxt(path, dtype=str, skiprows=1, usecols=(2), delimiter=',')
        target_seq_enc = []
        for elem in target_seq:
            target_seq_enc.append(tok_protein(elem, tokenizer))
        return target_id, np.array(target_seq_enc), dict(zip(target_seq, target_id)), dict(zip(target_id, target_seq))

    def load_data(self, path):
        drug_ids = np.loadtxt(path, dtype=str, skiprows=1, usecols=(0), comments="!", delimiter=',')
        target_ids = np.loadtxt(path, dtype=str, skiprows=1, usecols=(1), comments="!", delimiter=',')
        label = np.loadtxt(path, dtype=str, skiprows=1, usecols=(2), comments="!", delimiter=',')
        # print("d id=", drug_ids[0])
        # print("p id=", target_ids[0])
        label = np.array([float(x) for x in label])
        return drug_ids, target_ids, label

    def load_cold_data(self, path):
        cur_path = osp.join(path, self.cold, self.csv_name + ".csv")

        drug_ids = np.loadtxt(cur_path, dtype=str, skiprows=1, usecols=(0), comments="!", delimiter=',')
        target_ids = np.loadtxt(cur_path, dtype=str, skiprows=1, usecols=(1), comments="!", delimiter=',')
        label = np.loadtxt(cur_path, dtype=str, skiprows=1, usecols=(2), comments="!", delimiter=',')
        # print("d id=", drug_ids[0])
        # print("p id=", target_ids[0])
        label = np.array([float(x) for x in label])
        return drug_ids, target_ids, label


import argparse
import logging
logger = logging.getLogger(__name__)
def add_argument(parser):
    parser.add_argument("--dataset", type=str, default="yamanishi08")
    parser.add_argument("--fold", type=str, default="warm_0")
    parser.add_argument("--config-path", type=str, default="../configs/dti_base.json")
    parser.add_argument("--load-path", type=str, default="")
    parser.add_argument("--load-encoder-path", type=str, default="")
    parser.add_argument("--load-encoder-step", type=str, default="")
    parser.add_argument("--save-path", type=str, default="")
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    # Training parameters
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--plm-lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=20)
    parser.add_argument("--save-epochs", type=int, default=50)
    parser.add_argument("--device", type=int, nargs='+', default=[0])
    # Pre-train parameters
    parser.add_argument("--kg-path", type=str, default="")
    parser.add_argument("--subgraph", action="store_true")
    parser.add_argument("--pretrain-steps", type=int, default=10000)
    parser.add_argument("--pretrain-save-steps", type=int, default=5000)
    parser.add_argument("--alternate-steps", type=int, default=4)
    parser.add_argument("--pretrain-lr", type=float, default=3e-4)
    parser.add_argument("--pretrain-dataset-path", type=str, default="")
    return parser
if __name__ == "__main__":  
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(description='KMP DTI')
    add_argument(parser)
    args = parser.parse_args()
    config = json.load(open(args.config_path, "r"))
    fold = config['data']['type']
    
    folds = {"warm": 5, "colddrug": 5, "coldprotein": 5, "coldcluster": 9}
    for fold in folds:
        for i in range(folds[fold]):
            train_csv = "train_fold" + str(i)
            test_csv = "test_fold" +str(i)
            task = "classification"
            
            logger.info("Loading dataset %s" % (args.dataset))
            
            train_dataset = Yamanishi08(config, train_csv)
            test_dataset = Yamanishi08(config, test_csv)
            dataloader = DataLoader(train_dataset, batch_size=args.batch_size * len(args.device), shuffle=True, num_workers=8)
            for data in tqdm(dataloader):
                print(data)
                print(data.y)
                print(data.y.long())
                break
            dataloader = DataLoader(test_dataset, batch_size=args.batch_size * len(args.device), shuffle=True, num_workers=8)
            for data in tqdm(dataloader):
                print(data)
                print(data.y)
                print(data.y.long())
                break
            
            logger.info("Building Model...")
