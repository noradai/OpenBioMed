import torch
import torch.nn as nn
import torch.nn.functional as F

from models.drug_encoder.pyg_gnn import PygGNN
from models.text_encoder.xbert import BertConfig, BertForMaskedLM

class MolALBEF(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_n_nodes = config["max_n_nodes"]

        self.graph_encoder = PygGNN(
            num_layer=config["gin_num_layers"],
            emb_dim=config["gin_hidden_dim"],
            gnn_type="gin",
            drop_ratio=config["drop_ratio"],
            JK="last",
        )
        if "gin_ckpt" in config:
            self.graph_encoder.load_state_dict(torch.load(config["gin_ckpt"]))
        self.graph_proj_head = nn.Linear(config["gin_hidden_dim"], config["projection_dim"])
        
        bert_config = BertConfig.from_json_file(config["bert_config_path"])
        self.text_encoder = BertForMaskedLM(bert_config)
        if "bert_ckpt" in config:
            ckpt = torch.load(config["bert_ckpt"])
            processed_ckpt = {}
            if 'module.ptmodel.bert.embeddings.word_embeddings.weight' in ckpt:
                for k, v in ckpt.items():
                    if k.startswith("module.ptmodel."):
                        processed_ckpt[k[15:]] = v
                    else:
                        processed_ckpt[k] = v
            missing_keys, unexpected_keys = self.text_encoder.load_state_dict(processed_ckpt, strict=False)
            print("missing keys:", missing_keys)
            print("unexpected keys:", unexpected_keys)
        self.text_proj_head = nn.Linear(bert_config.hidden_size, config["projection_dim"])

        self.graph_linear = nn.Linear(config["gin_hidden_dim"], bert_config.hidden_size)
        self.mtm_head = nn.Linear(bert_config.hidden_size, 2)

        self.temperature = 0.1
        self.device = "cuda:0"

    def forward(self, mol, text):
        mol_embeds, node_embeds = self.graph_encoder(mol)
        mol_feats = F.normalize(self.graph_proj_head(mol_embeds), dim=-1)
        all_node_feats = self.graph_linear(node_embeds)
        # serialize node feature
        batch_size = mol_feats.shape[0]
        node_feats = []
        node_attention_mask = []
        for i in range(batch_size):
            feat = all_node_feats[torch.where(mol.batch == i)]
            if feat.shape[0] < self.max_n_nodes:
                node_feats.append(torch.cat((
                    feat,
                    torch.zeros(self.max_n_nodes - feat.shape[0], feat.shape[1]).to(feat.device)
                ), dim=0))
                node_attention_mask.append(torch.cat((
                    torch.ones(feat.shape[0]).to(feat.device), 
                    torch.zeros(self.max_n_nodes - feat.shape[0]).to(feat.device)
                ), dim=0))
            else:
                node_feats.append(feat[:self.max_n_nodes, :])
                node_attention_mask.append(torch.ones(self.max_n_nodes).to(feat.device))
        node_feats = torch.stack(node_feats, dim=0)
        node_attention_mask = torch.stack(node_attention_mask, dim=0)

        text_outputs = self.text_encoder.bert(text["input_ids"], attention_mask=text["attention_mask"], mode='text', return_dict=True)
        seq_feats = text_outputs["last_hidden_state"]
        text_feats = F.normalize(self.text_proj_head(seq_feats[:, 0, :]), dim=-1)
        
        output = self.text_encoder.bert(
            encoder_embeds=seq_feats,
            attention_mask=text["attention_mask"],
            encoder_hidden_states=node_feats,
            encoder_attention_mask=node_attention_mask,
            mode='fusion',
            return_dict=True
        )
        return output

    def encode_structure(self, structure):
        drug_embeds, _ = self.graph_encoder(structure)
        return self.graph_proj_head(drug_embeds)

    def encode_text(self, text):
        text_embeds = self.text_encoder.bert(text["input_ids"], attention_mask=text["attention_mask"], mode='text', return_dict=True)["last_hidden_state"]
        return self.text_proj_head(text_embeds[:, 0, :])