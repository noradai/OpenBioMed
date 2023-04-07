import torch
import torch.nn as nn

from transformers.modeling_outputs import BaseModelOutput

from models.drug_decoder.molt5 import MolT5
from models.drug_encoder import MoMu, MolALBEF

SUPPORTED_DRUG_ENCODER = {
    "MoMu": MoMu,
    "MolALBEF": MolALBEF
}

class GraphEnhancedMolCapModel(nn.Module):
    def __init__(self, config):
        super(GraphEnhancedMolCapModel, self).__init__()
        self.generate_model = MolT5(config["text"])
        self.use_graph = "graph" in config
        self.use_node_embeds = self.use_graph and config["graph"]["max_n_nodes"] > 0
        #self.use_node_embeds = False
        if self.use_graph:
            self.graph_encoder = SUPPORTED_DRUG_ENCODER[config["graph"]["name"]](config["graph"])
            if "init_checkpoint" in config["graph"]:
                ckpt = torch.load(config["graph"]["init_checkpoint"])
                if "param_key" in config["graph"]:
                    ckpt = ckpt[config["graph"]["param_key"]]
                self.graph_encoder.load_state_dict(ckpt)
            if config["graph"]["stop_grad"]:
                for k, v in self.graph_encoder.named_parameters():
                    v.requires_grad = False
            self.graph_projector = nn.Sequential(
                nn.Linear(config["graph"]["output_dim"], self.generate_model.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.generate_model.hidden_size, self.generate_model.hidden_size)
            )

    def forward(self, mol):
        h, encoder_attention_mask = self._combined_encodings(mol)
        labels = mol["text"]["input_ids"].masked_fill(~mol["text"]["attention_mask"].bool(), -100)
        return self.generate_model(
            encoder_outputs=h,
            encoder_attention_mask=encoder_attention_mask,
            decoder_attention_mask=mol["text"]["attention_mask"],
            labels=labels
        )

    def decode(self, mol, num_beams, max_length):
        h, encoder_attention_mask = self._combined_encodings(mol)
        return self.generate_model.decode(
            encoder_outputs=h,
            encoder_attention_mask=encoder_attention_mask,
            num_beams=num_beams,
            max_length=max_length
        )

    def _combined_encodings(self, mol):
        B, _ = mol["structure"]["SMILES"]["attention_mask"].shape
        device = mol["structure"]["SMILES"]["attention_mask"].device
        smi_feats = self.generate_model.encode(mol["structure"]["SMILES"])
        if self.use_graph:
            if self.use_node_embeds:
                graph_feats, node_feats, node_attention_mask = self.graph_encoder.encode_structure(mol["structure"]["graph"], proj=False, return_node_feats=True)
                graph_feats = self.graph_projector(graph_feats)
                node_feats = self.graph_projector(node_feats)
                h = BaseModelOutput(
                    last_hidden_state=torch.cat([graph_feats.unsqueeze(1), node_feats, smi_feats], dim=1),
                    #last_hidden_state=torch.cat([graph_feats.unsqueeze(1), node_feats], dim=1),
                    hidden_states=None,
                    attentions=None
                )
                encoder_attention_mask = torch.cat([torch.ones(B, 1).to(device), node_attention_mask, mol["structure"]["SMILES"]["attention_mask"]], dim=1)
            else:
                if "additional_text" in mol:
                    graph_feats = self.graph_encoder(mol["structure"]["graph"], mol["additional_text"])["last_hidden_state"][:, 0, :]
                else:
                    graph_feats = self.graph_encoder.encode_structure(mol["structure"]["graph"], proj=False)
                graph_feats = self.graph_projector(graph_feats)
                h = BaseModelOutput(
                    last_hidden_state=torch.cat([graph_feats.unsqueeze(1), smi_feats], dim=1),
                    hidden_states=None,
                    attentions=None
                )
                encoder_attention_mask = torch.cat([torch.ones(B, 1).to(device), mol["structure"]["SMILES"]["attention_mask"]], dim=1)
        else:
            h = BaseModelOutput(
                last_hidden_state=smi_feats,
                hidden_states=None,
                attentions=None
            )
            encoder_attention_mask = mol["structure"]["SMILES"]["attention_mask"]
        return h, encoder_attention_mask