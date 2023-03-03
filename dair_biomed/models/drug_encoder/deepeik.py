import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

from models.drug_encoder.cnn import CNN
from models.drug_encoder.momu_gnn import MoMuGNN
from models.drug_encoder.pyg_gnn import PygGNN
from models.drug_encoder.molclr_gnn import GINet

SUPPORTED_DRUG_ENCODER = {
    "cnn": CNN,
    "graphcl": MoMuGNN,
    "molclr": GINet,
    "graphmvp": PygGNN,
}

class DrugDeepEIK(nn.Module):
    def __init__(self, config):
        super(DrugDeepEIK, self).__init__()
        self.output_dim = config["projection_dim"]

        self.structure_encoder = SUPPORTED_DRUG_ENCODER[config["structure"]["name"]](**config["structure"])
        if "init_checkpoint" in config["structure"]:
            self.structure_encoder.load_state_dict(torch.load(config["structure"]["init_checkpoint"]))
        self.structure_dropout = nn.Dropout(config["structure"]["dropout"])
        self.structure_proj = nn.Linear(config["structure"]["emb_dim"], config["projection_dim"])

        if "text" in config:
            if "model_name_or_path" in config["text"]:
                self.text_encoder = BertModel.from_pretrained(config["text"]["model_name_or_path"])
            elif "config_name_or_path" in config["text"]:
                bert_config = BertConfig.from_json_file(config["text"]["config_name_or_path"])
                self.text_encoder = BertModel(bert_config)
            if "init_checkpoint" in config["text"]:
                ckpt = torch.load(config["text"]["init_checkpoint"])
                processed_ckpt = {}
                if 'module.ptmodel.bert.embeddings.word_embeddings.weight' in ckpt:
                    for k, v in ckpt.items():
                        if k.startswith("module.ptmodel.bert."):
                            processed_ckpt[k[20:]] = v
                        else:
                            processed_ckpt[k] = v
                elif 'bert.embeddings.word_embeddings.weight':
                    for k, v in ckpt.items():
                        if k.startswith("bert."):
                            processed_ckpt[k[5:]] = v
                        else:
                            processed_ckpt[k] = v
                else:
                    processed_ckpt = ckpt
                
                missing_keys, unexpected_keys = self.text_encoder.load_state_dict(processed_ckpt, strict=False)
                logger.info("missing_keys: %s" % " ".join(missing_keys))
                logger.info("unexpected_keys: %s" % " ".join(unexpected_keys))
            self.text_dropout = nn.Dropout(config["text"]["dropout"])
            self.text_proj = nn.Linear(config["text"]["hidden_dim"], config["projection_dim"])
            self.output_dim += config["projection_dim"]
        # TODO: configure kg encoder

    def forward(self, drug):
        # TODO: implement fusion with attention
        h, _ = self.structure_encoder(drug["structure"])
        return h, _

    def encode_structure(self, structure):
        h, _ = self.structure_encoder(structure)
        h = self.structure_dropout(h)
        return self.structure_proj(h)

    def encode_text(self, text):
        h = self.text_encoder(**text)["pooler_output"]
        h = self.text_dropout(h)
        return self.text_proj(h)

    def encode_knowledge(self, kg):
        pass