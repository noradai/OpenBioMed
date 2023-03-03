import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn

from transformers import BertConfig, BertModel

class KVPLM(nn.Module):
    def __init__(self, config):
        super(KVPLM, self).__init__()

        bert_config = BertConfig.from_json_file(config["bert_config_path"])
        self.text_encoder = BertModel(bert_config)
        ckpt = torch.load(config["checkpoint_path"])
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

        self.dropout = nn.Dropout(config["dropout"])
        
    def forward(self, drug):
        return self.encode_structure(drug["strcture"]), self.encode_text(drug["text"])

    def encode_structure(self, structure):
        h = self.text_encoder(**structure)["pooler_output"]
        return self.dropout(h)

    def encode_text(self, text):
        h = self.text_encoder(**text)["pooler_output"]
        return self.dropout(h)