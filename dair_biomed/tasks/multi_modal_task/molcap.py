import logging
logger = logging.getLogger(__name__)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from transformers.modeling_outputs import BaseModelOutput

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

from datasets.molcap_dataset import SUPPORTED_MOLCAP_DATASET
from models.drug_encoder import MoMu, MolALBEF, Text2MolMLP
from models.drug_decoder.molt5 import MolT5
from utils import AverageMeter, EarlyStopping, ToDevice, DrugCollator

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

def train_molcap(train_loader, val_loader, model, args, device):
    requires_grad = []
    for k, v in model.named_parameters():
        if v.requires_grad:
            requires_grad.append(k)
    logger.info("parameters requires grad: %s" % (" ".join(requires_grad)))

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    #stopper = EarlyStopping(mode='lower', patience=args.patience, filename=args.output_path)

    running_loss = AverageMeter()
    step = 0
    for epoch in range(args.epochs):
        logger.info("========Epoch %d========" % (epoch + 1))
        logger.info("Training...")
        #model.train()

        for mol in tqdm(train_loader):
            mol = ToDevice(mol, device)
            loss = model(mol)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.update(loss.detach().cpu().item())
            step += 1
            if step % args.logging_steps == 0:
                logger.info("Steps=%d Training Loss=%.4lf" % (step, running_loss.get_average()))
                running_loss.reset()
        val_molcap(val_loader, model, device)
        #if stopper.step(val_molcap(val_loader, model, device), model):
        #    break
        if epoch % 20 == 0:
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(args.output_path, "checkpoint_" + str(epoch) + ".pth"))
    #model.load_state_dict(torch.load(args.output_path)["model_state_dict"])
    return model

def val_molcap(val_loader, model, device):
    model.eval()
    val_loss = 0

    logger.info("Validating...")
    for mol in tqdm(val_loader):
        mol = ToDevice(mol, device)
        loss = model(mol)
        val_loss += loss.detach().cpu().item()
    logger.info("validation loss %.4lf" % (val_loss / len(val_loader)))
    return val_loss / len(val_loader)

def test_molcap(test_dataset, test_loader, model, args, device):
    model.eval()
    outputs = []
    gts = test_dataset.texts

    logger.info("Testing...")
    for i, mol in enumerate(tqdm(test_loader)):
        mol = ToDevice(mol, device)
        output = model.decode(mol, num_beams=5, max_length=512)
        outputs += output
        if i <= 3:
            for j in range(5):
                logger.info("Generated: %s" % outputs[-j])
                logger.info("Ground truth: %s" % gts[len(outputs) - j])
                logger.info("------------------------------------------------------")

    tokenizer = BertTokenizerFast.from_pretrained(args.text2mol_bert_path)
    output_tokens = []
    gt_tokens = []
    meteor_scores = []
    rouge_scores = []
    text2mol_scores = []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    text2mol = Text2MolMLP(
        ninp=768, 
        nhid=600, 
        nout=300, 
        model_name_or_path=args.text2mol_bert_path, 
        cid2smiles_path=os.path.join(args.text2mol_data_path, "cid_to_smiles.pkl"),
        cid2vec_path=os.path.join(args.text2mol_data_path, "test.txt")
    )
    text2mol.load_state_dict(torch.load(args.text2mol_ckpt_path))
    device = torch.device(args.device)
    text2mol.to(device)
    with open(args.caption_save_path, "w") as f:
        f.write("SMILES\tground truth\toutput\n")
        for i in range(len(outputs)):
            output_tokens.append(tokenizer.tokenize(outputs[i], truncation=True, max_length=512, padding='max_length'))
            output_tokens[i] = list(filter(('[PAD]').__ne__, output_tokens[i]))
            output_tokens[i] = list(filter(('[CLS]').__ne__, output_tokens[i]))
            output_tokens[i] = list(filter(('[SEP]').__ne__, output_tokens[i]))

            gt_tokens.append(tokenizer.tokenize(gts[i], truncation=True, max_length=512, padding='max_length'))
            gt_tokens[i] = list(filter(('[PAD]').__ne__, gt_tokens[i]))
            gt_tokens[i] = list(filter(('[CLS]').__ne__, gt_tokens[i]))
            gt_tokens[i] = [list(filter(('[SEP]').__ne__, gt_tokens[i]))]

            meteor_scores.append(meteor_score(gt_tokens[i], output_tokens[i]))
            rouge_scores.append(scorer.score(outputs[i], gts[i]))
            text2mol_scores.append(text2mol(test_dataset.smiles[i], outputs[i], device).detach().cpu().item())
            f.write(test_dataset.smiles[i] + '\t' + gts[i] + '\t' + outputs[i] + '\n')
    bleu2 = corpus_bleu(gt_tokens, output_tokens, weights=(0.5, 0.5))
    bleu4 = corpus_bleu(gt_tokens, output_tokens, weights=(0.25, 0.25, 0.25, 0.25))

    return {
        "BLEU-2": bleu2,
        "BLEU-4": bleu4,
        "Meteor": np.mean(meteor_scores),
        "ROUGE-1": np.mean([rs['rouge1'].fmeasure for rs in rouge_scores]),
        "ROUGE-2": np.mean([rs['rouge2'].fmeasure for rs in rouge_scores]),
        "ROUGE-L": np.mean([rs['rougeL'].fmeasure for rs in rouge_scores]),
        "Text2Mol": np.mean(text2mol_scores)
    }

def add_arguments(parser):
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--config_path", type=str, default="")
    parser.add_argument('--dataset', type=str, default='chebi-20')
    parser.add_argument("--dataset_path", type=str, default='../datasets/molcap/chebi-20')
    parser.add_argument("--output_path", type=str, default="../ckpts/finetune_ckpts/caption.pth")
    parser.add_argument("--caption_save_path", type=str, default="../assets/outputs.txt")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--logging_steps", type=int, default=300)
    parser.add_argument("--text2mol_bert_path", type=str, default="")
    parser.add_argument("--text2mol_data_path", type=str, default="")
    parser.add_argument("--text2mol_ckpt_path", type=str, default="")

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    config = json.load(open(args.config_path))

    device = torch.device(args.device)

    # load dataset
    train_dataset = SUPPORTED_MOLCAP_DATASET[args.dataset](args.dataset_path, config["data"]["drug"], split="train")
    val_dataset = SUPPORTED_MOLCAP_DATASET[args.dataset](args.dataset_path, config["data"]["drug"], split="validation")
    test_dataset = SUPPORTED_MOLCAP_DATASET[args.dataset](args.dataset_path, config["data"]["drug"], split="test")
    collator = DrugCollator(config["data"]["drug"])
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=collator, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, args.batch_size, shuffle=False, collate_fn=collator, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=collator, num_workers=args.num_workers)

    # load model
    model = GraphEnhancedMolCapModel(config["network"])
    model = model.to(device)

    if args.mode == "train":
        train_molcap(train_dataloader, val_dataloader, model, args, device)
    elif args.mode == "test":
        if os.path.exists(args.output_path):
            """
            state_dict = torch.load(args.output_path, map_location=device)["model_state_dict"]
            model.load_state_dict(state_dict)
            """
            state_dict = torch.load("/root/MoleculeCaption/saved_models/gint5_smiles2caption_small.pt", map_location=device)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("graph_projector"):
                    new_state_dict[k.lstrip("graph_projector.")] = v
            model.graph_projector.load_state_dict(new_state_dict)
        results = test_molcap(test_dataset, test_dataloader, model, args, device)
        print(results)
    elif args.mode == "traintest":
        train_molcap(train_dataloader, val_dataloader, model, args, device)
        results = test_molcap(test_dataset, test_dataloader, model, args, device)
        print(results)
