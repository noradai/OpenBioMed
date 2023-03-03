import logging
logger = logging.getLogger(__name__)

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
import math
from tqdm import tqdm

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.optimizer import required
from torch.nn.utils import clip_grad_norm_

from utils import EarlyStopping, AverageMeter, DrugCollator, ToDevice
from datasets.mtr_dataset import SUPPORTED_MTR_DATASETS
from models.drug_encoder import KVPLM, MoMu, MolALBEF, DrugBERT, DrugDeepEIK

SUPPORTED_MTR_MODEL = {
    "SciBERT": DrugBERT,
    "KV-PLM": KVPLM, 
    "KV-PLM*": KVPLM,
    "MoMu": MoMu, 
    "MolALBEF": MolALBEF,
    "DeepEIK": DrugDeepEIK
}

def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))

def warmup_constant(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return max((x - 1. )/ (warmup - 1.), 0.)
    
def warmup_poly(x, warmup=0.002, degree=0.5):
    if x < warmup:
        return x/warmup
    return (1.0 - x)**degree


SCHEDULES = {
    'warmup_cosine':warmup_cosine,
    'warmup_constant':warmup_constant,
    'warmup_linear':warmup_linear,
    'warmup_poly':warmup_poly,
}

class BertAdam(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix.
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """
    def __init__(self, params, lr=required, warmup=-1, t_total=-1, schedule='warmup_linear',
                 b1=0.9, b2=0.999, e=1e-6, weight_decay=0.01,
                 max_grad_norm=1.0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= b1 < 1.0:
            raise ValueError("Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError("Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
        if not e >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        b1=b1, b2=b2, e=e, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        super(BertAdam, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group['e'])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay'] > 0.0:
                    update += group['weight_decay'] * p.data

                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                state['step'] += 1

        return loss

def contrastive_loss(logits_des, logits_smi, margin, device):
    scores = torch.cosine_similarity(logits_smi.unsqueeze(1).expand(logits_smi.shape[0], logits_smi.shape[0], logits_smi.shape[1]), logits_des.unsqueeze(0).expand(logits_des.shape[0], logits_des.shape[0], logits_des.shape[1]), dim=-1)
    diagonal = scores.diag().view(logits_smi.size(0), 1)
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)
    
    cost_des = (margin + scores - d1).clamp(min=0)
    cost_smi = (margin + scores - d2).clamp(min=0)

    # clear diagonals
    mask = torch.eye(scores.size(0)) > .5
    I = Variable(mask)
    if torch.cuda.is_available():
        I = I.to(device)
    cost_des = cost_des.masked_fill_(I, 0)
    cost_smi = cost_smi.masked_fill_(I, 0)

    # keep the maximum violating negative for each query
    #if self.max_violation:
    cost_des = cost_des.max(1)[0]
    cost_smi = cost_smi.max(0)[0]

    return cost_des.sum() + cost_smi.sum()

def train_mtr(train_loader, val_loader, model, args):
    loss_fn = contrastive_loss
    params = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01
    },{
        'params': [p for n, p in params if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0
    }]
    #optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    optimizer = BertAdam(
        optimizer_grouped_parameters,
        weight_decay=0,
        lr=args.lr,
        warmup=args.warmup,
        t_total=len(train_loader) * args.epochs,
    )
    stopper = EarlyStopping(mode="higher", patience=args.patience, filename=args.output_path)

    running_loss = AverageMeter()
    for epoch in range(args.epochs):
        logger.info("========Epoch %d========" % (epoch + 1))
        logger.info("Training...")
        model.train()
        running_loss.reset()

        step = 0
        for drug in tqdm(train_loader):
            drug = ToDevice(drug, args.device)
            #print(drug["structure"], drug["text"])
            drug_rep = model.encode_structure(drug["structure"])
            text_rep = model.encode_text(drug["text"])
            loss = loss_fn(drug_rep, text_rep, margin=args.margin, device=args.device)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss.update(loss.detach().cpu().item())
            step += 1
            if step % args.log_every == 0:
                logger.info("Steps=%d Training Loss=%.4lf" % (step, running_loss.get_average()))
                running_loss.reset()

        val_metrics = val_mtr(val_loader, model, args)
        logger.info(", ".join(["val %s: %.4lf" % (k, val_metrics[k]) for k in val_metrics]))
        if stopper.step((val_metrics["acc@1_d2t"] + val_metrics["acc@1_t2d"]), model):
            break
    model.load_state_dict(torch.load(args.output_path)["model_state_dict"])
    return model

def val_mtr(val_loader, model, args):
    model.eval()
    drug_rep_total, text_rep_total = [], []
    acc_d2t, acc_t2d, n_samples = 0, 0, 0
    for drug in tqdm(val_loader):
        drug = ToDevice(drug, args.device)

        drug_rep = model.encode_structure(drug["structure"])
        text_rep = model.encode_text(drug["text"])
        drug_rep_total.append(drug_rep.detach().cpu())
        text_rep_total.append(text_rep.detach().cpu())

        # calculate #1 acc
        scores = torch.cosine_similarity(drug_rep.unsqueeze(1).expand(drug_rep.shape[0], drug_rep.shape[0], drug_rep.shape[1]), text_rep.unsqueeze(0).expand(text_rep.shape[0], text_rep.shape[0], text_rep.shape[1]), dim=-1)
        mx_d2t = torch.argmax(scores, axis=1)
        mx_t2d = torch.argmax(scores, axis=0)
        acc_d2t += sum((mx_d2t == torch.arange(mx_d2t.shape[0]).to(args.device)).int()).item()
        acc_t2d += sum((mx_t2d == torch.arange(mx_t2d.shape[0]).to(args.device)).int()).item()
        n_samples += mx_d2t.shape[0]

    drug_rep = torch.cat(drug_rep_total, dim=0)
    text_rep = torch.cat(text_rep_total, dim=0)
    score = torch.zeros(n_samples, n_samples)
    rec_d2t, rec_t2d = 0, 0
    for i in range(n_samples):
        score[i] = torch.cosine_similarity(drug_rep[i], text_rep)
    for i in range(n_samples):
        _, idx = torch.sort(score[i, :], descending=True)
        for j in range(min(n_samples, 20)):
            if idx[j] == i:
                rec_d2t += 1
        _, idx = torch.sort(score[:, i], descending=True)
        for j in range(min(n_samples, 20)):
            if idx[j] == i:
                rec_t2d += 1

    result = {
        "acc@1_d2t": acc_d2t / n_samples,
        "acc@1_t2d": acc_t2d / n_samples,
        "rec@20_d2t": rec_d2t / n_samples,
        "rec@20_t2d": rec_t2d / n_samples,
    }
    return result

def add_arguments(parser):
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--mode", type=str, default="zero_shot")
    parser.add_argument("--config_path", type=str, default="")
    parser.add_argument('--dataset', type=str, default='PCdes')
    parser.add_argument("--dataset_path", type=str, default='../datasets/mtr/PCdes/')
    parser.add_argument("--init_checkpoint", type=str, default="")
    parser.add_argument("--output_path", type=str, default="../ckpts/finetune_ckpts/finetune.pth")
    parser.add_argument("--param_key", type=str, default="state_dict")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup", type=float, default=0.03)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--margin", type=float, default=0.2)

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    config = json.load(open(args.config_path, "r"))

    dataset = SUPPORTED_MTR_DATASETS[args.dataset](args.dataset_path, config["data"])
    train_dataset = dataset.index_select(dataset.train_index)
    val_dataset = dataset.index_select(dataset.val_index)
    test_dataset = dataset.index_select(dataset.test_index)
    collator = DrugCollator(config["data"]["drug"])
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collator)

    model = SUPPORTED_MTR_MODEL[config["model"]](config["network"])
    if args.init_checkpoint != "":
        ckpt = torch.load(args.init_checkpoint)[args.param_key]
        model.load_state_dict(ckpt)
    model = model.to(args.device)
    
    if args.mode == "zero_shot":
        result = val_mtr(test_loader, model, args)
        print(result)
    elif args.mode == "train":
        train_mtr(train_loader, val_loader, model, args)
        result = val_mtr(test_loader, model, args)
        print(result)