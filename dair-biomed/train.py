# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import math
import numpy as np
import sklearn
import torch.optim as optim
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import argparse
import json

from utils.mg_metrics import accuracy, precision, auc_score, recall, f1_score, pr_auc
from datasets.dataset_dti import *
from models.deepeik_mgraph import MGraphDTA
from utils.utils import *
from log.train_logger import TrainLogger


def val(model, criterion, dataloader, device):
    model.eval()
    running_loss = AverageMeter()

    pred_list = []
    pred_cls_list = []
    label_list = []

    for data in dataloader:
        data.y = data.y.long()  
        data = data.to(device)

        with torch.no_grad():
            pred = model(data)
            loss = criterion(pred, data.y)
            pred_cls = torch.argmax(pred, dim=-1)

            pred_prob = F.softmax(pred, dim=-1)
            pred_prob, indices = torch.max(pred_prob, dim=-1)
            pred_prob[indices == 0] = 1. - pred_prob[indices == 0]

            pred_list.append(pred_prob.view(-1).detach().cpu().numpy())
            pred_cls_list.append(pred_cls.view(-1).detach().cpu().numpy())
            label_list.append(data.y.detach().cpu().numpy())
            running_loss.update(loss.item(), data.y.size(0))

    pred = np.concatenate(pred_list, axis=0)
    pred_cls = np.concatenate(pred_cls_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    """
    print('pred, pred_cls')
    print(pred, pred_cls)
    print('+++++++++++++++done')
    """
    rocauc = auc_score(label, pred)
    prauc = pr_auc(label, pred)
    pre = precision(label, pred_cls)
    rec = recall(label, pred_cls)
    f1 = f1_score(label, pred_cls)

    epoch_loss = running_loss.get_average()
    running_loss.reset()

    model.train()

    return epoch_loss, rocauc, prauc, pre, rec, f1

def main():
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument("--config-path", type=str, default="/share/project/biomed/hk/hk/Open_DAIR_BioMed/dair-biomed/configs/dti_base.json")
    parser.add_argument('--dataset', default="Y08", help='human or celegans')
    parser.add_argument('--save_model', action='store_true', help='whether save model or not')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    args = parser.parse_args()

    params = dict(
        data_root="data",
        save_dir="save",
        dataset=args.dataset,
        save_model=args.save_model,
        lr=args.lr,
        batch_size=args.batch_size
    )

    logger = TrainLogger(params)
    logger.info(__file__)
    config = json.load(open(args.config_path, "r"))

    DATASET = params.get("dataset")
    save_model = params.get("save_model")
    data_root = params.get("data_root")
    fpath = os.path.join(data_root, DATASET)
    train_csv = "train_fold0"
    test_csv = "test_fold0"
    
    train_dataset = Yamanishi08(config, train_csv)
    logger.info(f"Number of train: {len(train_dataset)}")
    test_dataset = Yamanishi08(config, test_csv)

    # train_set = GNNDataset(fpath, types='train')
    # # val_set = GNNDataset(fpath, types='val')
    # test_set = GNNDataset(fpath, types='test')

    
    # logger.info(f"Number of val: {len(val_set)}")
    logger.info(f"Number of test: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=2)

    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=2)
    
           
    # train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True, num_workers=8)
    # # val_loader = DataLoader(val_set, batch_size=params['batch_size'], shuffle=False, num_workers=8)
    # test_loader = DataLoader(test_set, batch_size=params['batch_size'], shuffle=False, num_workers=8)

    device = torch.device('cuda:0')
    model = MGraphDTA(3, 25 + 1, embedding_size=128, filter_num=32, out_dim=2).to(device)

    epochs = 100
    steps_per_epoch = 100
    num_iter = math.ceil((epochs * steps_per_epoch) / len(train_loader))

    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    criterion = nn.CrossEntropyLoss()

    global_step = 0
    global_epoch = 0

    running_loss = AverageMeter()

    logger.info(f"lr====: {params['lr']}")

    model.train()
    
    

    for i in range(num_iter):
        for data in train_loader:
            # print(data)

            global_step += 1   
            data.y = data.y.long()
            data = data.to(device)
            pred = model(data)
            
            # print('modle done!!!!!!!!!!!!')
            # print(pred,data.y)

            loss = criterion(pred, data.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.update(loss.item(), data.y.size(0))

            if global_step % steps_per_epoch == 0:

                global_epoch += 1

                epoch_loss = running_loss.get_average()
                running_loss.reset()

                # val_loss, val_roc, val_prc, val_pre, val_rec, val_f1  = val(model, criterion, val_loader, device)
                test_loss, test_roc, test_prc, test_pre, test_rec, test_f1  = val(model, criterion, test_loader, device)

                # msg = "epoch-%d, loss-%.4f, val_auc-%.4f, test_loss-%.4f, test_acc-%.4f, test_pre-%.4f, test_rec-%.4f, test_auc-%.4f" % (global_epoch, epoch_loss, val_auc, test_loss, test_acc, test_pre, test_rec, test_auc)
                
                # msg = "epoch-%d, loss-%.4f, val_loss-%.4f, test_loss-%.4f" % (global_epoch, epoch_loss, val_loss, test_loss)

                msg = "epoch-%d, loss-%.4f, test_loss-%.4f" % (global_epoch, epoch_loss, test_loss) #cold seting
                logger.info(msg)
                # msg = "val_roc_auc-%.4f, val_pr_auc-%.4f, val_f1-%.4f, val_precison-%.4f, val_recall-%.4f" % (val_roc, val_prc, val_f1, val_pre, val_rec)
                logger.info(msg)
                msg = "test_roc_auc-%.4f, test_pr_auc-%.4f, test_f1-%.4f, test_precison-%.4f, test_recall-%.4f" % (test_roc, test_prc, test_f1, test_pre, test_rec)
                logger.info(msg)
                if i == 10 or i == 20 or i == 30 or i ==40:
                    save_model_dict(model, logger.get_model_dir(), msg)


                # if save_model:
    save_model_dict(model, logger.get_model_dir(), msg)


if __name__ == "__main__":
    main()
