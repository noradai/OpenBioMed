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

from utils.mg_metrics import get_cindex, get_rm2, accuracy, precision, auc_score, recall, f1_score, pr_auc
from datasets.dataset_dti import *
from models.deepeik_mgraph import MGraphDTA
from utils.utils import *
from log.train_logger import TrainLogger

def val_reg(model, criterion, dataloader, device):
    model.eval()
    running_loss = AverageMeter()

    pred_list = []
    label_list = []

    for data in dataloader:
        data = data.to(device)

        with torch.no_grad():
            pred = model(data)
            loss = criterion(pred.view(-1), data.y.view(-1))
            label = data.y
            pred_list.append(pred.view(-1).detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            running_loss.update(loss.item(), label.size(0))

    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    epoch_cindex = get_cindex(label, pred)
    epoch_r2 = get_rm2(label, pred)
    epoch_loss = running_loss.get_average()
    running_loss.reset()
    model.train()

    return epoch_loss, epoch_cindex, epoch_r2

def val_cls(model, criterion, dataloader, device):
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
    parser.add_argument('--save_model', action='store_true', help='whether save model or not')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    args = parser.parse_args()

    config = json.load(open(args.config_path, "r"))
    modle_type = config['modle']['type']
    DATASET = config['data']['name']
    
    params = dict(
        data_root="data",
        save_dir="save",
        dataset=DATASET,
        save_model=args.save_model,
        lr=args.lr,
        batch_size=args.batch_size
    )

    logger = TrainLogger(params)
    logger.info(__file__)
    
    # modle_type = 'classification'
    # DATASET = params.get("dataset")
    # save_model = params.get("save_model")
    # data_root = params.get("data_root")

    # TODO:
    train_csv = "train_fold0"
    test_csv = "test_fold0"
    
    if DATASET == "Y08":
        # 08
        train_dataset = Yamanishi08(config, train_csv)
        logger.info(f"Number of train: {len(train_dataset)}")
        test_dataset = Yamanishi08(config, test_csv)
    
    if DATASET == "Davis" or DATASET == "KIBA":
    # davis AND KIBA:
        train_dataset = Davis_KIBA(config, train_csv)
        logger.info(f"Number of train: {len(train_dataset)}")
        # TODO: 这里为啥是train_csv?
        test_dataset = Davis_KIBA(config, train_csv)
        logger.info(f"Number of test: {len(test_dataset)}")
    if DATASET == "BMKG":
        # BMKG
        train_dataset = BMKG(config, 'train')
        logger.info(f"Number of train: {len(train_dataset)}")
        test_dataset = BMKG(config, 'test')
    
    
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=2)

    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=2)
    
           
    # train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True, num_workers=8)
    # # val_loader = DataLoader(val_set, batch_size=params['batch_size'], shuffle=False, num_workers=8)
    # test_loader = DataLoader(test_set, batch_size=params['batch_size'], shuffle=False, num_workers=8)

    device = torch.device('cuda:0')
    print(modle_type)
    if modle_type == 'regression':
        
        model = MGraphDTA(3, 25 + 1, embedding_size=128, filter_num=32, out_dim=1).to(device)
    else:
        model = MGraphDTA(3, 25 + 1, embedding_size=128, filter_num=32, out_dim=2).to(device)

    epochs = 3000
    steps_per_epoch = 50
    num_iter = math.ceil((epochs * steps_per_epoch) / len(train_loader))
    if modle_type == 'regression':
        optimizer = optim.Adam(model.parameters(), lr=5e-4)
        criterion = nn.MSELoss()
    else:
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        criterion = nn.CrossEntropyLoss()

    global_step = 0
    global_epoch = 0
    early_stop_epoch = 400

    running_loss = AverageMeter()
    running_cindex = AverageMeter()
    running_best_mse = BestMeter("min")

    logger.info(f"lr====: {params['lr']}")

    model.train()

    for i in range(num_iter):
        for data in train_loader:
            
            global_step += 1  
            data = data.to(device)
            pred = model(data)
            
            # print('modle done!!!!!!!!!!!!')
            # print(pred,data.y)
            
            if modle_type == 'regression':
                loss = criterion(pred.view(-1), data.y.view(-1))
                cindex = get_cindex(data.y.detach().cpu().numpy().reshape(-1), pred.detach().cpu().numpy().reshape(-1))

            else: 
                data.y = data.y.long()
                loss = criterion(pred, data.y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.update(loss.item(), data.y.size(0))
            if modle_type == 'regression':
                running_cindex.update(cindex, data.y.size(0))

            if global_step % steps_per_epoch == 0:

                global_epoch += 1

                epoch_loss = running_loss.get_average()
                if modle_type == 'regression':
                    epoch_cindex = running_cindex.get_average()
                running_loss.reset()
                if modle_type == 'regression':
                    running_cindex.reset()
                    test_loss, test_ci, test_r2 = val_reg(model, criterion, test_loader, device)
                    msg = "epoch-%d, loss-%.4f, cindex-%.4f, test_loss-%.4f, test_ci-%.4f, test_r2-%.4f" % (global_epoch, epoch_loss, epoch_cindex, test_loss, test_ci, test_r2)
                    logger.info(msg)

                    if test_loss < running_best_mse.get_best():
                        running_best_mse.update(test_loss)
                        if i == 10 or i == 20 or i == 30 or i ==40:
                            save_model_dict(model, logger.get_model_dir(), msg)
                    else:
                        count = running_best_mse.counter()
                        if count > early_stop_epoch:
                            logger.info(f"early stop in epoch {global_epoch}")
                            break_flag = True
                            break
                    
                else:
                    test_loss, test_roc, test_prc, test_pre, test_rec, test_f1  = val_cls(model, criterion, test_loader, device)
                    msg = "epoch-%d, loss-%.4f, test_loss-%.4f" % (global_epoch, epoch_loss, test_loss) #cold seting
                    logger.info(msg)

                    msg = "test_roc_auc-%.4f, test_pr_auc-%.4f, test_f1-%.4f, test_precison-%.4f, test_recall-%.4f" % (test_roc, test_prc, test_f1, test_pre, test_rec)
                    logger.info(msg)
                    
                    if i == 10 or i == 20 or i == 30 or i ==40:
                        save_model_dict(model, logger.get_model_dir(), msg)


                # if save_model:
    save_model_dict(model, logger.get_model_dir(), msg)


if __name__ == "__main__":
    main()
