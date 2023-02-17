import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn

from datasets.dti_dataset import SUPPORTED_DTI_DATASET
from utils.metric import *
from utils.meters import AverageMeter

def train_dti(dataset, model, config):
    train_loader, valid_loader, test_loader = dataset.get_data_loaders()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if config["task"] == "classification":
        loss_fn = nn.BCELoss()
    elif config["task"] == "regression":
        loss_fn = nn.MSELoss()
    
    running_loss = AverageMeter()
    best_result = -1e9
    best_model = None
    for epoch in range(args.epochs):
        logger.info("Epoch %d" % (epoch))
        for data in tqdm(dataloader):
            model.train()
            pred = model(data)
            loss = loss_fn(pred.view(-1), data.y.view(-1))
            loss.backward()
            running_loss.update(loss.item(), data.y.size(0))

            step += 1
            if step % args.gradient_accumulation_steps == 0:
                logger.info("step loss %.4lf" % (running_loss.get_average() / args.gradient_accumulation_steps))
                optimizer.step()
                optimizer.zero_grad()

        if i % 10 == 0:
            evaluate("train", train_loader, model, config)
        if valid_loader is not None:
            results = evaluate("valid", valid_loader, model, config)
            if results[0] > best_result:
                best_result = results[0]
                best_model = model
        if (i + 1) % args.save_epochs == 0:
            torch.save(model.state_dict(), osp.join(args.save_path, 'epoch' + str(epoch) + ".pth"))
    return best_model

def eval_dti(split, dataloader, model, config):
    if task == "classification":
        loss_fn = nn.BCELoss()
    elif task == "regression":
        loss_fn = nn.MSELoss()

    all_loss = 0
    all_preds = []
    all_labels = []
    for data in tqdm(dataloader):
        model.eval()
        pred = model(data)
        
        all_loss += loss_fn(pred, label).item()
        all_preds.append(np.array(pred.detach().cpu()))
        all_labels.append(np.array(label.detach().cpu()))
    print("Average ", split, "loss: ", all_loss / ((len(dataset) - 1) // args.batch_size + 1))
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    if task == "classification":
        outputs = np.array([1 if x >= 0.5 else 0 for x in all_preds])
        results = {
            "ROC_AUC": roc_auc(all_labels, all_preds), 
            "PR_AUC": pr_auc(all_labels, all_preds), 
            "F1": f1_score(all_labels, outputs),
            "Precision": precision_score(all_labels, outputs),
            "Recall": recall_score(all_labels, outputs), 
        }
    elif task == "regression":
        # print(all_labels, all_preds)
        results = {
            "MSE": mean_squared_error(all_labels, all_preds),
            "Pearson": pearsonr(all_labels, all_preds)[0],
            "Spearman": spearmanr(all_labels, all_preds)[0],
            "CI": concordance_index(all_labels, all_preds),
            "r_m^2": get_rm2(all_labels, all_preds)
        }
    print(results)
    return results