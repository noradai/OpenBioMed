import os
import numpy as np
import torch

import datetime

class BestMeter(object):
    """Computes and stores the best value"""

    def __init__(self, best_type):
        self.best_type = best_type  
        self.count = 0      
        self.reset()

    def reset(self):
        if self.best_type == 'min':
            self.best = float('inf')
        else:
            self.best = -float('inf')

    def update(self, best):
        self.best = best
        self.count = 0

    def get_best(self):
        return self.best

    def counter(self):
        self.count += 1
        return self.count


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)

        return self.avg

class EarlyStopping(object):
    """
    Parameters
    ----------
    mode : str
        * 'higher': Higher metric suggests a better model
        * 'lower': Lower metric suggests a better model
        If ``metric`` is not None, then mode will be determined
        automatically from that.
    patience : int
        The early stopping will happen if we do not observe performance
        improvement for ``patience`` consecutive epochs.
    filename : str or None
        Filename for storing the model checkpoint. If not specified,
        we will automatically generate a file starting with ``early_stop``
        based on the current time.
    metric : str or None
        A metric name that can be used to identify if a higher value is
        better, or vice versa. Default to None. Valid options include:
        ``'r2'``, ``'mae'``, ``'rmse'``, ``'roc_auc_score'``.
    """

    def __init__(self, mode='higher', patience=10, filename=None, metric=None):
        if filename is None:
            dt = datetime.datetime.now()
            folder = os.path.join(os.getcwd(), 'results')
            if not os.path.exists(folder):
                os.makedirs(folder)
            filename = os.path.join(folder, 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
                dt.date(), dt.hour, dt.minute, dt.second))

        if metric is not None:
            assert metric in ['r2', 'mae', 'rmse', 'roc_auc_score', 'pr_auc_score'], \
                "Expect metric to be 'r2' or 'mae' or " \
                "'rmse' or 'roc_auc_score', got {}".format(metric)
            if metric in ['r2', 'roc_auc_score', 'pr_auc_score']:
                print('For metric {}, the higher the better'.format(metric))
                mode = 'higher'
            if metric in ['mae', 'rmse']:
                print('For metric {}, the lower the better'.format(metric))
                mode = 'lower'

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        """Check if the new score is higher than the previous best score.
        Parameters
        ----------
        score : float
            New score.
        prev_best_score : float
            Previous best score.
        Returns
        -------
        bool
            Whether the new score is higher than the previous best score.
        """
        return score > prev_best_score

    def _check_lower(self, score, prev_best_score):
        """Check if the new score is lower than the previous best score.
        Parameters
        ----------
        score : float
            New score.
        prev_best_score : float
            Previous best score.
        Returns
        -------
        bool
            Whether the new score is lower than the previous best score.
        """
        return score < prev_best_score

    def step(self, score, model):
        """Update based on a new score.
        The new score is typically model performance on the validation set
        for a new epoch.
        Parameters
        ----------
        score : float
            New score.
        model : nn.Module
            Model instance.
        Returns
        -------
        bool
            Whether an early stop should be performed.
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when the metric on the validation set gets improved.
        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        torch.save({'model_state_dict': model.state_dict()}, self.filename)

    def load_checkpoint(self, model):
        '''Load the latest checkpoint
        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        model.load_state_dict(torch.load(self.filename)['model_state_dict'])

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def save_checkpoint(model, model_dir, epoch, val_loss, val_acc):
    model_path = os.path.join(model_dir, 'epoch:%d-val_loss:%.3f-val_acc:%.3f.model' % (epoch, val_loss, val_acc))
    torch.save(model, model_path)

def load_checkpoint(model_path):
    return torch.load(model_path)

def save_model_dict(model, model_dir, msg):
    model_path = os.path.join(model_dir, msg + '.pt')
    torch.save(model.state_dict(), model_path)
    print("model has been saved to %s." % (model_path))

def load_model_dict(model, ckpt):
    model.load_state_dict(torch.load(ckpt))

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def metrics_average(results):
    metrics = {key: [] for key in results[0]}
    for result in results:
        for key in result:
            metrics[key].append(result[key])
    for key in metrics:
        metrics[key] = (np.mean(metrics[key]), np.std(metrics[key]))
    return metrics
