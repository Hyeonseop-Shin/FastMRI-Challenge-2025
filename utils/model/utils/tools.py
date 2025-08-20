import torch
import math
import numpy as np

def lr_scheduler(optimizer: torch.optim,
                 epoch: int,
                 total_epoch: int,
                 type:str ='exp',
                 lr_init=1e-4,
                 lr_min=1e-6,
                 exp_rate=0.5):
    
    if type == 'cosine':
        phase = math.pi * epoch / (total_epoch - 1)
        cosine_decay = 0.5 * (1 + math.cos(phase))
        lr = (lr_init - lr_min) * cosine_decay + lr_min
    
    elif type == 'exp':
        lr = lr_init * math.pow(exp_rate, epoch)
        lr = lr if lr > lr_min else lr_min

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print(f'Updating learning rate to {lr:.4e}')


class EarlyStopping:
    def __init__(self, patience=3, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.Inf
        self.delta = delta

    def __call__(self, loss, model, path, name, val=True):
        score = -loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model, path, name, val=val)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(loss, model, path, name, val=val)
            self.counter = 0

    def save_checkpoint(self, loss, model, path, name, val=True):
        loss_type = 'Validation' if val else 'Train'
        if self.verbose:
            print(f'{loss_type} loss decreased ({self.loss_min:.5f} --> {loss:.5f}), Save Model')
        torch.save(model.state_dict(), path + f'{name}.pth')
        self.loss_min = loss