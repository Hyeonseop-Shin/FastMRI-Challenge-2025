import torch.optim
import numpy as np

class Scheduler:
    def __init__(self, optimizer: torch.optim):
        self.optimizer = optimizer

    def adjust_lr(self, step: int) -> float:
        lr = self.get_lr(step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        print(f"Adjusting learning rate to {lr:.6f}")
        return lr

    def get_lr(self, epoch: int) -> float:
        raise NotImplementedError("Must be implemented in subclasses")
    

class DoubleWarmupCosineScheduler(Scheduler):
    def __init__(
        self, optimizer: torch.optim.Optimizer, total_epochs: int,
        warmup1: int = 10, anneal1: int = 40,
        warmup2: int = 10, anneal2: int = 40,
        lr_max1: float = 0.0003, lr_min1: float = 0.00005,
        lr_max2: float = 0.00015, lr_min2: float = 0.0
    ):
        super().__init__(optimizer)

        assert total_epochs == warmup1 + anneal1 + warmup2 + anneal2, \
            "Total epochs must equal the sum of all warmup and anneal epochs"

        self.total_epochs = total_epochs
        self.warmup1 = warmup1
        self.anneal1 = anneal1
        self.warmup2 = warmup2
        self.anneal2 = anneal2
        self.lr_max1 = lr_max1
        self.lr_min1 = lr_min1
        self.lr_max2 = lr_max2
        self.lr_min2 = lr_min2

    def get_lr(self, epoch: int) -> float:
        stage1_end = self.warmup1
        stage2_end = stage1_end + self.anneal1
        stage3_end = stage2_end + self.warmup2
        stage4_end = stage3_end + self.anneal2

        if epoch < stage1_end:
            return self.lr_max1 * (epoch + 1) / self.warmup1

        elif epoch < stage2_end:
            progress = (epoch - stage1_end) / self.anneal1
            return self.lr_min1 + 0.5 * (self.lr_max1 - self.lr_min1) * (1 + np.cos(np.pi * progress))

        elif epoch < stage3_end:
            progress = (epoch - stage2_end + 1) / self.warmup2
            return self.lr_min1 + (self.lr_max2 - self.lr_min1) * progress

        elif epoch < stage4_end:
            progress = (epoch - stage3_end) / self.anneal2
            return self.lr_min2 + 0.5 * (self.lr_max2 - self.lr_min2) * (1 + np.cos(np.pi * progress))

        else:
            return self.lr_min2

    
class ConstantScheduler(Scheduler):
    def __init__(self, optimizer: torch.optim, lr: float):
        super().__init__(optimizer)
        self.lr = lr

    def get_lr(self, epoch: int) -> float:
        return self.lr


class CosineScheduler(Scheduler):
    def __init__(self, 
                 optimizer: torch.optim, 
                 total_epochs: int, 
                 lr_max: float = 0.0003, 
                 lr_min: float = 0.00003):
        super().__init__(optimizer)
        self.total_epochs = total_epochs
        self.lr_max = lr_max
        self.lr_min = lr_min

    def get_lr(self, epoch: int) -> float:
        progress = epoch / self.total_epochs
        return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(np.pi * progress))

class WarmupCosineScheduler(Scheduler):
    def __init__(self, 
                 optimizer: torch.optim.Optimizer, 
                 warmup_epochs: int, 
                 total_epochs: int):
        super().__init__(optimizer)
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

    def get_lr(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))