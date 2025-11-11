from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR

class WarmupCosineScheduler:
    def __init__(self, optimizer, total_steps, base_lr, 
                 warmup_percentage=0.2, 
                 min_lr_ratio=1.0e-3, 
                 warmup_min_lr_ratio=0.0):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = int(warmup_percentage * total_steps)
        self.base_lr = base_lr
        self.min_lr_ratio = min_lr_ratio
        self.warmup_min_lr_ratio = warmup_min_lr_ratio

        def warmup_lambda(step):
            if step <= self.warmup_steps:
                return self.warmup_min_lr_ratio + (1.0 - self.warmup_min_lr_ratio) * step / self.warmup_steps
            else:
                return 1.0

        self.scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LambdaLR(optimizer, lr_lambda=warmup_lambda),
                CosineAnnealingLR(optimizer, T_max=total_steps - self.warmup_steps, eta_min=base_lr * min_lr_ratio)
            ],
            milestones=[self.warmup_steps]
        )

    def step(self):
        self.scheduler.step()

    def get_scheduler(self):
        return self.scheduler