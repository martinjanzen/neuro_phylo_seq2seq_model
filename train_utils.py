# train_utils.py
import hashlib, torch, numpy as np
import torch.optim as optim
from pathlib import Path
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Subset
from models import PhonologicalLoss, ReconstructionLSTM

DEVICE = "cpu"
BATCH_SIZE, MAX_EPOCHS, PATIENCE, LEARNING_RATE = 8, 150, 30, 0.001
N_FOLDS = 10
GLOBAL_SEED = 42

class _PlateauScheduler:
    def __init__(self, opt, factor=0.5, patience=15, min_lr=1e-5):
        self.opt, self.factor, self.patience, self.min_lr = opt, factor, patience, min_lr
        self.best, self.wait = float("inf"), 0
    def step(self, loss):
        if loss < self.best - 1e-4: self.best, self.wait = loss, 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                for g in self.opt.param_groups: g["lr"] = max(g["lr"] * self.factor, self.min_lr)
                self.wait = 0

def train_simple(train_loader, val_loader, run_tag: str, save_path=None, model_class=ReconstructionLSTM):
    seed = (int(hashlib.md5(run_tag.encode()).hexdigest(), 16) % (2 ** 32)) ^ GLOBAL_SEED
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

    model = model_class().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = _PlateauScheduler(optimizer)
    criterion = PhonologicalLoss()

    # C1 FIX: always checkpoint to a temp file so best weights are recovered
    # regardless of whether save_path is provided. The original code only saved
    # when save_path was set, meaning all CV folds returned last-epoch weights
    # instead of best-validation weights.
    tmp_ckpt = Path(f"_tmp_{run_tag}.pth")
    best_val, no_improve = float("inf"), 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            criterion(model(x.to(DEVICE)), y.to(DEVICE)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = sum(criterion(model(vx.to(DEVICE)), vy.to(DEVICE)).item()
                          for vx, vy in val_loader) / len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val, no_improve = val_loss, 0
            torch.save(model.state_dict(), tmp_ckpt)   # always checkpoint best
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            break

    # C1 FIX: unconditionally restore best weights before returning
    model.load_state_dict(torch.load(tmp_ckpt, map_location=DEVICE, weights_only=True))
    tmp_ckpt.unlink(missing_ok=True)

    if save_path:
        torch.save(model.state_dict(), save_path)
    return model, best_val

def _groups(dataset):
    base = [d["concept"].split("_")[0] for d in dataset.data]
    cmap = {c: i for i, c in enumerate(sorted(set(base)))}
    return [cmap[c] for c in base]

def get_fold0_split(dataset, n_folds=N_FOLDS):
    groups = _groups(dataset)
    return next(iter(GroupKFold(n_splits=min(n_folds, len(set(groups)))).split(dataset.data, groups=groups)))

def run_cv(dataset, n_folds=N_FOLDS, exclude_idx=None, tag_prefix="cv"):
    groups = _groups(dataset)
    losses = []
    for fi, (tr_ids, va_ids) in enumerate(GroupKFold(n_splits=n_folds).split(dataset.data, groups=groups)):
        _, loss = train_simple(
            DataLoader(Subset(dataset, tr_ids), batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(Subset(dataset, va_ids), batch_size=BATCH_SIZE, shuffle=False),
            f"{tag_prefix}_excl{exclude_idx}_f{fi}")
        losses.append(loss)
    return losses