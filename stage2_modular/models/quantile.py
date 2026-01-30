# -*- coding: utf-8 -*-
import contextlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from ..core.device import resolve_device
from ..core.utils import Stopwatch

class QuantileMLP(nn.Module):
    def __init__(self, in_dim: int, taus, hidden=[512,512,256,128], act="relu", dropout=0.05):
        super().__init__()
        self.taus = list(taus)
        acts = {"relu": nn.ReLU(), "gelu": nn.GELU(), "tanh": nn.Tanh(), "silu": nn.SiLU()}
        layers=[]; prev=in_dim
        for h in hidden:
            layers += [nn.Linear(prev,h), acts.get(act, nn.ReLU())]
            if dropout and dropout>0:
                layers += [nn.Dropout(dropout)]
            prev=h
        layers += [nn.Linear(prev, len(self.taus))]
        self.net = nn.Sequential(*layers)
    def forward(self,x): 
        return self.net(x)

def pinball_loss(pred: torch.Tensor, y: torch.Tensor, taus):
    diff = y - pred  # [B,T]
    losses = []
    for i, tau in enumerate(taus):
        e = diff[:, i]
        losses.append(torch.maximum(tau * e, (tau - 1.0) * e).mean())
    return torch.stack(losses).sum()

def non_crossing_penalty(pred: torch.Tensor, weight: float = 1.0):
    if pred.shape[1] <= 1 or weight <= 0:
        return pred.new_tensor(0.0)
    diffs = pred[:, :-1] - pred[:, 1:]
    penalty = torch.relu(diffs).mean()
    return weight * penalty

def fit_quantile_mlp(Xtr, ytr, Xva, yva, taus, cfg=None, device="auto", verbose=1, gpu_cache_limit_bytes=20*1024**3):
    sw = Stopwatch()
    dev = resolve_device(device)
    use_cuda = (dev.type == "cuda")
    scaler_amp = torch.amp.GradScaler('cuda') if use_cuda else None
    autocast = torch.amp.autocast('cuda', dtype=torch.float16) if use_cuda else contextlib.nullcontext()

    hidden = cfg.get("hidden", [512,512,256,128]); act = cfg.get("act", "relu")
    dropout = cfg.get("dropout", 0.05); lr = cfg.get("lr", 1e-3); l2 = cfg.get("l2", 0.0)
    epochs = int(cfg.get("epochs", 40)); patience = int(cfg.get("patience", 8)); batch = int(cfg.get("batch", 65536))
    ncw = float(cfg.get("non_cross_penalty", 0.0))

    in_dim = Xtr.shape[1]
    model = QuantileMLP(in_dim=in_dim, taus=taus, hidden=hidden, act=act, dropout=dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    sw.lap("init model/opt")

    def to_tensor(X, y):
        X = torch.from_numpy(X.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32).reshape(-1,1))
        return X, y

    Xtr_cpu, ytr_cpu = to_tensor(Xtr, ytr); Xva_cpu, yva_cpu = to_tensor(Xva, yva)
    sw.lap("prepare CPU tensors")

    total_bytes = Xtr_cpu.element_size()*Xtr_cpu.nelement() + ytr_cpu.element_size()*ytr_cpu.nelement() +                   Xva_cpu.element_size()*Xva_cpu.nelement() + yva_cpu.element_size()*yva_cpu.nelement()
    use_cached = use_cuda and (total_bytes <= gpu_cache_limit_bytes)
    mode = "GPU-CACHED" if use_cached else "STREAMING"
    print(f"[Quantile] bytes≈{total_bytes/1024**2:.1f}MiB; mode={mode}")

    if use_cached:
        Xtr=Xtr_cpu.to(dev); ytr=ytr_cpu.to(dev)
        Xva=Xva_cpu.to(dev); yva=yva_cpu.to(dev)
        n=Xtr.shape[0]; steps=max(1,(n+batch-1)//batch)
        print(f"[Quantile] n_train={n}, steps/epoch={steps}, batch={batch}")
        best=float("inf"); bad=0
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, batch):
                idx = perm[i:i+batch]; xb=Xtr.index_select(0,idx); yb=ytr.index_select(0,idx)
                opt.zero_grad(set_to_none=True)
                with autocast:
                    pred = model(xb)
                    loss = pinball_loss(pred.float(), yb, taus) + non_crossing_penalty(pred.float(), ncw)
                if scaler_amp:
                    scaler_amp.scale(loss).backward(); scaler_amp.step(opt); scaler_amp.update()
                else:
                    loss.backward(); opt.step()
            model.eval()
            with torch.no_grad(), autocast:
                pv = model(Xva).float()
                vloss = pinball_loss(pv, yva, taus).item() + non_crossing_penalty(pv, ncw).item()
            if verbose and ((ep+1)%5==0 or ep==0):
                print(f"  [QMLP] epoch {ep+1:03d}, val_pinball={vloss:.6f}")
            if vloss < best - 1e-9:
                best=vloss; bad=0; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:
                bad+=1
                if bad>=patience: break
        if 'best_state' in locals():
            model.load_state_dict(best_state)
        sw.total(f"fit_quantile_mlp ({mode})")
        model.eval(); return model, mode

    tr_loader = DataLoader(TensorDataset(Xtr_cpu, ytr_cpu), batch_size=batch, shuffle=True, drop_last=False)
    va_loader = DataLoader(TensorDataset(Xva_cpu, yva_cpu), batch_size=batch, shuffle=False, drop_last=False)
    best=float("inf"); bad=0
    for ep in range(epochs):
        model.train()
        for xb_cpu,yb_cpu in tr_loader:
            xb=xb_cpu.to(dev); yb=yb_cpu.to(dev)
            opt.zero_grad(set_to_none=True)
            with (torch.amp.autocast('cuda', dtype=torch.float16) if use_cuda else contextlib.nullcontext()):
                pred=model(xb); loss=pinball_loss(pred.float(), yb, taus) + non_crossing_penalty(pred.float(), ncw)
            if scaler_amp:
                scaler_amp.scale(loss).backward(); scaler_amp.step(opt); scaler_amp.update()
            else:
                loss.backward(); opt.step()
        model.eval(); vsum=0.0; vcnt=0
        with torch.no_grad(), (torch.amp.autocast('cuda', dtype=torch.float16) if use_cuda else contextlib.nullcontext()):
            for xb_cpu,yb_cpu in va_loader:
                xb=xb_cpu.to(dev); yb=yb_cpu.to(dev)
                pred=model(xb)
                l = pinball_loss(pred.float(), yb, taus) + non_crossing_penalty(pred.float(), ncw)
                vsum += l.item()*len(xb); vcnt += len(xb)
        vloss=vsum/max(vcnt,1)
        if verbose and ((ep+1)%5==0 or ep==0):
            print(f"  [QMLP] epoch {ep+1:03d}, val_pinball={vloss:.6f}")
        if vloss < best - 1e-9:
            best=vloss; bad=0; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            bad+=1
            if bad>=patience: break
    if 'best_state' in locals():
        model.load_state_dict(best_state)
    sw.total("fit_quantile_mlp (STREAMING)"); model.eval(); return model, "STREAMING"

@torch.no_grad()
def predict_quantiles(model, X):
    dev = next(model.parameters()).device
    use_cuda = (dev.type == "cuda")
    X_t = torch.from_numpy(X.astype(np.float32)).to(dev)
    with torch.amp.autocast('cuda', dtype=torch.float16) if use_cuda else contextlib.nullcontext():
        preds = model(X_t).float().cpu().numpy()
    return preds
