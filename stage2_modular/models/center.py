# -*- coding: utf-8 -*-
import math, contextlib
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from ..core.device import resolve_device
from ..core.utils import Stopwatch
from ..core.dmode import build_D_from_yhat

class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden=[512,512,256,128], act="relu", dropout=0.05):
        super().__init__()
        acts = {"relu": nn.ReLU(), "gelu": nn.GELU(), "tanh": nn.Tanh(), "silu": nn.SiLU()}
        layers=[]; prev=in_dim
        for h in hidden:
            layers += [nn.Linear(prev,h), acts.get(act, nn.ReLU())]
            if dropout and dropout>0:
                layers += [nn.Dropout(dropout)]
            prev=h
        layers += [nn.Linear(prev,1)]
        self.net = nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

class LossBuilder:
    def __init__(self, kind: str = "mse", huber_delta_z: float = 1.0,
                 d_mode: str = "pred_or_both", eps_ratio: float = 0.05,
                 delta_power: float = 50.0, prated_used: float | None = None):
        self.kind = (kind or "mse").lower()
        self.huber_delta_z = float(huber_delta_z)
        self.d_mode = d_mode
        self.eps_ratio = float(eps_ratio)
        self.delta_power = float(delta_power)
        self.prated_used = (float(prated_used) if prated_used is not None else float("nan"))
    @torch.no_grad()
    def _make_D(self, pred: torch.Tensor) -> torch.Tensor:
        return build_D_from_yhat(pred.detach(), self.prated_used, self.d_mode,
                                 self.eps_ratio, self.delta_power)
    def __call__(self, pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.kind == "mse":
            return torch.mean((pred - y_true) ** 2)
        D = self._make_D(pred)
        if self.kind == "wmse":
            return torch.mean(((pred - y_true) ** 2) / (D * D))
        if self.kind == "huber_z":
            z = (pred - y_true) / D
            return torch.nn.functional.smooth_l1_loss(
                z, torch.zeros_like(z), beta=self.huber_delta_z, reduction="mean"
            )
        return torch.mean((pred - y_true) ** 2)

def fit_mlp_center(Xv_tr, Xr_tr, y_tr, Xv_va, Xr_va, y_va, use_rho=True, mlp_cfg=None,
                   device="auto", verbose=1, gpu_cache_limit_bytes=20*1024**3,
                   thresholds_cfg=None, prated_used=None):
    sw=Stopwatch()
    dev=resolve_device(device)
    use_cuda=(dev.type=="cuda")
    scaler_amp = torch.amp.GradScaler('cuda') if use_cuda else None
    autocast = torch.amp.autocast('cuda', dtype=torch.float16) if use_cuda else contextlib.nullcontext()

    in_dim=2 if use_rho else 1
    model=MLP(in_dim=in_dim, hidden=mlp_cfg["hidden"], act=mlp_cfg["act"], dropout=mlp_cfg["dropout"]).to(dev)
    opt=torch.optim.Adam(model.parameters(), lr=mlp_cfg["lr"], weight_decay=mlp_cfg["l2"])

    loss_kind = (mlp_cfg.get("loss","mse") if mlp_cfg else "mse")
    huber_delta_z = float(mlp_cfg.get("huber_delta_z", 1.0)) if mlp_cfg else 1.0
    d_mode      = (thresholds_cfg or {}).get("D_mode", "pred_or_both")
    eps_ratio   = float((thresholds_cfg or {}).get("eps_ratio", 0.05))
    delta_power = float((thresholds_cfg or {}).get("delta_power", 50.0))
    loss_builder = LossBuilder(kind=loss_kind, huber_delta_z=huber_delta_z,
                               d_mode=d_mode, eps_ratio=eps_ratio,
                               delta_power=delta_power, prated_used=prated_used)
    sw.lap("init model/opt/loss")

    def pack(v,r,y):
        X = np.c_[v,r].astype(np.float32) if use_rho else v.astype(np.float32).reshape(-1,1)
        y = y.astype(np.float32).reshape(-1,1)
        return torch.from_numpy(X), torch.from_numpy(y)

    Xtr_cpu,Ytr_cpu=pack(Xv_tr, (Xr_tr if use_rho else None), y_tr)
    Xva_cpu,Yva_cpu=pack(Xv_va, (Xr_va if use_rho else None), y_va)
    sw.lap("prepare CPU tensors")

    total_bytes = Xtr_cpu.element_size()*Xtr_cpu.nelement() + Ytr_cpu.element_size()*Ytr_cpu.nelement() +                   Xva_cpu.element_size()*Xva_cpu.nelement() + Yva_cpu.element_size()*Yva_cpu.nelement()
    use_cached = use_cuda and (total_bytes <= gpu_cache_limit_bytes)
    mode = "GPU-CACHED" if use_cached else "STREAMING"
    print(f"[Data] bytes≈{total_bytes/1024**2:.1f}MiB; mode={mode}")

    if use_cached:
        Xtr=Xtr_cpu.to(dev); Ytr=Ytr_cpu.to(dev)
        Xva=Xva_cpu.to(dev); Yva=Yva_cpu.to(dev)
        n=Xtr.shape[0]; steps=max(1,(n+mlp_cfg["batch"]-1)//mlp_cfg["batch"])
        print(f"[Data] n_train={n}, steps/epoch={steps}, batch={mlp_cfg['batch']}")
        best=float("inf"); bad=0
        for ep in range(mlp_cfg["epochs"]):
            model.train(); perm=torch.randperm(n, device=dev)
            for i in range(0,n,mlp_cfg["batch"]):
                idx=perm[i:i+mlp_cfg["batch"]]; xb=Xtr.index_select(0,idx); yb=Ytr.index_select(0,idx)
                opt.zero_grad(set_to_none=True)
                with autocast: pred=model(xb); loss=loss_builder(pred.float(), yb)
                if scaler_amp:
                    scaler_amp.scale(loss).backward(); scaler_amp.step(opt); scaler_amp.update()
                else:
                    loss.backward(); opt.step()
            model.eval(); 
            with torch.no_grad(), autocast: vloss=loss_builder(model(Xva).float(), Yva).item()
            if verbose and ((ep+1)%5==0 or ep==0):
                print(f"  [MLP] epoch {ep+1:03d}, val_loss={vloss:.6f} ({loss_kind})")
            if vloss+1e-9<best:
                best=vloss; bad=0
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:
                bad+=1
                if bad>=mlp_cfg["patience"]: break
        if 'best_state' in locals():
            model.load_state_dict(best_state)
        sw.total(f"fit_mlp_center ({mode})"); model.eval(); return model, mode

    tr_loader = DataLoader(TensorDataset(Xtr_cpu,Ytr_cpu), batch_size=mlp_cfg["batch"], shuffle=True, drop_last=False)
    va_loader = DataLoader(TensorDataset(Xva_cpu,Yva_cpu), batch_size=mlp_cfg["batch"], shuffle=False, drop_last=False)
    best=float("inf"); bad=0
    for ep in range(mlp_cfg["epochs"]):
        model.train()
        for xb_cpu,yb_cpu in tr_loader:
            xb=xb_cpu.to(dev); yb=yb_cpu.to(dev)
            opt.zero_grad(set_to_none=True)
            with (torch.amp.autocast('cuda', dtype=torch.float16) if use_cuda else contextlib.nullcontext()):
                pred=model(xb); loss=loss_builder(pred.float(), yb)
            if scaler_amp:
                scaler_amp.scale(loss).backward(); scaler_amp.step(opt); scaler_amp.update()
            else:
                loss.backward(); opt.step()
        model.eval(); vsum=0.0; vcnt=0
        with torch.no_grad(), (torch.amp.autocast('cuda', dtype=torch.float16) if use_cuda else contextlib.nullcontext()):
            for xb_cpu,yb_cpu in va_loader:
                xb=xb_cpu.to(dev); yb=yb_cpu.to(dev)
                pred=model(xb); loss=loss_builder(pred.float(), yb)
                vsum += loss.item()*len(xb); vcnt += len(xb)
        vloss=vsum/max(vcnt,1)
        if verbose and ((ep+1)%5==0 or ep==0):
            print(f"  [MLP] epoch {ep+1:03d}, val_loss={vloss:.6f} ({loss_kind})")
        if vloss+1e-9<best:
            best=vloss; bad=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            bad+=1
            if bad>=mlp_cfg["patience"]:
                break
    if 'best_state' in locals():
        model.load_state_dict(best_state)
    sw.total("fit_mlp_center (STREAMING)"); model.eval(); return model, "STREAMING"

def predict_mlp_center(model, wind_std_vec, rho_std_vec, prated, use_rho=True, clip_to_prated=True):
    dev=next(model.parameters()).device; use_cuda=(dev.type=="cuda")
    X = np.c_[wind_std_vec, rho_std_vec].astype(np.float32) if use_rho else wind_std_vec.astype(np.float32).reshape(-1,1)
    with torch.no_grad(), (torch.amp.autocast('cuda', dtype=torch.float16) if use_cuda else contextlib.nullcontext()):
        y = model(torch.from_numpy(X).to(dev)).float().cpu().numpy().reshape(-1)
    if clip_to_prated and math.isfinite(prated):
        y=np.clip(y,0.0,prated)
    return y
