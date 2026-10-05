"""Mini-batch Adam on the weighted pseudo-likelihood with early stopping on a held-out split."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch

from .model import AttentionPotts


def train(Z: np.ndarray, w: np.ndarray, H: int = 32, d: int = 23, lambd: float = 1e-3,
          lr: float = 5e-3, batch_size: int = 1000, max_epochs: int = 500, patience: int = 25,
          val_frac: float = 0.1, seed: int = 0, device: str = "cuda", verbose: bool = True,
          field: str = "none", cond: np.ndarray | None = None, cond_noise: float = 0.0,
          lambd_h: float = 1e-4):
    """Returns (best model, history dict). Validation loss = weighted NLPL per sequence (no penalty).

    field='const' adds per-site fields; field='quad' adds fields that depend on the conditioning
    coordinates cond (M, 2). With cond_noise > 0 a fresh N(0, cond_noise^2) perturbation is added
    to the training coordinates at every batch (validation uses the clean coordinates), so the model
    learns p(x | z~) with z~ = z(x) + noise, i.e. conditioning on a region of width cond_noise.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    M, L = Z.shape
    perm = rng.permutation(M)
    n_val = int(round(val_frac * M))
    iv, it = perm[:n_val], perm[n_val:]

    Zt = torch.as_tensor(Z[it], device=device)
    wt = torch.as_tensor(w[it], dtype=torch.float32, device=device)
    Zv = torch.as_tensor(Z[iv], device=device)
    wv = torch.as_tensor(w[iv] / w[iv].sum(), dtype=torch.float32, device=device)

    model = AttentionPotts(L, H=H, d=d, seed=seed, field=field).to(device)
    ct = cv = None
    if field == "quad":
        c = torch.as_tensor(cond, dtype=torch.float32, device=device)
        ct, cv = c[it], c[iv]
        model.cond_mu.copy_(ct.mean(0))
        model.cond_sd.copy_(ct.std(0))
    gen = torch.Generator(device=device).manual_seed(seed + 7)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = {"epoch": [], "train_nlpl": [], "val_nlpl": [], "time": []}
    best, best_state, bad = np.inf, None, 0
    t0 = time.time()
    for ep in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(Zt), device=device)
        tot, totw = 0.0, 0.0
        for s in range(0, len(Zt), batch_size):
            b = order[s:s + batch_size]
            wb = wt[b]
            zb = None
            if ct is not None:
                zb = ct[b]
                if cond_noise > 0:
                    zb = zb + cond_noise * torch.randn(zb.shape, generator=gen, device=device)
            loss, nlpl = model.neg_log_pl(Zt[b], wb / wb.sum(), lambd, z=zb, lambd_h=lambd_h)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += nlpl.item() * wb.sum().item()
            totw += wb.sum().item()
        model.eval()
        with torch.no_grad():
            _, v = model.neg_log_pl(Zv, wv, lambd, z=cv)
        v = v.item()
        hist["epoch"].append(ep); hist["train_nlpl"].append(tot / totw)
        hist["val_nlpl"].append(v); hist["time"].append(time.time() - t0)
        if v < best - 1e-4:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if verbose and (ep % 10 == 0 or ep == 1):
            print(f"ep {ep:4d}  train {tot / totw:.4f}  val {v:.4f}  best {best:.4f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if bad >= patience:
            if verbose:
                print(f"early stop at epoch {ep}; best val {best:.4f}")
            break
    model.load_state_dict(best_state)
    hist["best_val_nlpl"] = best
    return model, hist
