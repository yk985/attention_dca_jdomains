"""Train AttentionDCA on the jdoms training alignment.

python scripts/train_model.py --H 32 --d 23 --lambd 1e-3 --tag main
"""
import argparse
import json

import numpy as np
import torch

from common import OUT, add_split_arg, resolve
from attdca.data import load_msa, compute_weights
from attdca.train import train

p = add_split_arg(argparse.ArgumentParser())
p.add_argument("--H", type=int, default=32)
p.add_argument("--d", type=int, default=23)
p.add_argument("--lambd", type=float, default=1e-3)
p.add_argument("--lr", type=float, default=5e-3)
p.add_argument("--batch", type=int, default=1000)
p.add_argument("--epochs", type=int, default=500)
p.add_argument("--patience", type=int, default=25)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--tag", default="main")
a = p.parse_args()

TRAIN, _ = resolve(a.split)
Z = load_msa(str(TRAIN))
w, meff, theta = compute_weights(Z)
print(f"train ({a.split}): M={len(Z)} L={Z.shape[1]} theta={theta:.3f} Meff={meff:.1f}")

model, hist = train(Z, w, H=a.H, d=a.d, lambd=a.lambd, lr=a.lr, batch_size=a.batch,
                    max_epochs=a.epochs, patience=a.patience, seed=a.seed)

np.savez(OUT / f"weights_train_{a.split}.npz", w=w, theta=theta, meff=meff)
torch.save({"state": model.state_dict(), "cfg": vars(a), "L": Z.shape[1]}, OUT / f"model_{a.tag}.pt")
with torch.no_grad():
    np.save(OUT / f"J_{a.tag}.npy", model.couplings().cpu().numpy())
hist.update(cfg=vars(a), M=len(Z), theta=theta, meff=meff,
            n_params=sum(p.numel() for p in model.parameters()))
json.dump(hist, open(OUT / f"train_hist_{a.tag}.json", "w"), indent=1)
print(f"saved model_{a.tag}.pt  best val NLPL {hist['best_val_nlpl']:.4f}")
