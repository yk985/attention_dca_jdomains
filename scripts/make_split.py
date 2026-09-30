"""Build a sequence-disjoint train/test split from the full bacterial J-domain set.

python scripts/make_split.py --thresh 0.2
Writes outputs/split/jdoms_{train,test}_c{thresh}.fasta (+ .npy) and split_stats.json.
"""
import argparse
import json

import numpy as np

from common import DATA, OUT
from attdca.data import load_msa, write_fasta, read_fasta, encode
from attdca.split import greedy_cluster, split_by_cluster
from attdca.metrics import min_hamming

p = argparse.ArgumentParser()
p.add_argument("--source", default=str(DATA / "jdoms_bacteria_filtered.fasta"))
p.add_argument("--thresh", type=float, default=0.2, help="cluster/separation distance (1 - identity)")
p.add_argument("--test_frac", type=float, default=0.2)
p.add_argument("--drop", choices=["exact", "threshold"], default="exact",
               help="'exact': drop only test sequences identical to a training one (default); "
                    "'threshold': also drop those within --thresh of one")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

D = OUT / "split"
D.mkdir(exist_ok=True)
Z = load_msa(a.source)
names, _ = read_fasta(a.source)
print(f"source: {a.source}  {len(Z)} unique sequences of length {Z.shape[1]}")

lab = greedy_cluster(Z, thresh=a.thresh)
sizes = np.bincount(lab)
print(f"{len(sizes)} clusters at {100 * (1 - a.thresh):.0f}% identity; "
      f"largest {sizes[:5]}, singletons {int((sizes == 1).sum())}")

itr, ite, st = split_by_cluster(Z, lab, test_frac=a.test_frac, thresh=a.thresh,
                                drop=a.drop, seed=a.seed)
Ztr, Zte = Z[itr], Z[ite]
d = min_hamming(Zte, Ztr)
st.update(source=a.source, thresh=a.thresh, n_total=len(Z), n_train=len(Ztr), n_test=len(Zte),
          test_copies=float((d == 0).mean()), nn_mean=float(d.mean()), nn_min=float(d.min()),
          nn_q05=float(np.quantile(d, 0.05)), nn_median=float(np.median(d)),
          frac_nn_lt_thresh=float((d < a.thresh).mean()),
          cluster_sizes_top10=sizes[:10].tolist(), n_singletons=int((sizes == 1).sum()))
tag = f"c{a.thresh:g}"
write_fasta(str(D / f"jdoms_train_{tag}.fasta"), Ztr, prefix="train")
write_fasta(str(D / f"jdoms_test_{tag}.fasta"), Zte, prefix="test")
np.save(D / f"labels_{tag}.npy", lab)
json.dump(st, open(D / f"split_stats_{tag}.json", "w"), indent=1)
print(f"train {len(Ztr)}  test {len(Zte)} (dropped {st['dropped_from_test']}, rule '{a.drop}')")
print(f"test -> nearest train: mean {st['nn_mean']:.3f}  median {st['nn_median']:.3f}  "
      f"5% {st['nn_q05']:.3f}  min {st['nn_min']:.3f}  exact copies {st['test_copies']:.3f}  "
      f"within {a.thresh:g}: {100 * st['frac_nn_lt_thresh']:.1f}%")
