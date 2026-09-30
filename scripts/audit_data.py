"""Audit the J-domain FASTA files: sizes, duplicates, cross-file overlap, test-to-train distances.

python scripts/audit_data.py      ->  outputs/data_audit.json + printed tables
"""
import json

import numpy as np

from common import DATA, OUT, ROOT
from attdca.data import read_fasta, encode, GAP
from attdca.metrics import min_hamming

FILES = {
    "filtered": DATA / "jdoms_bacteria_filtered.fasta",
    "filtered2": DATA / "jdoms_bacteria_filtered2.fasta",
    "train": DATA / "jdoms_bacteria_train.fasta",
    "test": DATA / "jdoms_bacteria_test.fasta",
    "train2": DATA / "jdoms_bacteria_train2.fasta",
    "test2": DATA / "jdoms_bacteria_test2.fasta",
    "test_4000": DATA / "jdoms_bacteria_test_4000.fasta",
    "refprot_raw": DATA / "jdoms_refprot_aligned_filtered_new_before_species_filtering.fasta",
    "refprot": ROOT / "transfer_folder/DataAttentionDCA/matches/jdoms_refprot_aligned_filtered_new.fasta",
}

S, Z, ID, info = {}, {}, {}, {}
for k, f in FILES.items():
    names, seqs = read_fasta(str(f))
    z = encode(seqs)
    S[k] = set(map(tuple, z))
    Z[k] = np.unique(z, axis=0)
    ID[k] = {n.split("|")[0].split("_")[-1] for n in names}
    info[k] = {"n": len(seqs), "L": int(z.shape[1]), "unique_seqs": len(Z[k]), "unique_ids": len(ID[k]),
               "mean_gap": float((z == GAP).mean())}
    print(f"{k:12s} n={info[k]['n']:6d} L={info[k]['L']} unique={info[k]['unique_seqs']:6d} "
          f"ids={info[k]['unique_ids']:6d} gaps={info[k]['mean_gap']:.3f}")

keys = list(FILES)
ov, idov = {}, {}
for title, sets, store in [("unique sequences", S, ov), ("UniProt IDs", ID, idov)]:
    print(f"\nfraction of {title} of ROW also present in COLUMN")
    print(" " * 12 + "".join(f"{k[:9]:>10s}" for k in keys))
    for a in keys:
        row = []
        for b in keys:
            v = len(sets[a] & sets[b]) / len(sets[a])
            store[f"{a}|{b}"] = v
            row.append(f"{v:10.2f}")
        print(f"{a:12s}" + "".join(row))

PAIRS = [("train2", "test2"), ("train2", "test_4000"), ("train", "test"),
         ("train", "test2"), ("train2", "test")]
pair_stats = {}
print("\ncandidate splits (distances normalised by L)")
for tr, te in PAIRS:
    d = min_hamming(Z[te], Z[tr])
    st = {"n_train": len(Z[tr]), "n_test": len(Z[te]), "test_copies": float((d == 0).mean()),
          "nn_mean": float(d.mean()), "frac_nn_ge_01": float((d >= 0.1).mean()),
          "frac_nn_ge_02": float((d >= 0.2).mean())}
    pair_stats[f"{tr}|{te}"] = st
    print(f"{tr:7s} -> {te:10s} train {st['n_train']:6d} test {st['n_test']:6d} "
          f"copies {100 * st['test_copies']:5.1f}%  nn mean {st['nn_mean']:.3f}  "
          f">=0.1: {100 * st['frac_nn_ge_01']:5.1f}%  >=0.2: {100 * st['frac_nn_ge_02']:5.1f}%")

json.dump({"files": info, "seq_overlap": ov, "id_overlap": idov, "pairs": pair_stats},
          open(OUT / "data_audit.json", "w"), indent=1)
print(f"\nwrote {OUT / 'data_audit.json'}")
