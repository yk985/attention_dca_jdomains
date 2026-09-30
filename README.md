# AttentionDCA on J-domains — remastered

Potts model whose couplings are a factored attention layer,
`J_ij(a,b) = Σ_h softmax_j(Q_hᵀK_h)_ij V_h(a,b)`, trained by weighted pseudo-likelihood on
the bacterial J-domain MSA, then sampled with the pseudo-likelihood conditionals.

The original code is kept untouched in `transfer_folder/` (data lives in
`transfer_folder/DataAttentionDCA/jdoms/`).

## Layout

| path | content |
|---|---|
| `attdca/data.py` | FASTA reading, encoding (`ACDEFGHIKLMNPQRSTVWY-`, unknown → gap), dedup, reweighting |
| `attdca/model.py` | `AttentionPotts`: attention, couplings, vectorised weighted −log PL + L2 |
| `attdca/train.py` | Adam, per-batch renormalised weights, validation split, early stopping |
| `attdca/sampler.py` | GPU pseudo-likelihood sampler over many parallel chains; energy / PLL scores |
| `attdca/metrics.py` | frequencies, connected correlations, comparison stats, Hamming, τ_int, contact PPV |
| `attdca/split.py` | greedy identity clustering + sequence-disjoint train/test split |
| `scripts/` | `audit_data.py`, `make_split.py`, `train_model.py`, `sweep_table.py`, `beta_scan.py`, `sample_model.py`, `evaluate.py` |
| `tests/test_core.py` | vectorised code vs naive loops; exact-Gibbs check of the sampler |
| `outputs/` | models, J, generated FASTA/npy, metrics JSON, `figures/` |
| `report/` | `report.tex` → `report.pdf` (full write-up); `paper.tex` → `paper.pdf` (2-column paper version, same numbers) |

## Reproduce

`--split clean` uses the rebuilt sequence-disjoint split (the default `orig` is the supplied one).

```bash
python -m pytest tests -q
python scripts/audit_data.py                       # overlap between every supplied file
python scripts/make_split.py --thresh 0.2          # -> outputs/split/jdoms_{train,test}_c0.2.fasta
# sweep: loop train_model.py over H, d, lambda with --tag sweep_H{H}_d{d}_l{lambda}, then
python scripts/sweep_table.py
S="--split clean"
python scripts/train_model.py  $S --H 128 --d 10 --lambd 1e-3 --epochs 3000 --patience 50 --tag clean
python scripts/beta_scan.py    $S --tag clean      # beta* = 1.05: <E_gen> = weighted <E_train>
python scripts/sample_model.py $S --tag clean --beta 1.0
python scripts/sample_model.py $S --tag clean --beta 1.05
python scripts/evaluate.py     $S --tag clean --betas 1.0,1.05
python scripts/make_report_numbers.py --tag clean --compare main
cd report && latexmk -pdf report.tex && latexmk -pdf paper.tex
```

Chosen model: H=128, d=10, λ=1e-3 (picked on the supplied split from a 14-config sweep;
H=256 gained only 0.29 nats).

Headline results on the **clean split** (β=1 / β*=1.05): r(f_i)=0.990/0.992,
r(C_ij)=0.871/0.859 (train-vs-test ceiling 0.944) → **0.918/0.918 restricted to the PCA
clusters the sampler covers** (ceiling 0.941); mean distance to nearest training seq 0.42/0.39
(natural held-out: 0.18); gen–gen 0.68/0.65 (test–test 0.60); τ_int(E)=6.5 sweeps;
contact PPV 1.00 at L/2. Pseudo-likelihood per sequence 71.1 (train) vs 78.7 (test) — a real
11 % generalisation gap. Weak point: one minor HscB-like PCA cluster holds 14.1 % of natural
weight but 5.1 %/3.2 % of samples.

Slope vs r: they are linked by `slope = r · σ_gen/σ_nat` — r is tightness, slope is amplitude.
Neither should be read against 1: sampling noise in the natural C_ij (reliability 0.94 from a
half-split of the test set) attenuates any least-squares slope, so the disattenuated values are
0.83 (β=1), 0.91 (β*) against a 0.94 ceiling.

## Changes relative to the original code

* `trainer(..., lambd=...)` was ignored (model always built with `lambd=0.001`); fixed.
* Old sampler stored the state after **every single-site update** as a new sample, so
  consecutive "sequences" differed at ≤ 1 site; now each sample is the end state of an
  independent chain run for ≥ 20 τ_int sweeps.
* Old sampler: pure-Python loops over sites and amino acids (0.6–28 ms per update);
  now all chains are updated at once on the GPU (~0.3 µs per chain-update).
* Unknown symbols were mapped to a 22nd state (`q=22` possible); now mapped to gap.
* Missing `dcascore` import removed; hard-coded absolute paths removed.
* Best validation model is restored at the end of training (was: last epoch).

## Data: why the supplied split was replaced

`scripts/audit_data.py` compares every supplied bacterial file (results in
`outputs/data_audit.json`):

* `jdoms_bacteria_filtered.fasta` (36 713 unique) is a **superset** — every unique sequence of
  every other bacterial file is in it.
* `filtered2` = `train2` ∪ `test2` exactly.
* `train2` / `test2` have disjoint UniProt IDs but **78.8 % of test2 is byte-identical to a
  train2 sequence** (split made by species; the same J-domain recurs across species). No supplied
  pair is clean — the best, `train`→`test2`, still has 17 % copies with only 7 695 unique training
  sequences.

`scripts/make_split.py` therefore rebuilds the split from the superset: greedy CD-HIT-style
clustering at 80 % identity (10 942 clusters, 6 767 singletons), whole clusters assigned to one
side so none straddles the split, then **only exact copies of a training sequence removed**
(`--drop exact`, the default). Result: **29 312 train / 7 401 test, no exact copies**, median
test-to-nearest-train distance 0.175.

Test sequences that merely *resemble* a training sequence are kept on purpose: greedy clustering
separates cluster representatives by 0.2 but not every cross-cluster pair, so 61 % of the test set
still lies within 0.2 of some training sequence. Dropping those (`--drop threshold`, which leaves
2 917 test sequences) would keep only the family's isolated corners and misrepresent the density
the generative model has to reproduce. What the split guarantees is the thing a likelihood needs:
no test sequence was seen in training.
