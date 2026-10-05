import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "transfer_folder" / "DataAttentionDCA" / "jdoms"
CONTACTS = DATA / "jdom.dat"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
SPLIT = OUT / "split"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

# The original split shipped with the project: species-disjoint but not sequence-disjoint
# (79% of its test sequences are exact copies of a training sequence).
TRAIN_ORIG = DATA / "jdoms_bacteria_train2.fasta"
TEST_ORIG = DATA / "jdoms_bacteria_test2.fasta"
# Sequence-disjoint split rebuilt from jdoms_bacteria_filtered.fasta (scripts/make_split.py).
TRAIN_CLEAN = SPLIT / "jdoms_train_c0.2.fasta"
TEST_CLEAN = SPLIT / "jdoms_test_c0.2.fasta"

SPLITS = {"orig": (TRAIN_ORIG, TEST_ORIG), "clean": (TRAIN_CLEAN, TEST_CLEAN)}
TRAIN, TEST = SPLITS["orig"]          # default kept for backwards compatibility


def add_split_arg(parser):
    parser.add_argument("--split", choices=list(SPLITS), default="orig",
                        help="'orig' = shipped species split, 'clean' = sequence-disjoint split")
    return parser


def resolve(split: str):
    return SPLITS[split]


def load_frame(split: str):
    """The PCA frame used in every figure (fitted on the unweighted one-hot test set), cached."""
    from attdca.data import load_msa
    from attdca.pca_frame import LinearFrame
    path = OUT / f"pca_frame_{split}.npz"
    if path.exists():
        return LinearFrame.load(path)
    fr = LinearFrame.fit(load_msa(str(resolve(split)[1])))
    fr.save(path)
    return fr
