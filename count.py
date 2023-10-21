import re
from argparse import ArgumentParser, Namespace

import h5py
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch as pt
from tqdm import tqdm
from transformers import AutoTokenizer

parser = ArgumentParser()
parser.add_argument("-dev", "--device", type=str, default="cpu")
parser.add_argument("-ds", "--dataset_path", type=str, default=".")
parser.add_argument("-l", "--branch_length", type=int, default=9)
parser.add_argument("-n", "--num_seqs", type=int, default=2119719)
parser.add_argument("-o", "--output_dir", type=str, default="stats")
args = parser.parse_args()

dataset = h5py.File(f"{args.dataset_path}/tinystories-train.h5", "r")
extract = lambda i: pt.tensor(dataset["tokens"][str(i)], dtype=pt.int32)


# @title tokenizer (for decoding)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
C = len(tokenizer)


label_range = pt.arange(C, dtype=pt.int32, device=args.device)
toks_counts = pt.zeros(C, dtype=pt.int32, device=args.device)


for i in tqdm(range(args.num_seqs), ncols=79):
    seq = extract(i).to(args.device)
    idxs = seq[:, None] == label_range
    assert idxs.sum() == len(seq)

    toks_counts += idxs.sum(axis=0)[:, None].squeeze()


# @title labelling/graphing config
def classify(x):
    tok = tokenizer.decode(x).strip()
    if tok.isalpha():
        if tok[0].islower():
            return 0
        else:
            return 1
    if re.match(r"^[\W_]+$", tok) is not None:
        return 2

    return 3


selected = toks_counts > 0
X = np.arange(C)[selected.cpu()]
Y = toks_counts[selected].cpu().numpy()
LENGTHS = np.array([len(tokenizer.decode(x)) for x in X])
TYPES = np.array([classify(x) for x in X])

LABELS = ["regular lowercase", "start-word/proper noun", "punctuation", "special"]
SHORTS = [
    "regular",
    "start/proper",
    "punctuation",
    "special",
]
COLOURS = [
    "#3498db",
    "#27ae60",
    "#f39c12",
    "#e056fd",
]
ALPHAS = [0.3, 0.3, 0.7, 0.7]
MARKERS = ["o", "^", "x", "*"]
SIZES = [10, 5, 30, 40]


# @title plotting

ratios = [2, 1, 1]

fig = plt.figure(figsize=(16, 4))
gs = gridspec.GridSpec(1, sum(ratios))

ax1 = plt.subplot(gs[:, : ratios[0]])
for t in range(len(LABELS)):
    ax1.scatter(
        X[TYPES == t],
        Y[TYPES == t],
        c=COLOURS[t],
        alpha=ALPHAS[t],
        label=LABELS[t],
        marker=MARKERS[t],
        s=SIZES[t],
    )
legend = ax1.legend(markerscale=2)
for handle in legend.legend_handles:
    handle.set_alpha(0.85)
ax1.set_xlabel("token ID (as designed for the tokenizer)")
ax1.set_ylabel(f"token frequency \u2265 1")
ax1.set_title("Token Frequencies in the TinyStories Dataset")
ax1.set_yscale("log")


def shuffle(props):
    props[1], props[2] = props[2], props[1]
    return props


ax2 = plt.subplot(gs[:, ratios[0] : ratios[0] + ratios[1]])
vocab_share = [(TYPES == t).sum() for t in range(max(TYPES) + 1)]
ax2.pie(
    shuffle(vocab_share),
    labels=shuffle(SHORTS),
    colors=shuffle(COLOURS),
    autopct="%1.1f%%",
    startangle=140,
)
ax2.set_title("Share of TinyStories Vocabulary")

ax3 = plt.subplot(gs[:, ratios[0] + ratios[1] :])
token_share = [(Y[TYPES == t]).sum() for t in range(max(TYPES) + 1)]
ax3.pie(
    shuffle(token_share),
    labels=SHORTS,
    colors=COLOURS,
    autopct="%1.1f%%",
    startangle=140,
)
ax3.set_title("Share of TinyStories Dataset")

plt.suptitle("Token Analysis of the GPT-generated TinyStories Dataset", fontsize=16)
plt.tight_layout()

plt.savefig("tinystories.pdf")
