from argparse import ArgumentParser
from os.path import exists

import h5py
import numpy as np
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

UINT16_FILENAME = "tinystories-train.h5"


def load_naive_uint16(path: str) -> Dataset:
    filepath = f"{path}/{UINT16_FILENAME}"

    if exists(filepath):
        return h5py.File(f"{args.dataset_path}/{UINT16_FILENAME}", "r")
    else:
        return save_naive_uint16(path)


def save_naive_uint16(path: str):
    dataset = load_dataset("roneneldan/TinyStories", cache_dir=path)
    loader = DataLoader(dataset["train"], batch_size=1, num_workers=2)

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token

    # store the input_ids

    filepath = f"{path}/{UINT16_FILENAME}"

    f = h5py.File(filepath, "a")
    if "tokens" not in f:
        tokens_group = f.create_group("tokens")
    tokens_group = f["tokens"]

    modified = False
    for b_idx, batch in enumerate(tqdm(loader, ncols=79)):
        tokenized = tokenizer(batch["text"], return_tensors="np")
        tokens = tokenized["input_ids"][0].astype(np.uint16)
        if str(b_idx) in tokens_group:
            stored = tokens_group[str(b_idx)]
            if (stored != tokens).any():
                modified = True
                tokens_group[str(b_idx)] = tokens
        else:
            modified = True
            tokens_group.create_dataset(str(b_idx), data=tokens)
    f.close()

    if not modified:
        exit(0)

    if "Y" in input("tokenization complete; validate? ").upper():
        f = h5py.File(filepath, "r")
        tokens_group = f["tokens"]

        for b_idx, batch in enumerate(tqdm(loader)):
            tokenized = tokenizer(batch["text"], return_tensors="np")
            tokens = tokenized["input_ids"][0].astype(np.uint16)
            assert (tokens == tokens_group[str(b_idx)]).all()
        f.close()

    return dataset


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-ds", "--dataset_path", type=str, default=".")
    args = parser.parse_args()

    save_naive_uint16(args.dataset_path)
