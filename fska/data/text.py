"""Training-only vocabulary and dynamic batching, including untruncated documents."""
import json
from collections import Counter
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split

from fska.common import fingerprint

SENTENCE_DATASETS = ("dbpedia", "ag_news", "trec", "rt_polarity", "sst2", "sst5")
DOCUMENT_DATASETS = ("scotus", "bbc_news", "twenty_news", "arxiv")
NUM_CLASSES = dict(zip(SENTENCE_DATASETS + DOCUMENT_DATASETS, (14, 4, 6, 2, 2, 5, 13, 5, 20, 11)))


def tokens(text):
    # Empty documents remain explicit UNK observations, never all-padding examples.
    return str(text).lower().split() or ["<UNK>"]


def make_vocab(texts, min_freq=1, max_size=50000):
    counts = Counter(token for text in texts for token in tokens(text))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, count in counts.most_common():
        if count >= min_freq and token not in vocab and len(vocab) < max_size:
            vocab[token] = len(vocab)
    return vocab


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, split, vocab, max_seq_len=None):
        self.ids = []
        self.labels = split["labels"]
        for text in split["texts"]:
            words = tokens(text)
            if max_seq_len is not None:
                words = words[:max_seq_len]
            self.ids.append(torch.tensor([vocab.get(w, 1) for w in words], dtype=torch.long))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return self.ids[index], int(self.labels[index])


def collate_text(batch):
    ids, labels = zip(*batch)
    padded = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=0)
    return padded, padded.ne(0).long(), torch.tensor(labels, dtype=torch.long)


def load_splits(data_dir, dataset, split_seed=42):
    """Read prepared JSON. Reserve validation before fitting vocabulary, with a fixed seed."""
    path = Path(data_dir) / dataset / "data.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run python -m experiments.prepare_data --datasets {dataset}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    for name, split in raw.items():
        if name not in ("train", "validation", "test"):
            continue
        if len(split["texts"]) != len(split["labels"]) or not split["texts"]:
            raise ValueError(f"Invalid/empty {name} split in {path}")
        if any(type(y) is not int or not 0 <= y < NUM_CLASSES[dataset] for y in split["labels"]):
            raise ValueError(f"Labels must be integers in [0, {NUM_CLASSES[dataset]})")
    if "validation" not in raw:
        tr = raw["train"]
        train_indices, val_indices = train_test_split(list(range(len(tr["texts"]))), test_size=0.1,
                                                      random_state=split_seed, stratify=tr["labels"])
        raw["train"] = {key: [tr[key][i] for i in train_indices] for key in ("texts", "labels")}
        raw["validation"] = {key: [tr[key][i] for i in val_indices] for key in ("texts", "labels")}
    result = {s: raw[s] for s in ("train", "validation", "test")}
    return result, fingerprint(result)

