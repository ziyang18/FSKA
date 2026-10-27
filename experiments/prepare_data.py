"""Prepare only the six sentence and four document classification datasets."""
import argparse
import json
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
import urllib.error
import zipfile
from pathlib import Path

from sklearn.model_selection import train_test_split

from fska.common import save_json
from fska.data.text import DOCUMENT_DATASETS, SENTENCE_DATASETS
from fska.data.sentence_sources import load_rt_polarity, load_sst


HF_SOURCES = {
    "trec": ("CogComp/trec", None, "text", "coarse_label"),
    "ag_news": ("fancyzhx/ag_news", None, "text", "label"),
    "dbpedia": ("fancyzhx/dbpedia_14", None, "content", "label"),
    "scotus": ("coastalcph/lex_glue", "scotus", "text", "label"),
    "bbc_news": ("SetFit/bbc-news", None, "text", "label"),
    "arxiv": ("ccdv/arxiv-classification", "default", "text", "label"),
}


def download_file(url, destination):
    """Download atomically; use verified HTTPS with either urllib or system curl."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        partial = Path(tmp) / "download"
        try:
            with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as f:
                shutil.copyfileobj(response, f)
        except urllib.error.URLError:
            curl = shutil.which("curl")
            if curl is None:
                raise
            subprocess.run([curl, "--fail", "--location", "--silent", "--show-error",
                            "--proto", "=https", "--proto-redir", "=https", "--max-time", "120",
                            "--output", str(partial), url], check=True)
        partial.replace(destination)


def subset(texts, labels, cap=None, seed=42):
    texts, labels = list(texts), [int(y) for y in labels]
    if cap is not None and len(texts) > cap:
        keep, _ = train_test_split(list(range(len(texts))), train_size=cap,
                                   random_state=seed, stratify=labels)
        texts, labels = [texts[i] for i in keep], [labels[i] for i in keep]
    return {"texts": texts, "labels": labels}


def download_archive(url, destination):
    """Extract data archives without accepting paths outside the destination."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    def safe(name):
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Unsafe archive path: {name}")
        return path
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "download"
        download_file(url, archive)
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as z:
                for member in z.infolist():
                    target = safe(member.filename)
                    if member.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(z.read(member))
        else:
            with tarfile.open(archive) as t:
                for member in t.getmembers():
                    target = safe(member.name)
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    elif member.isfile():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with t.extractfile(member) as f:
                            target.write_bytes(f.read())
                    else:
                        raise ValueError("Only regular data files and directories are accepted")


def prepare(name, root, seed):
    raw_dir = root / "raw"
    if name == "trec":
        # Read the original question files; no obsolete dataset loading script is needed.
        labels = {name: i for i, name in enumerate(("ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"))}
        result = {}
        for split, filename in (("train", "train_5500.label"), ("test", "TREC_10.label")):
            path = raw_dir / "trec" / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                download_file("https://cogcomp.seas.upenn.edu/Data/QA/QC/" + filename, path)
            texts, targets = [], []
            for line in path.read_bytes().splitlines():
                if not line.strip():
                    continue
                label, text = line.replace(b"\xf0", b" ").decode("utf-8").split(" ", 1)
                texts.append(text)
                targets.append(labels[label.split(":")[0]])
            result[split] = subset(texts, targets)
        return result
    if name in ("sst2", "sst5"):
        if not (raw_dir / "stanfordSentimentTreebank" / "datasetSentences.txt").exists():
            download_archive("https://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip", raw_dir)
        tr, val, te = load_sst(str(raw_dir), binary=name == "sst2")
        return {"train": subset(*tr), "validation": subset(*val), "test": subset(*te)}
    if name == "rt_polarity":
        if not list(raw_dir.glob("rt-polaritydata/**/rt-polarity.pos")):
            download_archive("https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz", raw_dir)
        texts, labels = load_rt_polarity(str(raw_dir))
        tr, te = train_test_split(list(range(len(texts))), test_size=0.2, random_state=seed, stratify=labels)
        return {key: subset([texts[i] for i in ids], [labels[i] for i in ids]) for key, ids in (("train", tr), ("test", te))}
    if name == "twenty_news":
        from sklearn.datasets import fetch_20newsgroups
        raw = {s: fetch_20newsgroups(subset=s, remove=("headers", "footers", "quotes"),
                                     data_home=str(root / "cache")) for s in ("train", "test")}
        return {s: subset(raw[s].data, raw[s].target, 4000 if s == "train" else 2000, seed) for s in raw}
    from datasets import load_dataset
    dataset_id, config, text_key, label_key = HF_SOURCES[name]
    # Do not execute dataset-provided loading scripts. Local JSON is also supported.
    ds = load_dataset(dataset_id, config, cache_dir=str(root / "cache"))
    is_document = name in DOCUMENT_DATASETS
    return {s: subset(ds[s][text_key], ds[s][label_key],
                       (4000 if s == "train" else 2000) if is_document else None, seed)
            for s in ("train", "test")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", required=True, choices=SENTENCE_DATASETS + DOCUMENT_DATASETS)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()
    for name in args.datasets:
        target = args.data_dir / name / "data.json"
        if target.exists():
            print(f"Using existing {target}")
            continue
        data = prepare(name, args.data_dir, args.split_seed)
        save_json(target, data)
        save_json(target.with_name("metadata.json"), {
            "dataset": name, "split_seed": args.split_seed,
            "prepared_sizes": {s: len(v["texts"]) for s, v in data.items()},
            "source": "https://cogcomp.seas.upenn.edu/Data/QA/QC/" if name == "trec"
                      else HF_SOURCES.get(name, ("original release",))[0]})
        print(f"Prepared {target}")


if __name__ == "__main__":
    main()
