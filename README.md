# Flow-based Spectral Kernel Attention (FSKA)

FSKA model code and runnable configurations for three experiments.

## Installation

From the repository root, using Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For GPU training, install a CUDA-compatible PyTorch build. `--device auto` selects
CUDA when available, otherwise CPU.

## Experiments

| Experiment | Tasks / variants | Configuration | Command |
|---|---|---|---|
| Sentence classification | DBPedia, AG News, TREC, RT-Polarity, SST-2, SST-5 | `configs/sentence.yaml` | `python -m experiments.sentence` |
| Full-document classification | SCOTUS, BBC News, 20 Newsgroups, arXiv | `configs/document.yaml` | `python -m experiments.document` |
| Nonstationary toy regression | FSKA on generated nonstationary data | `configs/toy.yaml` | `python -m experiments.toy` |

Each command runs five model seeds with a fixed data/split seed of 42.
Preview settings without training:

```bash
python -m experiments.document --dry-run
```

### Sentence classification

```bash
python -m experiments.prepare_data --datasets dbpedia ag_news trec rt_polarity sst2 sst5
python -m experiments.sentence --device cuda
python -m experiments.sentence --datasets sst2 trec --device cuda
```

Text is lowercased and tokenized by whitespace. Vocabulary is fitted on training
text only (maximum 50,000 entries, minimum frequency 1). Sentence sequences have
a maximum of 256 tokens. SST uses the original sentence-level annotations and
official splits; RT-Polarity uses a stratified 80/20 split. Datasets without an
explicit validation split reserve 10% of their training pool for validation.

### Full-document classification

```bash
python -m experiments.prepare_data --datasets scotus bbc_news twenty_news arxiv
python -m experiments.document --device cuda
python -m experiments.document --datasets bbc_news --device cuda
```

Documents are **not truncated or chunked** (`max_seq_len: null`). Batches are
padded to their longest document, padded keys are masked, and position encodings
extend to the observed length. The data preparation cap is 4,000 training-pool
documents and 2,000 test documents; smaller source splits are retained. Ten percent
of the resulting training pool is held out for validation. Vocabulary uses minimum
frequency 2 and at most 50,000 entries. 20 Newsgroups removes headers, footers and
quoted text.

Use `--batch-size 8` to reduce GPU memory usage while retaining full documents.

### Nonstationary toy regression

```bash
python -m experiments.toy --device cpu
```

The generator creates 500 points with four smoothly changing regimes and
heteroscedastic noise. Data and the 70/30 split are fixed across the five model
seeds. Prediction plots and per-run metrics are generated locally.

## Training and outputs

Settings are provided in `configs/`. Each experiment runs five seeds by default.
Use `--config path/to/config.yaml` for custom settings, or `--seeds 0 --epochs 1`
for a quick check.

Checkpoints, metrics and predictions are saved locally under `outputs/`, separated
by experiment and seed. Set `--output-dir` to choose another directory. Generated
outputs are not included in this repository.

## Data sources and local data

- [RT-Polarity](https://www.cs.cornell.edu/people/pabo/movie-review-data/)
- [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/treebank.html)
- [TREC question classification](https://cogcomp.seas.upenn.edu/Data/QA/QC/)
- [AG News](https://huggingface.co/datasets/fancyzhx/ag_news)
- [DBPedia](https://huggingface.co/datasets/fancyzhx/dbpedia_14)
- [SCOTUS / LexGLUE](https://huggingface.co/datasets/coastalcph/lex_glue)
- [BBC News](https://huggingface.co/datasets/SetFit/bbc-news)
- [20 Newsgroups](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset)
- [arXiv classification](https://huggingface.co/datasets/ccdv/arxiv-classification)

Data is downloaded from the original providers and is not redistributed here.
If an original download server is unavailable, place the extracted SST directory
at `data/raw/stanfordSentimentTreebank/`, or the Cornell files at
`data/raw/rt-polaritydata/`, and rerun the preparation command. Existing raw files
are used without downloading them again.
You can also provide `<data-dir>/<dataset>/data.json`, using the following schema:

```json
{
  "train": {"texts": ["training example"], "labels": [0]},
  "validation": {"texts": ["validation example"], "labels": [0]},
  "test": {"texts": ["test example"], "labels": [0]}
}
```

This is a schema illustration, not a runnable dataset. Labels must be zero-based
integers. Omit `validation` to reserve 10% of the training pool using a stratified
split. Prepared full-document data should respect the same sample caps above.
Use `--data-dir /path/to/data` for a different location.

## Code checks

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

Tests cover model execution, padding, full-document handling and training entry points.

## Citation

```bibtex
@inproceedings{yang2026fska,
  title = {Flow-based Spectral Kernel Learning for Nonstationary Attention},
  author = {Yang, Zi and Li, Ying and Lancho, Alejandro and Larese, Dar{\'i}o C. and Olmos, Pablo M. and Zhang, Michael Minyi},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2026}
}
```
