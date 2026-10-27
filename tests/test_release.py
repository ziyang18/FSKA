"""Offline functional checks; fixtures are not paper experiments."""
import json
from pathlib import Path

import pytest
import torch

from fska.common import read_config, seed_everything, validate_config
from fska.data.text import TextDataset, collate_text, load_splits, make_vocab
from fska.models.text_model import OursClassifier
from fska.training import classification_metrics, make_vector_model, run_text, run_toy

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]


def tiny(suite, arch="per_head_full"):
    config = read_config(ROOT / "configs" / f"{suite}.yaml")
    config["model"].update(architecture=arch, hidden_dim=16, n_heads=2, n_layers=1,
                            M=4, num_flows=3, num_mixtures=2, dropout=0.0)
    config["training"]["epochs"] = 1
    if suite in ("sentence", "document"):
        config["model"].update(embed_dim=16, pf_dim=16)
        config["training"].update(batch_size=2, scheduler_t_max=1)
    return config


@pytest.mark.parametrize("suite", ["sentence", "document", "toy"])
def test_all_defaults_are_five_independent_runs(suite):
    config = read_config(ROOT / "configs" / f"{suite}.yaml")
    assert config["seeds"] == [0, 1, 2, 3, 4]
    assert config["model"]["architecture"] == "per_head_full"


@pytest.mark.parametrize("arch", ["per_head_full", "hybrid"])
def test_text_forward_backward_and_independent_heads(arch):
    model = OursClassifier(20, 3, embed_dim=16, hidden_dim=16, n_heads=2, n_layers=1,
                           pf_dim=16, M=4, num_flows=3, num_mixtures=2, architecture=arch, dropout=0)
    ids = torch.randint(1, 20, (2, 17))
    logits, penalty = model(ids, torch.ones_like(ids))
    (logits.square().mean() + penalty).backward()
    assert logits.shape == (2, 3) and torch.isfinite(logits).all()
    attn = model.layers[0].self_attention
    assert attn.shared_flow is False
    if arch == "per_head_full":
        assert len(attn.flows) == 2 and attn.flow is None
        assert set(map(id, attn.flows[0].parameters())).isdisjoint(map(id, attn.flows[1].parameters()))
        assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in attn.flows[0].parameters())
    else:
        assert len(attn.hybrid_flow.head_tails) == 2


@pytest.mark.parametrize("arch", ["per_head_full", "hybrid"])
def test_vector_forward_backward(arch):
    model = make_vector_model(tiny("toy", arch), "cpu", 6, 10, 3)
    output, penalty = model(torch.randn(6, 10))
    (output.square().mean() + penalty).backward()
    assert output.shape == (1, 6, 3) and torch.isfinite(output).all()
    attention = model.layers[0].self_attention.frska_layer.frska_attention
    assert attention.architecture == arch
    if arch == "per_head_full":
        assert len(attention.flows) == 2 and attention.flow is None


def test_padding_does_not_change_valid_token_predictions():
    seed_everything(7)
    model = OursClassifier(20, 2, embed_dim=16, hidden_dim=16, n_heads=2, n_layers=1,
                           M=8, num_flows=3, num_mixtures=2, dropout=0)
    model.eval()
    short = torch.tensor([[1, 2, 3, 4]])
    model(short, torch.ones_like(short))  # Initialize ActNorm before comparing.
    padded = torch.tensor([[1, 2, 3, 4, 0, 0, 0]])
    seed_everything(19)
    a, _ = model(short, short.ne(0))
    seed_everything(19)
    b, _ = model(padded, padded.ne(0))
    torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)


def test_full_document_sequence_and_no_dense_attention():
    model = OursClassifier(10, 2, embed_dim=8, hidden_dim=8, n_heads=2, n_layers=1,
                           M=4, num_flows=3, num_mixtures=2, max_seq_len=16, dropout=0)
    ids = torch.ones((2, 1201), dtype=torch.long)
    logits, _ = model(ids, torch.ones_like(ids))
    assert logits.shape == (2, 2)
    assert model.pos_encoder.pe.shape[1] >= 1201
    x = torch.randn(2, 33, 8)
    _, attention, _ = model.layers[0].self_attention(x, x, x)
    assert attention is None


def test_different_seeds_are_not_reset_by_model_constructors():
    states = []
    for seed in (0, 1):
        seed_everything(seed)
        model = make_vector_model(tiny("toy"), "cpu", 4, 10, 1)
        states.append(model.fc_in.weight.detach().clone())
    assert not torch.equal(*states)


def test_macro_f1_for_binary_labels():
    metrics = classification_metrics([0, 0, 0, 1], [0, 0, 0, 0], 2)
    assert metrics["f1_macro"] == pytest.approx(3 / 7)


def test_vocab_and_document_collation():
    vocab = make_vocab(["known known"], min_freq=2)
    assert "testonly" not in vocab
    ds = TextDataset({"texts": ["known " * 1200, "testonly"], "labels": [0, 1]}, vocab)
    ids, mask, labels = collate_text([ds[0], ds[1]])
    assert ids.shape == (2, 1200) and mask[0].sum() == 1200 and mask[1].sum() == 1
    assert ids[1, 0] == 1


def write_fixture(root, dataset, long=False):
    folder = root / dataset
    folder.mkdir(parents=True)
    splits = {}
    for split in ("train", "validation", "test"):
        splits[split] = {"texts": [("alpha " if i % 2 else "beta ") * (1050 if long else 8) for i in range(4)],
                         "labels": [0, 1, 0, 1]}
    (folder / "data.json").write_text(json.dumps(splits))


@pytest.mark.parametrize("suite,dataset", [("sentence", "sst2"), ("document", "bbc_news")])
@pytest.mark.parametrize("arch", ["per_head_full", "hybrid"])
def test_text_training_end_to_end(tmp_path, suite, dataset, arch):
    write_fixture(tmp_path / "data", dataset, long=suite == "document")
    out = tmp_path / "outputs"
    out.mkdir()
    result = run_text(tiny(suite, arch), dataset, 0, tmp_path / "data", out, torch.device("cpu"))
    assert result["epochs_completed"] == 1 and (out / "model.pt").exists()
    assert "f1_macro" in result["metrics"]
    if suite == "document":
        assert result["max_observed_tokens"]["test"] == 1050


@pytest.mark.parametrize("arch", ["per_head_full", "hybrid"])
def test_toy_training_end_to_end(tmp_path, arch):
    config = tiny("toy", arch)
    config["data"]["n_points"] = 24
    out = tmp_path / "out"
    out.mkdir()
    result = run_toy(config, 0, out, torch.device("cpu"))
    assert result["metrics"]["mse"] >= 0 and (out / "prediction.png").exists()


def test_invalid_hybrid_and_document_truncation_rejected():
    config = tiny("document", "hybrid")
    config["model"]["trunk_layers"] = config["model"]["num_flows"]
    with pytest.raises(ValueError):
        validate_config(config)
    config = tiny("document")
    config["data"]["max_seq_len"] = 1024
    with pytest.raises(ValueError):
        validate_config(config)
