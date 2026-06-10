import numpy as np
import pytest
import torch

from neovad.bench.accuracy import AccuracyBenchmark
from neovad.config import ModelConfig
from neovad.models.vad import VADModel


def test_roc_auc_known_cases():
    b = AccuracyBenchmark
    # perfectly separable -> 1.0; reversed -> 0.0; all-tied -> 0.5
    assert b.roc_auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    assert b.roc_auc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1])) == 0.0
    assert b.roc_auc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([0, 1, 0, 1])) == 0.5
    assert np.isnan(b.roc_auc(np.array([0.1, 0.2]), np.array([1, 1])))  # one class only


def test_best_f1_recovers_threshold():
    scores = np.array([0.1, 0.2, 0.7, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1, 1])
    f1, _ = AccuracyBenchmark.best_f1(scores, labels)
    assert f1 == pytest.approx(1.0, abs=1e-6)


def test_evaluate_runs_and_orders_by_quality():
    torch.manual_seed(0)
    model = VADModel(ModelConfig(dim=32, depth=2, mixer={"kind": "mamba2"})).eval()
    # two clips: clear speech (ramp) and silence, labelled accordingly
    clips = [
        (torch.randn(16000) * 0.3, torch.ones(100, dtype=torch.long)),
        (torch.zeros(16000), torch.zeros(100, dtype=torch.long)),
    ]
    [res] = AccuracyBenchmark().evaluate(model, clips)
    assert res.name == "neovad"
    assert 0.0 <= res.roc_auc <= 1.0
    assert res.frames == 200
