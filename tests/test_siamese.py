import unittest
import torch
from pathlib import Path
from PIL import Image
import numpy as np

from src.siamese import (
    SiameseSignatureNet,
    ContrastiveLoss,
    SignaturePairDataset,
    SiameseSignatureVerifier
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_siamese_net_architecture():
    model = SiameseSignatureNet(embedding_dim=128, pretrained=False)
    x1 = torch.randn(2, 3, 224, 224)
    x2 = torch.randn(2, 3, 224, 224)
    z1, z2 = model(x1, x2)

    assert z1.shape == (2, 128)
    assert z2.shape == (2, 128)

    # Verify L2 normalization
    norm1 = torch.norm(z1, p=2, dim=1)
    norm2 = torch.norm(z2, p=2, dim=1)
    assert torch.allclose(norm1, torch.ones_like(norm1), atol=1e-5)
    assert torch.allclose(norm2, torch.ones_like(norm2), atol=1e-5)


def test_contrastive_loss():
    loss_fn = ContrastiveLoss(margin=1.0)
    # Identical embeddings with positive label => loss should be 0
    z1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    z2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = torch.tensor([[1.0], [1.0]])
    loss = loss_fn(z1, z2, target)
    assert abs(float(loss)) < 1e-5


def test_signature_pair_dataset():
    data_dir = PROJECT_ROOT / "data" / "signature_forgery" / "global_test"
    if not data_dir.exists():
        return

    ds = SignaturePairDataset(data_dir, pairs_per_writer=10, seed=42)
    assert len(ds) > 0
    img1, img2, label = ds[0]
    assert isinstance(img1, Image.Image)
    assert isinstance(img2, Image.Image)
    assert float(label) in (0.0, 1.0)


def test_siamese_verifier_interface():
    verifier = SiameseSignatureVerifier(similarity_threshold=0.65)
    img_ref = Image.new("RGB", (224, 224), color="white")
    img_cand = Image.new("RGB", (224, 224), color="white")

    result = verifier.verify(img_ref, img_cand)
    assert "similarity_percent" in result
    assert "is_genuine" in result
    assert "decision" in result
    assert "latency_ms" in result
    assert result["similarity_percent"] >= 0.0
    assert result["similarity_percent"] <= 100.0
