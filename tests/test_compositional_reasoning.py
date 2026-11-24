"""Tests for the two-stage compositional reasoning benchmark."""

import random

import numpy as np
import pytest
import torch

from compositional_reasoning.dataset import (
    NUM_CLASSES,
    Stage1Dataset,
    Stage2Dataset,
    compute_panel_labels,
    generate_stage2_sample,
)
from compositional_reasoning.model import POS_TYPES, ViTClassifier
from compositional_reasoning.train import parse_args, train


class TestStage1:
    def test_item(self):
        item = Stage1Dataset(num_samples=8)[3]
        assert item["pixel_values"].shape == (3, 224, 224)
        assert 0 <= item["label"] < NUM_CLASSES

    def test_train_labels_are_balanced(self):
        ds = Stage1Dataset(num_samples=16, split="train")
        assert [ds.render(i)[1] for i in range(16)] == [i % 8 for i in range(16)]


class TestStage2:
    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_grid_follows_its_rule(self, n):
        for seed in range(50):
            sample = generate_stage2_sample(random.Random(seed), n=n)
            rows = [list(r) for r in sample.grid_angles]
            assert rows[-1][-1] is None
            rows[-1][-1] = sample.label
            if sample.rule == "rotation":
                steps = {(r[c + 1] - r[c]) % 8 for r in rows for c in range(n - 1)}
                assert len(steps) == 1
            else:
                assert all(r == r[::-1] for r in rows)

    def test_2x2_uses_rotation_only(self):
        rules = {generate_stage2_sample(random.Random(s), n=2).rule for s in range(40)}
        assert rules == {"rotation"}

    def test_3x3_uses_both_rules(self):
        rules = {generate_stage2_sample(random.Random(s), n=3).rule for s in range(40)}
        assert rules == {"rotation", "mirror"}

    def test_deterministic(self):
        ds = Stage2Dataset(n=3, num_samples=4)
        assert torch.equal(ds[2]["pixel_values"], ds[2]["pixel_values"])

    def test_train_and_val_are_disjoint(self):
        def images(split):
            ds = Stage2Dataset(n=3, num_samples=200, split=split)
            return {np.asarray(ds.render(i).image).tobytes() for i in range(200)}

        assert not images("train") & images("val")

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_panel_labels(self, n):
        labels = compute_panel_labels(n)
        assert labels.shape == (196,)
        assert labels.unique().tolist() == list(range(n * n))
        assert labels.view(14, 14)[0, 0] == 0 and labels.view(14, 14)[-1, -1] == n * n - 1


class TestModel:
    @pytest.mark.parametrize("pos_type", POS_TYPES)
    def test_forward(self, pos_type):
        model = ViTClassifier(size="tiny", pos_type=pos_type)
        assert model(torch.randn(2, 3, 224, 224)).shape == (2, 8)

    def test_adapter_parameter_counts(self):
        """ViT-B, 3x3 grid: LoRA (r=8, Q/K) = 0.295M, OPRO (rho=8) = 9,216."""
        model = ViTClassifier(size="base", pos_type="rope")
        model.add_lora(rank=8)
        model.add_opro(num_panels=9, rank=8)
        lora = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
        opro = sum(p.numel() for p in model.operator.parameters())
        assert lora == 12 * 2 * 2 * 8 * 768 == 294_912
        assert opro == 9 * 64 * 8 * 2 == 9_216

    def test_freeze_backbone(self):
        model = ViTClassifier(size="tiny", pos_type="comrope")
        model.add_lora()
        model.add_opro(num_panels=9)
        model.freeze_backbone()
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        assert trainable and all(
            n.startswith(("classifier.", "operator.")) or "lora_" in n for n in trainable
        )
        assert "operator.L" in trainable and "classifier.weight" in trainable

    @pytest.mark.parametrize("pos_type", POS_TYPES)
    def test_adapters_start_as_identity(self, pos_type):
        """LoRA (zero up-projection) and OPRO (zero init) leave the backbone output unchanged."""
        torch.manual_seed(0)
        model = ViTClassifier(size="tiny", pos_type=pos_type).eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            reference = model(x)
            model.add_lora()
            model.add_opro(num_panels=9)
            out = model(x, compute_panel_labels(3))
        assert torch.allclose(out, reference, atol=1e-5)

    def test_opro_receives_gradient(self):
        model = ViTClassifier(size="tiny", pos_type="rope")
        model.add_opro(num_panels=4)
        model(torch.randn(2, 3, 224, 224), compute_panel_labels(2)).sum().backward()
        assert model.operator.L.grad.abs().sum() > 0

    def test_load_backbone_keeps_new_classifier(self):
        source = ViTClassifier(size="tiny", pos_type="liere")
        target = ViTClassifier(size="tiny", pos_type="liere")
        head = target.classifier.weight.clone()
        target.load_backbone(source.state_dict())
        assert torch.equal(target.blocks[0].attn.query.weight, source.blocks[0].attn.query.weight)
        assert torch.equal(target.positional.generators, source.positional.generators)
        assert torch.equal(target.classifier.weight, head)


def test_two_stage_training_runs(tmp_path):
    """A few optimization steps of both stages on the tiny model (CPU)."""
    common = ["--size", "tiny", "--pos_type", "rope", "--batch_size", "4", "--num_workers", "0",
              "--train_samples", "8", "--val_samples", "4", "--max_steps", "2", "--val_interval", "2"]
    stage1 = tmp_path / "stage1"
    train(parse_args(["--stage", "1", "--output_dir", str(stage1)] + common))
    result = train(parse_args(["--stage", "2", "--output_dir", str(tmp_path / "stage2"),
                               "--pretrain_ckpt", str(stage1 / "last.pt"), "--grid_size", "3",
                               "--use_lora", "--use_opro"] + common))
    assert result["params"]["opro"] > 0 and result["params"]["lora"] > 0
    assert 0.0 <= result["final_val_acc"] <= 100.0
