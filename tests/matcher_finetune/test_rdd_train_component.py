import torch
from torch import nn

from contrastive_finetuning.train_by_lg_matches import (
    configure_rdd_trainable_component,
    set_rdd_training_mode,
)


class FakeRDD(nn.Module):
    def __init__(self):
        super().__init__()
        self.detector = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3))
        self.descriptor = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3))


def test_rdd_descriptor_mode_freezes_detector_and_trains_descriptor():
    model = FakeRDD()
    configure_rdd_trainable_component(model, True, "descriptor")
    set_rdd_training_mode(model, True, "descriptor")

    assert all(not parameter.requires_grad for parameter in model.detector.parameters())
    assert all(parameter.requires_grad for parameter in model.descriptor.parameters())
    assert not model.detector.training
    assert model.descriptor.training


def test_legacy_rdd_mode_still_trains_all_rdd_weights():
    model = FakeRDD()
    configure_rdd_trainable_component(model, True, "all")
    set_rdd_training_mode(model, True, "all")

    assert all(parameter.requires_grad for parameter in model.parameters())
    assert model.detector.training
    assert model.descriptor.training


def test_frozen_rdd_mode_keeps_every_parameter_frozen():
    model = FakeRDD()
    configure_rdd_trainable_component(model, False, "all")
    set_rdd_training_mode(model, False, "all")

    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert not model.training
