import torch
import pytest
from diff_interp.vit_model import ViTModel, NUM_CLASSES, INPUT_SHAPE

@pytest.fixture(scope='module')
def model() -> ViTModel:
    model = ViTModel()
    return model

def test_vit_model(model: ViTModel):
    dummy_input = torch.randn(1, *INPUT_SHAPE)
    output = model(dummy_input)
    assert output.shape == (1, NUM_CLASSES)