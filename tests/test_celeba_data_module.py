import torch
import pytest
from diff_interp.celeba_data_module import CelebADataModule

BATCH_SIZE = 1

@pytest.fixture(scope='module')
def datamodule() -> CelebADataModule:
    datamodule = CelebADataModule(batch_size=BATCH_SIZE)
    datamodule.setup()
    return datamodule

def test_celeba_data_module(datamodule: CelebADataModule):
    
    train_loader = datamodule.train_dataloader()
    image, label = next(iter(train_loader))
    assert image.shape == (BATCH_SIZE, 3, 224, 224)
    assert image.dtype == torch.float32
    assert torch.all(-1 <= image)
    assert torch.all(image <= 1)
    assert label.shape == (BATCH_SIZE, 40)
    # Should be Long
    assert label.dtype == torch.float32
    # Assert label is 0 or 1
    assert torch.all(label >= 0)
    assert torch.all(label <= 1)