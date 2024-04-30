import os
import pytorch_lightning as pl
from torchvision import transforms
from torchvision.datasets import CelebA
from diff_interp.variables import DATA_DIR
from torch.utils.data import DataLoader

# get number of cpu cores
NUM_CORES = os.cpu_count()

class CelebADataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir: str = DATA_DIR):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        
        # TODO: dims
        # TODO: n_classes

    @property 
    def transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # TODO: Unsure if we should be resizing as this changes aspect ratio
            transforms.Resize((224, 224))
        ])
    
    @property
    def target_transform(self):
        # cast label to float
        return lambda x: x.float()

    def prepare_data(self):
        # download
        for split in ['train', 'valid', 'test']:
            CelebA(
                self.data_dir, 
                split=split,
                target_type='attr',
                download=True
            )

    def setup(self, stage=None):
        self.train_dataset = CelebA(self.data_dir, split='train', transform=self.transform, target_transform=self.target_transform)
        self.val_dataset = CelebA(self.data_dir, split='valid', transform=self.transform, target_transform=self.target_transform)
        self.test_dataset = CelebA(self.data_dir, split='test', transform=self.transform, target_transform=self.target_transform)


    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=NUM_CORES)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=NUM_CORES)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=NUM_CORES)


