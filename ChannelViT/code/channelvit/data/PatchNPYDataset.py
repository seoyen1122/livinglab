# channelvit/data/patch_dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

class PatchNPYDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform_cfg=None, is_train=True, **kwargs):
        self.image_paths = sorted(os.listdir(image_dir))
        self.mask_paths = sorted(os.listdir(mask_dir))
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = None
        if transform_cfg is not None:
            from channelvit import transformations
            self.transform = getattr(transformations, transform_cfg.name)(
                is_train=is_train, **transform_cfg.args
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.load(os.path.join(self.image_dir, self.image_paths[idx]))
        mask = np.load(os.path.join(self.mask_dir, self.mask_paths[idx]))

        image_tensor = torch.from_numpy(image).float()  # (C, H, W)
        mask_tensor = torch.from_numpy(mask).long()     # (H, W)

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, {"label": mask_tensor}

    @staticmethod
    def collate_fn(batch):
        return default_collate(batch)
