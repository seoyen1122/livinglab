import os
import torch
from torch.utils.data import Dataset
import numpy as np
import xarray as xr
from omegaconf import DictConfig
from channelvit import transformations
from torch.utils.data._utils.collate import default_collate


class ARDataset(Dataset):
    def __init__(self, root: str, split: str, transform_cfg, is_train: bool = True, channels: list = None, name=None, loader=None, **kwargs):
        self.root = root
        self.split = split
        self.is_train = is_train
        self.transform = getattr(transformations, transform_cfg.name)(
            is_train=is_train, **transform_cfg.args
        )
        self.variables = channels if channels is not None else ["TMQ", "U850", "V850", "PSL"]

        # 파일 목록 로드
        split_file = os.path.join(root, f"{split}.txt")
        with open(split_file, 'r') as f:
            self.file_list = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]  # file_list는 절대경로 or root 상대경로일 수 있음
        if not os.path.isabs(file_path):
            file_path = os.path.join(self.root, file_path)

        if file_path.endswith(".npz"):
            # .npz 로드 → dict-like (NpzFile)
            arr = np.load(file_path)

            # 1) 4개 채널 이미지 (각 arr[var]는 (H, W)) → (4, H, W)로 쌓기
            try:
                img = np.stack([arr[var] for var in self.variables], axis=0)  # (4, H, W)
            except KeyError as e:
                raise KeyError(f"{file_path} 내부에 {e} 키가 없습니다. .npz 저장 시 TMQ/U850/V850/PSL 키 확인 필요.")

            # 2) 이진 레이블 → arr["LABEL"] (이미 AR=1, 나머지=0으로 저장되었음)
            if "LABEL" not in arr:
                raise KeyError(f"{file_path} 내부에 'LABELS' 키가 없습니다. .npz 저장 시 'LABEL' 키 확인 필요.")
            label_np = arr["LABEL"]  # (H, W), uint8 or int

            # 3) NumPy → TorchTensor, transform 적용
            img_tensor = torch.tensor(img, dtype=torch.float32)  # (4, H, W)
            img_tensor = self.transform(img_tensor)

            label_tensor = torch.tensor(label_np, dtype=torch.long)  # (H, W)

        else:
            # .nc 파일 처리
            ds = xr.open_dataset(file_path)
            img = np.stack([np.squeeze(ds[var].values) for var in self.variables], axis=0)
            img_tensor = torch.tensor(img, dtype=torch.float32)
            img_tensor = self.transform(img_tensor)

            label = ds["LABELS"].values.astype(np.int64)
            label = np.where(label == 2, 1, 0)  # TC를 배경으로
            label_tensor = torch.tensor(label).long()

        return img_tensor, {"label": label_tensor}

    @staticmethod
    def collate_fn(batch):
        return default_collate(batch)

