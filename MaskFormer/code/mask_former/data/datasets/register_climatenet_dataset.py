import os
import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset
from detectron2.data import DatasetCatalog, MetadataCatalog
from functools import partial

# ✅ Dataset 클래스
class NetCDFSegmentationDataset(Dataset):
    def __init__(self, nc_dir, variables):
        self.variables = variables
        self.nc_files = []

        for fname in sorted(os.listdir(nc_dir)):
            if not fname.endswith(".nc"):
                continue

            path = os.path.join(nc_dir, fname)
            ds = xr.open_dataset(path)
            label_set = set(np.unique(ds["LABELS"].values))

            if label_set.issubset({0, 1}):  # TC-only or background-only
                print(f"[SKIP INIT] TC-only or empty file skipped: {fname}")
                continue

            self.nc_files.append(path)

    def __len__(self):
        return len(self.nc_files)

    def __getitem__(self, idx):
        ds = xr.open_dataset(self.nc_files[idx])
        image_np = np.stack([ds[var].isel(time=0).values.astype(np.float32) for var in self.variables], axis=0)
        mask_np = ds["LABELS"].values.astype(np.int64)

        # TC = 0 (배경), AR = 1
        mask_np[mask_np == 1] = 0  # TC 무시
        mask_np[mask_np == 2] = 1  # AR = 1

        # ✅ 디버깅 코드 추가: 마스크 라벨 분포 확인
        unique, counts = np.unique(mask_np, return_counts=True)
        print(f"[DEBUG] Sample {idx} label distribution after remapping: {dict(zip(unique, counts))}")

        return {
            "image": image_np,
            "sem_seg": mask_np,
            "height": image_np.shape[1],
            "width": image_np.shape[2],
            "file_name": self.nc_files[idx],
            "sem_seg_file_name": self.nc_files[idx],
            "image_id": idx,
        }


def load_climatenet_dataset(split="train"):
    """
    split: "train", "test" 또는 "val_devided"
    해당 split 이름에 맞는 NC 폴더를 읽어서 리스트를 반환합니다.
    """
    variables = ["U850", "V850", "PSL", "TMQ"]

    if split == "val_devided":
        nc_dir = "/data0/aix23606/soyeon/climatenet_val_devided"
    else:
        # split 이 "train" 또는 "test" 일 때
        # → "/data0/aix23606/soyeon/climatenet_full_train"
        #   "/data0/aix23606/soyeon/climatenet_full_test"
        base = "/data0/aix23606/soyeon/climatenet_full_"
        nc_dir = base + split

    dataset = NetCDFSegmentationDataset(nc_dir, variables)
    # Dataset.__getitem__ 에 None 리턴 없음이 보장되므로 단순히 전부 반환
    return [dataset[i] for i in range(len(dataset))]

def register_climatenet():
    """
    Climatenet 원본 NC 파일(full resolution)을
    TRAIN: climatenet_full_train
    TEST : climatenet_full_test
    VAL  : climatenet_val_devided
    이름으로 DatasetCatalog 에 등록합니다.
    """
    for split in ["train", "test", "val_devided"]:
        # config 에서는 TRAIN=climatenet_full_train_patches, TEST=climatenet_full_test 을 쓰므로
        # 여기서는 full 해상도 데이터셋만 이름 맞춰 등록
        if split == "train":
            name = "climatenet_full_train"
        elif split == "test":
            name = "climatenet_full_test"
        else:  # val_devided
            name = "climatenet_val_devided"

        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
        DatasetCatalog.register(name, partial(load_climatenet_dataset, split=split))

        meta = MetadataCatalog.get(name)
        meta.stuff_classes  = ["background", "AR"]
        meta.ignore_label   = 255
        meta.evaluator_type = "sem_seg"

        print(f"Registered dataset: {name}")

# ✅ 이건 그냥 standalone 실행용
if __name__ == "__main__":
    register_climatenet()
