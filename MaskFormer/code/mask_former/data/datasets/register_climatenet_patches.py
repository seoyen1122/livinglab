import os
import numpy as np
from torch.utils.data import Dataset
from detectron2.data import DatasetCatalog, MetadataCatalog
from functools import partial

# ✅ Dataset 클래스 (npy 기반)
class ClimatenetPatchDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.endswith(".npy")
        ])
        self.mask_paths = sorted([
            os.path.join(mask_dir, f)
            for f in os.listdir(mask_dir)
            if f.endswith(".npy")
        ])
        assert len(self.image_paths) == len(self.mask_paths), "Image-mask count mismatch"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.load(self.image_paths[idx])  # (C, H, W)
        mask = np.load(self.mask_paths[idx])    # (H, W)

        return {
            "image": image,
            "sem_seg": mask,
            "height": image.shape[1],
            "width": image.shape[2],
            "file_name": self.image_paths[idx],
            "sem_seg_file_name": self.mask_paths[idx],
            "image_id": idx,
        }


def load_climatenet_patches(split="train"):
    """
    split 은 무시하고, 오직 train patch 폴더만 사용합니다.
    출력 이름은 climatenet_full_train_patches 로 고정됩니다.
    """
    base_path = "/data0/aix23606/soyeon/climatenet_full_train_patches"
    image_dir = os.path.join(base_path, "images")
    mask_dir  = os.path.join(base_path, "masks")

    dataset = ClimatenetPatchDataset(image_dir, mask_dir)
    return [dataset[i] for i in range(len(dataset))]


def register_climatenet_patches():
    """
    Climatenet을 192×192 patch로 잘라놓은 데이터를
    TRAIN: climatenet_full_train_patches
    이름으로 DatasetCatalog 에 등록합니다.
    """
    # Train 패치 등록
    name = "climatenet_full_train_patches"
    base_path = "/data0/aix23606/soyeon/climatenet_full_train_patches"
    image_dir = os.path.join(base_path, "images")
    mask_dir  = os.path.join(base_path, "masks")
    if name in DatasetCatalog.list():
        DatasetCatalog.remove(name)
    DatasetCatalog.register(name, partial(ClimatenetPatchDataset, image_dir, mask_dir))
    meta = MetadataCatalog.get(name)
    meta.stuff_classes  = ["background", "AR"]
    meta.ignore_label   = 255
    meta.evaluator_type = "sem_seg"

    # Test 패치 등록 (추가!)
    name_test = "climatenet_full_test_patches"
    base_path_test = "/data0/aix23606/soyeon/climatenet_full_test_patches"
    image_dir_test = os.path.join(base_path_test, "images")
    mask_dir_test  = os.path.join(base_path_test, "masks")
    if name_test in DatasetCatalog.list():
        DatasetCatalog.remove(name_test)
    DatasetCatalog.register(name_test, partial(ClimatenetPatchDataset, image_dir_test, mask_dir_test))
    meta_test = MetadataCatalog.get(name_test)
    meta_test.stuff_classes  = ["background", "AR"]
    meta_test.ignore_label   = 255
    meta_test.evaluator_type = "sem_seg"

    print(f"✅ Registered patch datasets: {name}, {name_test}")


# ✅ standalone 실행용
if __name__ == "__main__":
    register_climatenet_patches()