import copy
import torch
import numpy as np
from PIL import Image
from detectron2.data.transforms import ResizeShortestEdge
from detectron2.data.transforms.transform import ResizeTransform
from detectron2.structures import Instances, BitMasks
from detectron2.data import transforms

class NetCDFSemanticMapper:
    def __init__(self, cfg, is_train=True):
        self.cfg = cfg
        self.is_train = is_train
        self.target_height = cfg.INPUT.MIN_SIZE_TRAIN[0]
        # self.target_width = cfg.INPUT.MIN_SIZE_TRAIN[0] # patches 일때때
        self.target_width = cfg.INPUT.MAX_SIZE_TRAIN
        self.ignore_label = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE

        # === config에서 mean/std 읽기
        self.pixel_mean = np.array(cfg.MODEL.PIXEL_MEAN).reshape(-1, 1, 1)  # (C,1,1)
        self.pixel_std = np.array(cfg.MODEL.PIXEL_STD).reshape(-1, 1, 1)

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)

        image = dataset_dict["image"]  # (C, H, W)
        sem_seg = dataset_dict["sem_seg"]  # (H, W)

        # === Normalize BEFORE resize
        image = (image - self.pixel_mean) / self.pixel_std  # broadcasting

        _, H_img, W_img = image.shape
        H_seg, W_seg = sem_seg.shape
        assert H_img == H_seg and W_img == W_seg, "Resolution mismatch"

        ## patches
        # tfm = ResizeTransform(H_img, W_img, self.target_height, self.target_width, interp=Image.BILINEAR)
        
        # if self.is_train:
        #     image = np.transpose(image, (1, 2, 0))
        #     image = tfm.apply_image(image)
        #     image = np.transpose(image, (2, 0, 1))
        #     sem_seg = sem_seg.astype(np.float32)
        #     sem_seg = tfm.apply_image(sem_seg, interp=Image.NEAREST)
        #     sem_seg = sem_seg.astype(np.int64)

        ## patches 말고 whole
        if self.is_train:
            # === ResizeShortestEdge 적용 ===
            # 1. 이미지 형식 변환 (C, H, W) → (H, W, C)
            image = np.transpose(image, (1, 2, 0))
            
            # 2. 리사이즈 변환 객체 생성
            resize_aug = ResizeShortestEdge(
                short_edge_length=[self.cfg.INPUT.MIN_SIZE_TRAIN[0]],
                max_size=self.cfg.INPUT.MAX_SIZE_TRAIN,
                sample_style="choice"
            )
            transform = resize_aug.get_transform(image)
            
            # 3. 이미지 변환 적용
            image = transform.apply_image(image)
            image = np.transpose(image, (2, 0, 1))  # (C, H, W) 복원
            
            # 4. 세그멘테이션 마스크 변환
            sem_seg = sem_seg.astype(np.float32)
            sem_seg = transform.apply_segmentation(sem_seg)
            sem_seg = sem_seg.astype(np.int64)

        image_tensor = torch.as_tensor(image).float()
        sem_seg_tensor = torch.as_tensor(sem_seg).long()

        height, width = sem_seg.shape
        instances = Instances((height, width))
        masks = []
        classes = []
        for label in np.unique(sem_seg):
            if label == self.ignore_label:
                continue
            masks.append(torch.from_numpy(sem_seg == label))
            classes.append(label)

        if masks:
            instances.gt_masks = BitMasks(torch.stack(masks))
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
        else:
            instances.gt_masks = BitMasks(torch.empty((0, height, width), dtype=torch.bool))
            instances.gt_classes = torch.empty((0,), dtype=torch.int64)

        dataset_dict["image"] = image_tensor
        dataset_dict["sem_seg"] = sem_seg_tensor
        dataset_dict["instances"] = instances

        if self.is_train:
            print(f"[DEBUG] Normalized image min/max: {image.min():.2f}, {image.max():.2f}")
            print(f"[DEBUG] file: {dataset_dict.get('file_name', 'N/A')}")

        return dataset_dict