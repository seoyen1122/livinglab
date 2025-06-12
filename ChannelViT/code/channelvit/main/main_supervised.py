import logging
import boto3
import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, open_dict, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from channelvit.meta_arch.supervised import MetricsPlotCallback


import os
import numpy as np
from PIL import Image

import channelvit.data as data  # ARDataset이 여기에 정의되어 있음
from channelvit.meta_arch import Supervised

# supervised.py 맨 위 import 아래에 추가

import torch.nn as nn

def get_train_loader(cfg: DictConfig):
    # Case 1: 단일 데이터셋 (cfg.train_data는 _target_ 키를 가진 config 객체)
    if "_target_" in cfg.train_data:
        print("Single training dataset detected")
        train_data_cfg = cfg.train_data

        train_data = hydra.utils.instantiate(
            train_data_cfg,
            is_train=True,
            transform_cfg=cfg.train_transformations,
        )
        train_loader = DataLoader(
            train_data, **train_data_cfg.loader, collate_fn=train_data.collate_fn
        )

        with open_dict(cfg):
            cfg.train_data.loader.num_batches = (
                len(train_loader) // cfg.trainer.devices + 1
            )

        return train_loader

    # Case 2: 여러 데이터셋이 딕셔너리로 들어옴
    else:
        print("Multiple training datasets detected")
        train_loaders = {}
        len_loader = None
        batch_size = 0

        for name, train_data_cfg in cfg.train_data.items():
            print(f"Loading {train_data_cfg.name}")

            train_data = hydra.utils.instantiate(
                train_data_cfg,
                is_train=True,
                transform_cfg=cfg.train_transformations,
            )

            train_loader = DataLoader(
                train_data, **train_data_cfg.loader, collate_fn=train_data.collate_fn
            )
            train_loaders[name] = train_loader

            print(f"Dataset {name} has length {len(train_loader)}")

            if len_loader is None:
                len_loader = len(train_loader)
            else:
                len_loader = max(len_loader, len(train_loader))

            batch_size = train_data_cfg.loader.batch_size

        with open_dict(cfg):
            # 여러 데이터로더가 있을 때 공통 batch_size / num_batches 설정
            cfg.train_data.loader = {}
            cfg.train_data.loader.num_batches = len_loader // cfg.trainer.devices + 1
            cfg.train_data.loader.batch_size = batch_size

        return train_loaders


@hydra.main(version_base=None, config_path="../config", config_name="main_supervised")
def train(cfg: DictConfig) -> None:
    """
    - checkpoint가 None이면 학습을 수행
    - checkpoint가 지정되어 있으면 바로 predict(cfg)로 넘어감
    """
    if cfg.checkpoint is not None:
        # checkpoint가 지정되어 있으면 inference 모드로 넘어감
        return predict(cfg)

    print("cfg.train_data:", cfg.train_data)

    # ─── 1) Train DataLoader 준비 ──────────────────────────────────────────────────
    train_loader = get_train_loader(cfg)

    # ─── 2) 모델 정의 ─────────────────────────────────────────────────────────────
    model = Supervised(cfg)
    print(model) 
    print(hasattr(model, "decoder"), hasattr(model, "channel_fuser"))


    logger = TensorBoardLogger(save_dir="snapshots/lightning_logs", name="channelvit")
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(dirpath=cfg.trainer.default_root_dir, save_top_k=1),
        MetricsPlotCallback(),
    ]

    trainer = pl.Trainer(
        logger=logger,
        strategy="ddp_find_unused_parameters_true",
        callbacks=callbacks,
        **cfg.trainer
    )

    # ─── 3) 학습 수행 ────────────────────────────────────────────────────────────
    trainer.fit(model=model, train_dataloaders=train_loader, ckpt_path=cfg.checkpoint)

    # ─── 4) 학습 완료 후 가장 좋은 체크포인트로 predict 호출 ───────────────────────
    print("Training complete. Now running inference...")
    best_ckpt = callbacks[1].best_model_path
    print(f"Best checkpoint: {best_ckpt}")
    cfg.checkpoint = best_ckpt

    predict(cfg)


def sliding_window_predict(model, img_tensor, patch_size=192, device='cuda'):
    """
    - model: Supervised(cfg)로 로드되어 eval 상태인 ChannelViT 모델
    - img_tensor: (C, H, W) 또는 (1, C, H, W)인 torch.Tensor
    - patch_size: 192
    - device: 'cuda' 혹은 'cpu'
    → 전체 (H, W) 크기의 예측 마스크를 리턴함 (numpy uint8, 0/1)
    """
    if img_tensor.dim() == 3:
        img = img_tensor
    else:
        # 배치 차원이 있으면 (1, C, H, W) → (C, H, W)
        img = img_tensor.squeeze(0)
    C, H, W = img.shape
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size

    full_pred = np.zeros((H, W), dtype=np.uint8)

    for row in range(num_patches_h):
        for col in range(num_patches_w):
            y0 = row * patch_size
            y1 = y0 + patch_size
            x0 = col * patch_size
            x1 = x0 + patch_size

            patch = img[:, y0:y1, x0:x1].unsqueeze(0).to(device)  # (1, C, 192, 192)
            covariates = {'channels': [list(range(C))]}

            with torch.no_grad():
                output = model.backbone(patch, covariates)

                # ── 1) backbone 출력 형태 정리 ───────────────────────────────────
                if output.dim() == 3:
                    # output: (B, N, D) 형태 → (B, D, C_ * patch_Hf, patch_Wf)로 reshape
                    B, N, D = output.shape
                    patch_Hf = patch.shape[2] // model.cfg.backbone.patch_size
                    patch_Wf = patch.shape[3] // model.cfg.backbone.patch_size
                    C_ = patch.shape[1]
                    # output = output.permute(0, 2, 1).contiguous().view(B, D, C_, patch_Hf, patch_Wf)
                    # output = output.reshape(B, D, C_ * patch_Hf, patch_Wf)
                    feat = output.permute(0, 2, 1).contiguous().view(B, D, C_, patch_Hf, patch_Wf)
                    feat = feat.reshape(1, D * C_, patch_Hf, patch_Wf)
                    #output = feat.mean(dim=2)
                    output = model.channel_fuser(feat)
                elif output.dim() == 2:
                    # output: (B, D) → (B, D, 1, 1)
                    output = output[:, :, None, None]

                # ── 2) classifier로 픽셀 logits 획득 ─────────────────────────────
                #logits = model.classifier(output)
                #*decoderunet
                logits = model.decoder(output)
                # logits.shape == (1, num_classes, Hf, Wf)

                # ── 3) patch_feature → 원본 patch 해상도(192×192)로 upsample ───────
                '''
                logits_upsampled = F.interpolate(
                    logits,
                    size=(patch_size, patch_size),
                    mode="bilinear",
                    align_corners=False
                )
                '''
                # logits_upsampled.shape == (1, num_classes, 192, 192)

                # ── 4) 픽셀 단위 argmax → (1, 192, 192)
                # preds = torch.argmax(logits_upsampled, dim=1)
                preds = torch.argmax(logits, dim=1)
                pred_np = preds.squeeze(0).cpu().numpy().astype(np.uint8)

            full_pred[y0:y1, x0:x1] = pred_np

    return full_pred


def map_label(mask):
    """
    mask: (H, W) int64 tensor or numpy array, 값 {0,1,2}
    AR 레이블(=2)이면 1, 나머지는 0으로 매핑
    """
    mapped = np.zeros_like(mask, dtype=np.uint8)
    mapped[mask == 2] = 1
    return mapped


def compute_iou(pred, gt):
    """
    pred, gt: (H, W) binary mask (0/1), uint8
    """
    intersection = np.logical_and(pred == 1, gt == 1).sum()
    union = np.logical_or(pred == 1, gt == 1).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def save_side_by_side_mask(gt_mask, pred_mask, save_path):
    """
    gt_mask, pred_mask: (H, W) numpy array (0/1), uint8
    save_path: 저장할 파일 경로(.png)
    """
    gt_img = Image.fromarray(gt_mask * 255)
    pred_img = Image.fromarray(pred_mask * 255)
    assert gt_img.size == pred_img.size, "마스크 크기가 다릅니다!"
    w, h = gt_img.size
    combined = Image.new("L", (w * 2, h))
    combined.paste(gt_img, (0, 0))
    combined.paste(pred_img, (w, 0))
    combined.save(save_path)
    print(f"저장됨: {save_path}")

def predict(cfg: DictConfig):
    """
    - cfg.checkpoint: 실행할 체크포인트 경로
    - cfg.test_data: Hydra로 지정된 ar_test.yaml 내용
    1) DataLoader를 순회하며 (img, {"label": label}) 을 얻어옴
    2) sliding_window_predict로 예측 mask 얻기
    3) label_tensor는 이미 (0/1) 이진화된 상태 → 그대로 사용
    4) IoU 계산
    5) 결과(이미지, 비교 이미지, IoU.txt)를 인덱스 단위로 저장
    """
    # ─── 1) 모델 로드 ─────────────────────────────────────────────────────────────
    model = "/home/aix23606/limseoyen/new/ChannelViT/snapshots/epoch=99-step=242400.ckpt" #Supervised.load_from_checkpoint(cfg.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # ─── 2) test DataLoader 준비 ─────────────────────────────────────────────────
    test_data_cfg = cfg.test_data
    test_dataset = hydra.utils.instantiate(test_data_cfg)
    test_loader = DataLoader(
        test_dataset,
        **test_data_cfg.loader,
        collate_fn=test_dataset.collate_fn
    )

    os.makedirs("./seg_output", exist_ok=True)

    # ─── 3) DataLoader 순회하며 예측 처리 ─────────────────────────────────────────
    for idx, batch in enumerate(test_loader):
        # ARDataset.__getitem__이 (img_tensor, {"label": label_tensor})를 반환하므로,
        # default_collate을 거친 batch는 tuple(배치_이미지, {"label":배치_레이블})
        img_tensor = batch[0].to(device)              
        # label_tensor는 이미 0/1 형태이므로 바로 가져옴
        label_tensor = batch[1]["label"].squeeze(0).cpu().numpy().astype(np.uint8)  # (H, W), {0,1}

        # ── 3-1) sliding window inference ─────────────────────────────────
        pred_mask = sliding_window_predict(model, img_tensor, patch_size=192, device=device)
        # pred_mask: (H, W) uint8, {0,1}

        # ── 3-2) label_tensor는 그대로 GT 이진 마스크이므로 별도 map 필요 없음
        gt_mask = label_tensor  # 이미 AR=1, 그 외=0

        # ── 3-3) IoU 계산 ────────────────────────────────────────────────
        iou = compute_iou(pred_mask, gt_mask)
        print(f"[Sample {idx}] IoU (AR vs not-AR): {iou:.4f}")

        # ── 3-4) 결과 저장: GT, Pred, Compare, IoU.txt ───────────────────
        base_name = f"sample_{idx}"
        gt_path = f"./seg_output/{base_name}_gt.png"
        pred_path = f"./seg_output/{base_name}_pred.png"
        compare_path = f"./seg_output/{base_name}_compare.png"
        iou_txt_path = f"./seg_output/{base_name}_iou.txt"

        Image.fromarray(gt_mask * 255).save(gt_path)
        Image.fromarray(pred_mask * 255).save(pred_path)
        save_side_by_side_mask(gt_mask, pred_mask, compare_path)

        with open(iou_txt_path, "w") as f:
            f.write(f"IoU: {iou:.6f}\n")

        print(f"[Sample {idx}] Saved → {gt_path}, {pred_path}, {compare_path}, {iou_txt_path}")


if __name__ == "__main__":
    train()

