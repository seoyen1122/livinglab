
import os
import hydra
import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
import csv

import channelvit.data as data  # ARDataset
from channelvit.meta_arch import Supervised


def compute_class_iou(pred: np.ndarray, gt: np.ndarray, class_idx: int) -> float:
    """pred, gt: (H, W) uint8 masks; class_idx in {0,1}"""
    inter = np.logical_and(pred == class_idx, gt == class_idx).sum()
    union = np.logical_or(pred == class_idx, gt == class_idx).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def sliding_window_predict(model, img_tensor, patch_size=192, device="cuda", overlap=0.5):
    """
    img_tensor: (C, H, W) 또는 (1, C, H, W)
    return: full_pred (H, W) uint8, {0,1}
    """
    if img_tensor.dim() == 4:
        img = img_tensor.squeeze(0)
    else:
        img = img_tensor
    C, H, W = img.shape
    final_mask = np.zeros((H, W), dtype=np.uint8)

    stride = int(patch_size * (1 - overlap))

    score_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.uint16)

    for y0 in range(0, H - patch_size + 1, stride):
        for x0 in range(0, W - patch_size + 1, stride):
            y1 = y0 + patch_size
            x1 = x0 + patch_size
            
            patch = img[:, y0:y1, x0:x1].unsqueeze(0).to(device)
            cov = {"channels": [list(range(C))]}
            with torch.no_grad():
                out = model.backbone(patch, cov)
                if out.dim() == 3:  # (B, N, D)
                    B, N, D = out.shape
                    ph = patch_size // model.cfg.backbone.patch_size
                    pw = patch_size // model.cfg.backbone.patch_size
                    C  = patch.shape[1]
                    feat = out.permute(0, 2, 1).contiguous().view(B, D, C, ph, pw)
                    out = feat.mean(dim=2)      # (B, D, ph, pw)
                elif out.dim() == 2:  # (B, D)
                    out = out[:, :, None, None]  # (B, D, 1, 1)
                logits = model.decoder(out)  # logits: (B, num_classes, H', W')
                probs = F.softmax(logits, dim=1)  # (B, num_classes, H', W')

            prob_np = probs[0, 1].cpu().numpy()
            score_map[y0:y1, x0:x1] += prob_np
            count_map[y0:y1, x0:x1] += 1

    x0 = W - patch_size
    for y0 in range(0, H - patch_size + 1, stride):
            y1 = y0 + patch_size
            x1 = x0 + patch_size
            
            patch = img[:, y0:y1, x0:x1].unsqueeze(0).to(device)
            cov = {"channels": [list(range(C))]}
            with torch.no_grad():
                out = model.backbone(patch, cov)
                if out.dim() == 3:  # (B, N, D)
                    B, N, D = out.shape
                    ph = patch_size // model.cfg.backbone.patch_size
                    pw = patch_size // model.cfg.backbone.patch_size
                    C  = patch.shape[1]
                    feat = out.permute(0, 2, 1).contiguous().view(B, D, C, ph, pw)
                    out = feat.mean(dim=2)       # (B, D, ph, pw)
                elif out.dim() == 2:  # (B, D)
                    out = out[:, :, None, None]  # (B, D, 1, 1)
                logits = model.decoder(out)  # logits: (B, num_classes, H', W')
                probs = F.softmax(logits, dim=1)  # (B, num_classes, H', W')

            prob_np = probs[0, 1].cpu().numpy()
            score_map[y0:y1, x0:x1] += prob_np
            count_map[y0:y1, x0:x1] += 1

    y0 = H - patch_size
    for x0 in range(0, W - patch_size + 1, stride):
            y1 = y0 + patch_size
            x1 = x0 + patch_size
            
            patch = img[:, y0:y1, x0:x1].unsqueeze(0).to(device)
            cov = {"channels": [list(range(C))]}
            with torch.no_grad():
                out = model.backbone(patch, cov)
                if out.dim() == 3:  # (B, N, D)
                    B, N, D = out.shape
                    ph = patch_size // model.cfg.backbone.patch_size
                    pw = patch_size // model.cfg.backbone.patch_size
                    C  = patch.shape[1]
                    feat = out.permute(0, 2, 1).contiguous().view(B, D, C, ph, pw)
                    out = feat.mean(dim=2) # (B, D, ph, pw)
                elif out.dim() == 2:  # (B, D)
                    out = out[:, :, None, None]  # (B, D, 1, 1)
                logits = model.decoder(out)  # logits: (B, num_classes, H', W')
                probs = F.softmax(logits, dim=1)  # (B, num_classes, H', W')

            prob_np = probs[0, 1].cpu().numpy()
            score_map[y0:y1, x0:x1] += prob_np
            count_map[y0:y1, x0:x1] += 1
    y0, x0 = H - patch_size, W - patch_size

    avg_score = score_map / count_map
    final_mask = (avg_score > 0.5).astype(np.uint8)

    return final_mask


def compute_iou(pred, gt):
    inter = np.logical_and(pred == 1, gt == 1).sum()
    union = np.logical_or(pred == 1, gt == 1).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def save_compare(gt, pred, path):
    gt_img = Image.fromarray(gt * 255)
    pd_img = Image.fromarray(pred * 255)
    w, h = gt_img.size
    combo = Image.new("L", (2 * w, h))
    combo.paste(gt_img, (0, 0))
    combo.paste(pd_img, (w, 0))
    combo.save(path)


@hydra.main(version_base=None, config_path="../config", config_name="main_supervised")
def inference(cfg: DictConfig):
    # 1) 모델 로드
    model = Supervised.load_from_checkpoint(cfg.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # 2) 테스트 DataLoader (ARDataset via npz)
    test_ds = hydra.utils.instantiate(cfg.test_data)
    test_loader = DataLoader(
        test_ds, **cfg.test_data.loader, collate_fn=test_ds.collate_fn
    )
    output_dir = "/home/aix23606/limseoyen/new/ChannelViT/seg_output/v1/stne10"
    os.makedirs(output_dir, exist_ok=True)

    # 3) CSV 파일 열기
    csv_path = os.path.join(output_dir, "iou_results.csv")

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["index", "iou0", "iou1", "miou"])

        font = ImageFont.load_default()
        all_iou0, all_iou1, all_miou = [], [], []

        # 4) 추론 루프
        for idx, batch in enumerate(test_loader):
            img, lbl_dict = batch
            img = img.to(device)
            gt  = lbl_dict["label"].squeeze(0).cpu().numpy().astype(np.uint8)

            # 사이징 & 예측
            pred = sliding_window_predict(model,
                                          img,
                                          patch_size=192,
                                          device=device,
                                          overlap=0.5)

            # 클래스별 IoU
            iou0 = compute_class_iou(pred, gt, class_idx=0)
            iou1 = compute_class_iou(pred, gt, class_idx=1)
            miou = 0.5 * (iou0 + iou1)

            all_iou0.append(iou0)
            all_iou1.append(iou1)
            all_miou.append(miou)

            # CSV 기록
            writer.writerow([idx,
                             f"{iou0:.6f}",
                             f"{iou1:.6f}",
                             f"{miou:.6f}"])

            # 5) 비교 이미지 생성 & 텍스트 오버레이
            # GT 이미지
            gt_img = Image.fromarray(gt * 255).convert("RGB")
            w, h   = gt_img.size

            # Pred 이미지
            pd_img = Image.fromarray(pred * 255).convert("RGB")

            # 합칠 캔버스
            combo = Image.new("RGB", (w*2, h))
            combo.paste(gt_img, (0, 0))
            combo.paste(pd_img, (w, 0))

            draw = ImageDraw.Draw(combo)
            # 왼쪽 타이틀
            draw.text((5, 5), "Ground Truth", fill=(255, 0, 0), font=font)
            # 오른쪽 타이틀 (IoU)
            draw.text((w + 5, 5),
                      f"Prediction IoU: {iou1:.4f}",
                      fill=(255, 0, 0),
                      font=font)

            # 저장
            cmp_path = os.path.join(output_dir,
                                    f"sample_{idx:04d}_compare.png")
            combo.save(cmp_path)

        # 6) 평균값 CSV에 한 줄 더 기록
        writer.writerow(["mean",
                         f"{np.mean(all_iou0):.6f}",
                         f"{np.mean(all_iou1):.6f}",
                         f"{np.mean(all_miou):.6f}"])

    print(f"▶️ Saved compare images and CSV to {output_dir}")
    print("=== Inference complete ===")

if __name__ == "__main__":
    inference()

    '''
 
    # 3) 추론 반복
    for idx, batch in enumerate(test_loader):
        img, lbl_dict = batch
        img = img.to(device)
        gt = lbl_dict["label"].squeeze(0).cpu().numpy().astype(np.uint8)

        pred = sliding_window_predict(model, img, patch_size=192, device=device, overlap=0.5)

        iou0 = compute_class_iou(pred, gt, class_idx=0)
        iou1 = compute_class_iou(pred, gt, class_idx=1)
        miou = 0.5 * (iou0 + iou1)
        print(f"[{idx:04d}] IoU0={iou0:.4f}, IoU1={iou1:.4f}, mIoU={miou:.4f}")

        base = f"sample_{idx:04d}"
        gt_path = os.path.join(output_dir, base + "_gt.png")
        pd_path = os.path.join(output_dir, base + "_pred.png")
        cmp_path = os.path.join(output_dir, base + "_compare.png")
        iou_path = os.path.join(output_dir, base + "_iou.txt")

        Image.fromarray(gt * 255).save(gt_path)
        Image.fromarray(pred * 255).save(pd_path)
        save_compare(gt, pred, cmp_path)
        with open(iou_path, "w") as f:
            f.write(f"IoU: {iou1:.6f}\n")

    print("=== Inference complete ===")


if __name__ == "__main__":
    inference()
'''


'''
import os
import hydra
import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from PIL import Image

import channelvit.data as data  # ARDataset
from channelvit.meta_arch import Supervised

def sliding_window_predict(model, img_tensor, patch_size=192, device="cuda"):
    """
    img_tensor: (C, H, W) 또는 (1, C, H, W)
    return: full_pred (H, W) uint8, {0,1}
    """
    if img_tensor.dim() == 4:
        img = img_tensor.squeeze(0)
    else:
        img = img_tensor
    C, H, W = img.shape
    full_pred = np.zeros((H, W), dtype=np.uint8)

    num_h = H // patch_size
    num_w = W // patch_size


    for i in range(num_h):
        for j in range(num_w):
            y0, y1 = i * patch_size, (i + 1) * patch_size
            x0, x1 = j * patch_size, (j + 1) * patch_size
            patch = img[:, y0:y1, x0:x1].unsqueeze(0).to(device)
            cov = {"channels": [list(range(C))]}
            with torch.no_grad():
                out = model.backbone(patch, cov)
                if out.dim() == 3:
                    B, N, D = out.shape
                    ph = patch.shape[2] // model.cfg.backbone.patch_size
                    pw = patch.shape[3] // model.cfg.backbone.patch_size
                    C = patch.shape[1]
                    feat = out.permute(0, 2, 1).view(B, D, C, ph, pw)

                    #*channel_fuser
                    out = feat.mean(dim=2)
                    # feat = feat.reshape(B, D * C, ph, pw)
                    # out = model.channel_fuser(feat)

                else:  # dim==2
                    out = out[:, :, None, None]

                logits = model.decoder(out)  
                pred = torch.argmax(logits, dim=1)
                pred = torch.argmax(logits, dim=1).cpu().numpy()
            
            full_pred[y0:y1, x0:x1] = pred

    return full_pred


def compute_iou(pred, gt):
    inter = np.logical_and(pred == 1, gt == 1).sum()
    union = np.logical_or(pred == 1, gt == 1).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def save_compare(gt, pred, path):
    gt_img = Image.fromarray(gt * 255)
    pd_img = Image.fromarray(pred * 255)
    w, h = gt_img.size
    combo = Image.new("L", (2 * w, h))
    combo.paste(gt_img, (0, 0))
    combo.paste(pd_img, (w, 0))
    combo.save(path)


@hydra.main(version_base=None, config_path="../config", config_name="main_supervised")
def inference(cfg: DictConfig):
    # 1) 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Supervised.load_from_checkpoint(cfg.checkpoint)
    model.to(device).eval()
    print(model)

    # 2) 테스트 DataLoader (ARDataset via npz)
    test_ds = hydra.utils.instantiate(cfg.test_data)
    test_loader = DataLoader(
        test_ds, **cfg.test_data.loader, collate_fn=test_ds.collate_fn
    )
    output_dir = "/home/aix23606/limseoyen/new/ChannelViT/seg_output/p16b8_epoch50_v6"
    os.makedirs(output_dir, exist_ok=True)

    # 3) 추론 반복
    for idx, batch in enumerate(test_loader):
        img, lbl_dict = batch
        img = img.to(device)
        gt = lbl_dict["label"].squeeze(0).cpu().numpy().astype(np.uint8)

        pred = sliding_window_predict(model, img, patch_size=192, device=device)

        iou = compute_iou(pred, gt)
        print(f"[{idx:04d}] IoU = {iou:.4f}")

        base = f"sample_{idx:04d}"
        gt_path = os.path.join(output_dir, base + "_gt.png")
        pd_path = os.path.join(output_dir, base + "_pred.png")
        cmp_path = os.path.join(output_dir, base + "_compare.png")
        iou_path = os.path.join(output_dir, base + "_iou.txt")

        Image.fromarray(gt * 255).save(gt_path)
        Image.fromarray(pred * 255).save(pd_path)
        save_compare(gt, pred, cmp_path)
        with open(iou_path, "w") as f:
            f.write(f"IoU: {iou:.6f}\n")

    print("=== Inference complete ===")


if __name__ == "__main__":
    inference()
'''