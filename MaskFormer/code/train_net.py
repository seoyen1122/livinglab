# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import copy
import itertools
import logging
import os
import re
from collections import OrderedDict
from typing import Any, Dict, List, Set
from tqdm import tqdm
from collections import defaultdict

import torch

import numpy as np
import matplotlib.pyplot as plt

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

from detectron2.utils.visualizer import Visualizer
from detectron2.utils.events import EventStorage, get_event_storage

from mask_former.data.dataset_mappers.netcdf_semantic_mapper import NetCDFSemanticMapper
# from sliding_window_inference import sliding_window_inference

# MaskFormer
from mask_former import (
    DETRPanopticDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_mask_former_config,
)

from mask_former.data.datasets.register_climatenet_dataset import register_climatenet
from mask_former.data.datasets.register_climatenet_patches import register_climatenet_patches
#from register_climatenet_multiscale_patches import register_climatenet_multiscale_patches

register_climatenet()
register_climatenet_patches()
#register_climatenet_multiscale_patches()

# SemSegEvaluator 커스터마이징    
class NetCDFSemSegEvaluator(SemSegEvaluator):
    def __init__(self, dataset_name, model, device=None, distributed=False, output_dir=None):
       super().__init__(
           dataset_name,
           distributed=distributed,
           output_dir=output_dir,
       )
       # mirror the protected var so we can read it as self.output_dir
       self.output_dir = self._output_dir
       self.model = model.eval()  # eval 모드 보장
       self.device = device if device else "cuda"

    def process(self, inputs, outputs):
        """
        inputs: detectron2 dict, outputs: model output dict (둘 다 patch 단위)
        """
        for input, output in zip(inputs, outputs):
            gt = input["sem_seg"]
            if hasattr(gt, "numpy"):
                gt = gt.numpy()
            elif hasattr(gt, "values"):
                gt = gt.values
            else:
                gt = np.array(gt)

            # (수정) output["sem_seg"]를 바로 사용
            sem_seg_logits = output["sem_seg"]
            if hasattr(sem_seg_logits, "dim") and sem_seg_logits.dim() == 4:
                sem_seg_logits = sem_seg_logits.squeeze(0)
            pred = sem_seg_logits.argmax(dim=0).cpu().numpy()

            print(f"\n[VAL PATCH {os.path.basename(input['file_name'])}]")
            print("GT unique:         ", np.unique(gt, return_counts=True))
            print("Pred unique:       ", np.unique(pred, return_counts=True))
            print("Logits min/max:    ", sem_seg_logits.min().item(), sem_seg_logits.max().item())
            print("Mean logits [bg,AR]:", [sem_seg_logits[0].mean().item(), sem_seg_logits[1].mean().item()])

            self._predictions.append((gt.astype(np.uint8), pred))


    def evaluate(self):
        per_class_ious = {0: [], 1: []}  # 0 = background, 1 = AR

        for gt, pred in self._predictions:
            for class_id in [0, 1]:
                intersection = np.logical_and(gt == class_id, pred == class_id).sum()
                union = np.logical_or(gt == class_id, pred == class_id).sum()
                iou = intersection / (union + 1e-6)
                per_class_ious[class_id].append(iou)

        mean_iou_bg = np.mean(per_class_ious[0])
        mean_iou_ar = np.mean(per_class_ious[1])
        mean_iou = (mean_iou_bg + mean_iou_ar) / 2

        print(f"[EVAL] Mean IoU:       {mean_iou:.4f}")
        print(f"[EVAL] Background IoU: {mean_iou_bg:.4f}")
        print(f"[EVAL] AR IoU:         {mean_iou_ar:.4f}")

        # --- Accuracy/val loss 계산 (binary segmentation 기준) ---
        total = 0
        correct = 0
        total_loss = 0
        for gt, pred in self._predictions:
            mask = (gt != 255)
            total += mask.sum()
            correct += ((gt == pred) & mask).sum()
            total_loss += np.mean(gt[mask] != pred[mask])
        acc = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(self._predictions)

        # metrics/eval_iou.txt 저장
        # use the evaluator’s output_dir (inherited from SemSegEvaluator)
        vis_dir = os.path.join(self._output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        with open(os.path.join(vis_dir, "eval_iou.txt"), "w") as f:
            f.write(f"Background IoU: {mean_iou_bg:.4f}\n")
            f.write(f"AR IoU:         {mean_iou_ar:.4f}\n")
            f.write(f"Mean IoU:       {mean_iou:.4f}\n")

        # --- Trainer에 기록 (detectron2 Trainer와 연결 필요) ---

        # --- Tensorboard에도 기록 (optional) ---
        with EventStorage(0):
            storage = get_event_storage()
            storage.put_scalar("val/mean_iou", mean_iou)
            storage.put_scalar("val/iou_background", mean_iou_bg)
            storage.put_scalar("val/iou_ar", mean_iou_ar)
            storage.put_scalar("val/accuracy", acc)
            storage.put_scalar("val/loss", avg_loss)

        return {
            "Mean IoU": mean_iou,
            "Background IoU": mean_iou_bg,
            "AR IoU": mean_iou_ar,
            "Val Loss": avg_loss,
            "Val Accuracy": acc,
        }
            

class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to DETR.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def train(self):
        best_score = -1.0
        max_iter   = self.max_iter

        with EventStorage(start_iter=self.start_iter):
            for self.iter in range(self.start_iter, max_iter):
                self.run_step()

                # expose our ddp-wrapped model and device to the evaluator
                global model, device
                model  = self.model
                device = self.model.device  # or: cfg.MODEL.DEVICE

                # 1000 iter마다, 또는 마지막 iter마다 평가
                if (self.iter + 1) % self.cfg.TEST.EVAL_PERIOD == 0 or (self.iter + 1) == max_iter:
                    val_results = self.test(self.cfg, self.model)
                    # make sure we go back to train mode for the next run_step()
                    self.model.train()
                    mean_iou   = val_results["Mean IoU"]

                    from detectron2.utils.comm import is_main_process
                    if mean_iou > best_score and is_main_process():
                        print(f"[EVAL] 🥇 New Best Mean IoU: {mean_iou:.4f} → Saving model_best.pth")
                        best_score = mean_iou
                        self.checkpointer.save("model_best")

                    # (기존 train_val 기록 로직 유지)
                    if "Val Accuracy" in val_results and "Val Loss" in val_results:
                        self.val_accuracies.append(val_results["Val Accuracy"])
                        self.val_losses.append(val_results["Val Loss"])

                if (self.iter + 1) % self.cfg.SOLVER.CHECKPOINT_PERIOD == 0:
                    self.checkpointer.save(f"model_{self.iter + 1:07d}")

        return OrderedDict()
    
    @classmethod
    def test(cls, cfg, model, evaluators=None):
        # Detectron2 default test 로직 복붙 + evaluator에 model 넘기도록 수정
        import logging
        import detectron2.evaluation
        from detectron2.data import build_detection_test_loader

        logger = logging.getLogger(__name__)

        dataset_names = cfg.DATASETS.TEST
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        if evaluators is not None:
            assert len(evaluators) == len(dataset_names), (
                f"Evaluators length ({len(evaluators)}) does not match dataset length ({len(dataset_names)})"
            )
        else:
            evaluators = [cls.build_evaluator(cfg, name, model) for name in dataset_names]
        results = OrderedDict()
        for idx, dataset_name in enumerate(dataset_names):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            evaluator = evaluators[idx]
            results_i = detectron2.evaluation.inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if hasattr(evaluator, "reset"):
                evaluator.reset()
        if len(results) == 1:
            results = list(results.values())[0]
        return results


    @classmethod
    def build_evaluator(cls, cfg, dataset_name, model, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        # evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            return NetCDFSemSegEvaluator(
                dataset_name,
                model,
                device=cfg.MODEL.DEVICE if hasattr(cfg.MODEL, "DEVICE") else "cuda",
                distributed=True,
                output_dir=output_folder,
            )
            # evaluator_list.append(
            #     NetCDFSemSegEvaluator(
            #         dataset_name,
            #         distributed=True,
            #         output_dir=output_folder,
            #     )
            # )
        # if evaluator_type == "coco":
        #     evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        # if evaluator_type in [
        #     "coco_panoptic_seg",
        #     "ade20k_panoptic_seg",
        #     "cityscapes_panoptic_seg",
        # ]:
        #     evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        # if evaluator_type == "cityscapes_instance":
        #     assert (
        #         torch.cuda.device_count() >= comm.get_rank()
        #     ), "CityscapesEvaluator currently do not work with multiple machines."
        #     return CityscapesInstanceEvaluator(dataset_name)
        # if evaluator_type == "cityscapes_sem_seg":
        #     assert (
        #         torch.cuda.device_count() >= comm.get_rank()
        #     ), "CityscapesEvaluator currently do not work with multiple machines."
        #     return CityscapesSemSegEvaluator(dataset_name)
        # if evaluator_type == "cityscapes_panoptic_seg":
        #     assert (
        #         torch.cuda.device_count() >= comm.get_rank()
        #     ), "CityscapesEvaluator currently do not work with multiple machines."
        #     evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
        # if len(evaluator_list) == 0:
        #     raise NotImplementedError(
        #         "no Evaluator for the dataset {} with the type {}".format(
        #             dataset_name, evaluator_type
        #         )
        #     )
        # elif len(evaluator_list) == 1:
        #     return evaluator_list[0]
        # return DatasetEvaluators(evaluator_list)


    # @classmethod
    # def build_train_loader(cls, cfg):
    #     # Semantic segmentation dataset mapper
    #     if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
    #         mapper = MaskFormerSemanticDatasetMapper(cfg, True)
    #     # Panoptic segmentation dataset mapper
    #     elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
    #         mapper = MaskFormerPanopticDatasetMapper(cfg, True)
    #     # DETR-style dataset mapper for COCO panoptic segmentation
    #     elif cfg.INPUT.DATASET_MAPPER_NAME == "detr_panoptic":
    #         mapper = DETRPanopticDatasetMapper(cfg, True)
    #     else:
    #         mapper = None
    #     return build_detection_train_loader(cfg, mapper=mapper)


    @classmethod
    def build_train_loader(cls, cfg):
        mapper = NetCDFSemanticMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)


    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    def run_step(self):
        assert self.model.training, "[Trainer] model was changed to eval mode!"
        if not hasattr(self, "_data_loader_iter"):
            self._data_loader_iter = iter(self.data_loader)
        data = next(self._data_loader_iter)

        # Forward
        loss_dict = self.model(data)
        losses = sum(loss_dict.values())
        self.optimizer.zero_grad()
        losses.backward()

        # --- Gradient flow 체크 ---
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                print(f"[Grad] {name}: grad mean={param.grad.mean().item():.6f}")
                break  # 첫 번째 있는 것만 확인해도 됨
        else:
            print("❌ No gradients computed!")

        self.optimizer.step()

        # 기록
        self.train_losses.append(losses.item())

        # ──────────────────────────────────────────────────
        # Train 정확도 계산 (eval 모드로 한 번 forward)
        self.model.eval()
        with torch.no_grad():
            preds_list = self.model(data)
            sem_logits = preds_list[0]["sem_seg"]       # (1, 2, H, W)
            sem_logits = sem_logits.squeeze(0)          # → (2, H, W)
            preds = sem_logits.argmax(dim=0)            # → (H, W)
            labels = data[0]["sem_seg"].to(preds.device)
            valid = (labels != 255)
            acc = (preds[valid] == labels[valid]).float().mean().item()
            self.train_accuracies.append(acc)
            
            # === 디버깅/분포 출력 ===
            gt_mask = labels.cpu().numpy()
            pred_mask = preds.cpu().numpy()
            print(f"\n[TRAIN STEP {self.iter}]")
            print("GT unique:          ", np.unique(gt_mask, return_counts=True))
            print("Pred mask unique:   ", np.unique(pred_mask, return_counts=True))
            print("Pred logits shape:  ", sem_logits.shape)
            print("Pred logits min/max:", sem_logits.min().item(), sem_logits.max().item())
            print("Mean logits [bg, AR]:", [sem_logits[0].mean().item(), sem_logits[1].mean().item()])

            # === 샘플 시각화 (10 iter마다 한 번씩만, 또는 if self.iter < 20:)
            if self.iter % 10 == 0:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(8,4))
                plt.subplot(1,2,1); plt.imshow(gt_mask, cmap='gray'); plt.title("GT"); plt.axis('off')
                plt.subplot(1,2,2); plt.imshow(pred_mask, cmap='gray'); plt.title("Pred"); plt.axis('off')
                plt.suptitle(f"Train Patch: iter {self.iter}")
                plt.tight_layout()
                os.makedirs("output_patch_vis", exist_ok=True)
                plt.savefig(f"output_patch_vis/train_{self.iter}.png")
                plt.close()
        self.model.train()

        # TensorBoard 기록 (기존대로)
        storage = get_event_storage()
        storage.put_scalar("total_loss", losses.item())
        for k, v in loss_dict.items():
            storage.put_scalar(k, v.item())

        # 콘솔 진행률 출력
        cur_iter = self.iter
        total_iter = self.max_iter
        print(f"[STEP {cur_iter}/{total_iter}] total_loss={losses.item():.4f}")


    # build_test_loader() 오버라이드
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(
            cfg, dataset_name, mapper=NetCDFSemanticMapper(cfg, is_train=False)
        )
    
    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

def plot_curves(trainer, output_dir):
    import matplotlib.pyplot as plt
    import os

    total_iters = len(trainer.train_losses)
    epochs     = list(range(1, total_iters + 1))
    eval_period = trainer.cfg.TEST.EVAL_PERIOD
    # 1000,2000,3000,… 처럼 val 이 기록된 iteration 번호 리스트
    val_iters   = list(range(eval_period, total_iters+1, eval_period))

    plt.figure(figsize=(12, 5))

    # ─ Loss Curve ───────────────────────────────────────
    plt.subplot(1, 2, 1)
    # Train Loss: 실선
    plt.plot(epochs, trainer.train_losses, label='Train Loss')
    # Val Loss: 실선
    if trainer.val_losses:
        plt.plot(val_iters, trainer.val_losses, color='orange', label='Val Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')

    # ─ Accuracy Curve ───────────────────────────────────
    plt.subplot(1, 2, 2)
    # Train Accuracy: 실선
    if trainer.train_accuracies:
        plt.plot(epochs, trainer.train_accuracies, label='Train Accuracy')
    # Val Accuracy: 실선
    if trainer.val_accuracies:
        plt.plot(val_iters, trainer.val_accuracies, color='green', label='Val Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'train_val_curves.png'))
    plt.close()


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_mask_former_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask_former")
    return cfg

def extract_coords(fname):
    m = re.search(r'_r(\d+)_c(\d+)', fname)
    if m:
        return int(m.group(1)), int(m.group(2))
    else:
        raise ValueError(f"No coords in filename: {fname}")

def reconstruct_full_image(patch_preds, patch_coords, full_shape, patch_size):
    """
    patch_preds: [N_patch, patch_h, patch_w] 또는 [N_patch, 1, patch_h, patch_w]
    patch_coords: [(i, j), ...]
    full_shape: (H, W)
    patch_size: (ph, pw)
    """
    ph, pw = patch_size
    H, W = full_shape
    recon = np.zeros(full_shape, dtype=patch_preds[0].dtype)
    for patch, (i, j) in zip(patch_preds, patch_coords):
        recon[i:i+ph, j:j+pw] = patch.squeeze()
    return recon

def save_prediction_visuals(dataset_dicts, outputs, save_dir, metadata_name, full_shape=(768, 1152), patch_size=(192,192)):
    """
    패치별 prediction을 원본 이미지별로 이어붙여 IoU/시각화
    dataset_dicts: detectron2의 리스트(dict), 패치 단위
    outputs: 모델 output 리스트(패치 단위)
    save_dir: 시각화 저장 폴더
    """
    os.makedirs(save_dir, exist_ok=True)
    # 1. 원본 이미지 기준으로 패치 grouping
    patches_by_img = defaultdict(list)
    gts_by_img = defaultdict(list)
    coords_by_img = defaultdict(list)
    for d, output in zip(dataset_dicts, outputs):
        fname = os.path.basename(d["file_name"])
        base = fname.split('_r')[0]  # ex: 20-03-03_06
        i, j = extract_coords(fname)
        # GT
        gt = d["sem_seg"]
        if hasattr(gt, "values"):
            gt = gt.values
        elif not isinstance(gt, np.ndarray):
            gt = np.array(gt)
        gt = gt.astype(np.int32)
        # Prediction
        sem_seg_logits = output["sem_seg"]
        if hasattr(sem_seg_logits, "dim") and sem_seg_logits.dim() == 4:
            sem_seg_logits = sem_seg_logits.squeeze(0)
        pred = sem_seg_logits.argmax(dim=0).cpu().numpy()
        # 저장
        patches_by_img[base].append(pred)
        gts_by_img[base].append(gt)
        coords_by_img[base].append((i, j))

    # 2. 이미지별로 이어붙이고 시각화
    for base in patches_by_img:
        pred_full = reconstruct_full_image(patches_by_img[base], coords_by_img[base], full_shape, patch_size)
        gt_full   = reconstruct_full_image(gts_by_img[base], coords_by_img[base], full_shape, patch_size)
        # IoU (AR 기준: class==1)
        intersection = np.logical_and(gt_full == 1, pred_full == 1).sum()
        union = np.logical_or(gt_full == 1, pred_full == 1).sum()
        iou = intersection / (union + 1e-6)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(gt_full, cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(pred_full, cmap="gray")
        plt.title(f"Prediction (IoU: {iou:.3f})")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{base}_iou_{iou:.3f}.png"))
        plt.close()

def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)

        # metrics 저장 및 시각화
        if comm.is_main_process():
            import json
            with open(os.path.join(cfg.OUTPUT_DIR, "metrics.json"), "w") as f:
                json.dump(res, f, indent=2)

            # 패치 단위 inference, 시각화
            from detectron2.data import DatasetCatalog, build_detection_test_loader
            dataset_dicts = DatasetCatalog.get(cfg.DATASETS.TEST[0])
            data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], mapper=NetCDFSemanticMapper(cfg, is_train=False))
            outputs = []
            model.eval()
            with torch.no_grad():
                for batch in data_loader:
                    predictions = model(batch)
                    for pred in predictions:
                        if "sem_seg" not in pred:
                            print("[ERROR] Missing 'sem_seg' in prediction!")
                            continue
                        outputs.append(pred)

            save_prediction_visuals(
                dataset_dicts,
                outputs,
                save_dir=os.path.join(cfg.OUTPUT_DIR, "visualizations"),
                metadata_name=cfg.DATASETS.TEST[0],
                full_shape=(768, 1152),      # ← 원본 shape
                patch_size=(192, 192),       # ← patch shape
            )

        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res
        
        # ## multiscale inference를 슬라이딩 윈도우로 할 때
        # # ──────────────────────────────────
        # # 1) 모델 로드 & eval 모드
        # # ──────────────────────────────────
        # model = Trainer.build_model(cfg).to(cfg.MODEL.DEVICE)
        # DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        #     cfg.MODEL.WEIGHTS, resume=args.resume
        # )
        # model.eval()

        # # ──────────────────────────────────
        # # 2) 슬라이딩 윈도우 추론 & 결과 수집
        # # ──────────────────────────────────
        # from sliding_window_inference import sliding_window_inference
        # from detectron2.data import DatasetCatalog
        # dataset_dicts = DatasetCatalog.get(cfg.DATASETS.TEST[0])
        # outputs = []

        # # 클래스별 IoU 저장용
        # per_class_ious = {0: [], 1: []}  # 0=background, 1=AR

        # for d in tqdm(dataset_dicts, desc="Sliding-window Inference"):
        #     # (1) 원본 이미지만 tensor로, 모델이 내부에서 정규화 처리함
        #     img_t = torch.from_numpy(d["image"]).float().to(cfg.MODEL.DEVICE)

        #     # (2) 슬라이딩 윈도우 적용
        #     pred_mask = sliding_window_inference(
        #         img_t,
        #         model,
        #         patch_size=(192, 192),
        #         stride=(96, 96),
        #         device=cfg.MODEL.DEVICE,
        #     )  # returns numpy (H,W) of class indices

        #     # (3) 클래스별 IoU 계산
        #     gt = d["sem_seg"]
        #     if hasattr(gt, "values"):
        #         gt = gt.values
        #     elif hasattr(gt, "numpy"):
        #         gt = gt.numpy()
        #     else:
        #         gt = np.array(gt)

        #     for class_id in [0, 1]:
        #         intersection = np.logical_and(gt == class_id, pred_mask == class_id).sum()
        #         union = np.logical_or(gt == class_id, pred_mask == class_id).sum()
        #         iou = intersection / (union + 1e-6)
        #         per_class_ious[class_id].append(iou)

        #     # (4) save_prediction_visuals용 output 포맷
        #     sem_seg_tensor = torch.zeros((1, 2, *pred_mask.shape), dtype=torch.float32)
        #     sem_seg_tensor[0, 1, :, :][pred_mask == 1] = 1.0
        #     outputs.append({"sem_seg": sem_seg_tensor})

        # # === 결과 출력 및 txt 저장 ===
        # mean_iou_bg = np.mean(per_class_ious[0])
        # mean_iou_ar = np.mean(per_class_ious[1])
        # mean_iou = (mean_iou_bg + mean_iou_ar) / 2

        # print(f"[RESULT] Background IoU: {mean_iou_bg:.4f}")
        # print(f"[RESULT] AR IoU:         {mean_iou_ar:.4f}")
        # print(f"[RESULT] Mean IoU:       {mean_iou:.4f}")

        # if comm.is_main_process():
        #     vis_dir = os.path.join(cfg.OUTPUT_DIR, "visualizations")
        #     os.makedirs(vis_dir, exist_ok=True)
        #     with open(os.path.join(vis_dir, "eval_iou.txt"), "w") as f:
        #         f.write(f"Background IoU: {mean_iou_bg:.4f}\n")
        #         f.write(f"AR IoU:         {mean_iou_ar:.4f}\n")
        #         f.write(f"Mean IoU:       {mean_iou:.4f}\n")

        # # ──────────────────────────────────
        # # 3) 시각화 (optional)
        # # ──────────────────────────────────
        # save_prediction_visuals(
        #     dataset_dicts,
        #     outputs,
        #     save_dir=os.path.join(cfg.OUTPUT_DIR, "visualizations"),
        #     metadata_name=cfg.DATASETS.TEST[0],
        # )
        # return


    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=cfg.MODEL.WEIGHTS != "")
    trainer.train()
    # === 학습 끝난 뒤 그래프 플롯 ===
    plot_curves(trainer, cfg.OUTPUT_DIR)
    return


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    args.resume = False  # ← 수동 override
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )