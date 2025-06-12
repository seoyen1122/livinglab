import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import hydra
from channelvit.meta_arch import Supervised
import channelvit.data as data
from omegaconf import OmegaConf, open_dict



@torch.no_grad()
def run_inference(cfg):
    # Load model from checkpoint
    print(f"[INFO] Loading model from checkpoint: {cfg.checkpoint}")
    model = Supervised.load_from_checkpoint(cfg.checkpoint, cfg=cfg)
    model.eval()
    model.cuda()

    # Load dataset (직접 ARDataset 호출)
    val_data_cfg_key = next(iter(cfg.val_data_dict.keys()))
    with open_dict(cfg.val_data_dict):
        val_data_cfg = cfg.val_data_dict[val_data_cfg_key]    
        
    dataset = data.ARDataset(
        root=val_data_cfg.args.root,
        split=val_data_cfg.args.split,
        transform_cfg=val_data_cfg.args.transform_cfg,
        channels=val_data_cfg.args.channels,
        name=val_data_cfg.name if "name" in val_data_cfg else None,
        loader=val_data_cfg.loader if "loader" in val_data_cfg else None,
    )

    loader = DataLoader(
        dataset,
        batch_size=val_data_cfg.loader.batch_size,
        shuffle=val_data_cfg.loader.shuffle,
        num_workers=val_data_cfg.loader.num_workers,
        collate_fn=dataset.collate_fn,
    )

    os.makedirs(cfg.output_dir, exist_ok=True)

    for batch_idx, (imgs, covariates) in enumerate(loader):
        imgs = imgs.cuda()
        covariates = {k: v.cuda() for k, v in covariates.items()}

        output = model.backbone(imgs, covariates)

        # Reshape based on output shape
        if output.dim() == 2:
            output = output[:, :, None, None]
        elif output.dim() == 3:
            N, D = output.shape[1], output.shape[2]
            patch_size = cfg.meta_arch.backbone.args.patch_size
            H = imgs.shape[2] // patch_size
            W = imgs.shape[3] // patch_size
            output = output.permute(0, 2, 1).view(imgs.size(0), D, H, W)

        logits = model.classifier(output)
        preds = torch.argmax(logits, dim=1)  # (B, H, W)

        for i in range(preds.shape[0]):
            pred_mask = preds[i].cpu().numpy().astype(np.uint8) * 255
            save_path = os.path.join(cfg.output_dir, f"mask_{batch_idx}_{i}.png")
            Image.fromarray(pred_mask).save(save_path)
            pred_np = pred_mask.astype(np.uint8)            
            print("Predicted mask shape:", pred_np.shape)
            print("Unique values in predicted mask:", np.unique(pred_np))
            print(f"[INFO] Saved: {save_path}")


@hydra.main(version_base=None, config_path="../ChannelViT/channelvit/config", config_name="main_supervised")
def main(cfg):
    assert cfg.checkpoint is not None, "Please specify cfg.checkpoint"
    if not hasattr(cfg, "output_dir"):
        cfg.output_dir = "./seg_output"
    run_inference(cfg)


if __name__ == "__main__":
    main()

