# Evaluate the DINO embedding through linear probing
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader

import channelvit.backbone as backbone
import channelvit.data as data
import channelvit.utils as utils
import matplotlib.pyplot as plt
import os



#*decoderunet - patchsize = 16
class UNetLiteDecoder(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )
        # 4단계 업샘
        self.layer1 = nn.Sequential(conv_block(in_channels, 512),
                                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.layer2 = nn.Sequential(conv_block(512, 256),
                                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.layer3 = nn.Sequential(conv_block(256, 128),
                                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.layer4 = nn.Sequential(conv_block(128,  64),
                                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # x: (B, D, Hf, Wf)
        x = self.layer1(x)   # → (B,512, 2×Hf, 2×Wf)
        x = self.layer2(x)   # → (B,256, 4×Hf, 4×Wf)
        x = self.layer3(x)   # → (B,128, 8×Hf, 8×Wf)
        x = self.layer4(x)   # → (B, 64,16×Hf,16×Wf) == (B,64, H, W)
        return self.classifier(x)  # (B, num_classes, H, W)

# # #*decoderunet - patchsize = 8
# class UNetLiteDecoder(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super().__init__()
#         def conv_block(in_c, out_c):
#             return nn.Sequential(
#                 nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
#                 nn.BatchNorm2d(out_c),
#                 nn.ReLU(inplace=True),
#             )
#         # 4단계 업샘
#         self.layer1 = nn.Sequential(conv_block(in_channels, 256),
#                                     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
#         self.layer2 = nn.Sequential(conv_block(256, 128),
#                                     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
#         self.layer3 = nn.Sequential(conv_block(128, 64),
#                                     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
#         self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

#     def forward(self, x):
#         # x: (B, D, Hf, Wf)
#         x = self.layer1(x)   
#         x = self.layer2(x)   
#         x = self.layer3(x)   
#         return self.classifier(x)  



# # #*decoderunet - patchsize = 32
# class UNetLiteDecoder(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super().__init__()
#         def conv_block(in_c, out_c):
#             return nn.Sequential(
#                 nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
#                 nn.BatchNorm2d(out_c),
#                 nn.ReLU(inplace=True),
#             )
#         # 5단계 업샘
#         self.layer1 = nn.Sequential(conv_block(in_channels, 1024),
#                                     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
#         self.layer2 = nn.Sequential(conv_block(1024, 512),
#                                     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
#         self.layer3 = nn.Sequential(conv_block(512, 256),
#                                     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
#         self.layer4 = nn.Sequential(conv_block(256, 128),
#                                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
#         self.layer5 = nn.Sequential(conv_block(128,  64),
#                                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
#         self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

#     def forward(self, x):
#         # x: (B, D, Hf, Wf)
#         x = self.layer1(x)   
#         x = self.layer2(x)   
#         x = self.layer3(x)
#         x = self.layer4(x)   
#         x = self.layer5(x)   

#         return self.classifier(x)  



class Supervised(pl.LightningModule):
    def __init__(self, main_supervised_cfg: DictConfig) -> None:
        super().__init__()

        self.cfg = self.set_cfg(main_supervised_cfg)

        # load the data cfg from the root cfg
        self.val_data_cfg = main_supervised_cfg.val_data_dict

        # load the transformation cfg from the root cfg
        # note that in the transformation, we can play with different channel masking
        # baselines
        self.val_transform_cfg = main_supervised_cfg.val_transformations

        self.save_hyperparameters()

        # get the backbone
        self.backbone = getattr(backbone, self.cfg.backbone.name)(
            **self.cfg.backbone.args
        )

        #self.classifier = nn.Linear(self.backbone.embed_dim, self.cfg.num_classes)
        self.classifier = nn.Conv2d(
            in_channels=self.backbone.embed_dim,
            out_channels=self.cfg.num_classes,
            kernel_size=1,
            stride=1
        )

        #*decoderunet
        self.decoder = UNetLiteDecoder(
            in_channels=self.backbone.embed_dim,
            num_classes=self.cfg.num_classes
        )

        #*channel_fuser
        # D = self.backbone.embed_dim
        # C = self.backbone.in_chans

        # self.channel_fuser = nn.Conv2d(
        #    in_channels=D * C,
        #    out_channels=D,
        #    kernel_size=1
        # )

        self.register_buffer("class_weights", torch.tensor([1.0, 10.0], dtype=torch.float32))

        self.compute_loss = torch.nn.CrossEntropyLoss(
            weight=self.class_weights, label_smoothing=self.cfg.label_smoothing
        )

        # We treat the prediction target as one of the covariates. Here we specify the
        # prediction target key.
        self.target = self.cfg.target

        self.validation_step_outputs = []
        self.best_validation_results = {}

        self.configure_scheduler()
        self.val_losses = []



    def set_cfg(self, main_supervised_cfg: DictConfig) -> DictConfig:
        cfg = main_supervised_cfg.meta_arch

        # set the optimization configurations
        with open_dict(cfg):
            try:
                cfg.total_batch_size = (
                    main_supervised_cfg.trainer.devices
                    * main_supervised_cfg.train_data.loader.batch_size
                    * main_supervised_cfg.trainer.accumulate_grad_batches
                )
                cfg.num_batches = (
                    main_supervised_cfg.train_data.loader.num_batches
                    // main_supervised_cfg.trainer.accumulate_grad_batches
                )
                cfg.num_batches_original = (
                    main_supervised_cfg.train_data.loader.num_batches
                )
            except Exception as e:
                print(e)
                cfg.total_batch_size = (
                    main_supervised_cfg.trainer.devices
                    * main_supervised_cfg.train_data.loader.batch_size
                )
                cfg.num_batches = main_supervised_cfg.train_data.loader.num_batches
                cfg.num_batches_original = (
                    main_supervised_cfg.train_data.loader.num_batches
                )

            cfg.max_epochs = main_supervised_cfg.trainer.max_epochs
            cfg.backbone.patch_size = cfg.patch_size

        print(cfg)
        return cfg

    def val_dataloader(self):
        """Create the validation data loaders for the linear probing.
        """
        print("Loading the validation data loaders.")
        val_loaders = []
        val_data_name_list = []
        for val_data_name, val_data_cfg in self.val_data_cfg.items():
            val_data_name_list.append(val_data_name)
            print(f"Loading {val_data_name}")
            val_data = getattr(data, val_data_cfg.name)(
                #여기 원래 안쓰는것: is_train=False,
                #여기 원래 안쓰는것: transform_cfg=self.val_transform_cfg,
                **val_data_cfg.args,
            )
            val_loaders.append(
                DataLoader(
                    val_data, **val_data_cfg.loader, collate_fn=val_data.collate_fn
                )
            )

        self.val_data_name_list = val_data_name_list

        return val_loaders

    def train_single_batch(self, batch):
        imgs, covariates = batch
        B = imgs.shape[0]

        if "channels" not in covariates:
            covariates["channels"] = [list(range(imgs.shape[1]))] * B  #ChannelViT
        label = covariates[self.target]
        output = self.backbone(imgs, covariates)  # Could be [B, D], [B, N, D], or [B, D, H, W]
        B, N, D = output.shape
        C = imgs.shape[1]
        patch_size = self.cfg.backbone.patch_size
        H_patch = imgs.shape[2] // patch_size
        W_patch = imgs.shape[3] // patch_size
        print(B, N, D, C, H_patch, W_patch)
        assert N == C * H_patch * W_patch, f"N({N}) != C({C})*H_patch({H_patch})*W_patch({W_patch})"
        #output = output.permute(0, 2, 1).reshape(B, D, C*H_patch, W_patch)

        feat = output.permute(0,2,1).view(B, D, C, H_patch, W_patch)

        #*channel_fuser
        output = feat.mean(dim=2)  # [B, D, Hf, Wf]
        # feat = feat.reshape(B, D * C, H_patch, W_patch)
        # output = self.channel_fuser(feat)

        #*decoderunet
        # logits = self.classifier(logits)
        # logits = F.interpolate(logits, size=(192, 192), mode='bilinear', align_corners=False)
        logits = self.decoder(output)

        loss = self.compute_loss(logits, label)
        with torch.no_grad():
            pred = torch.argmax(logits, dim=1)
            print(f"[DEBUG] train pred unique: {torch.unique(pred)}")
        return loss, logits, label
    
    def training_step(self, batch, batch_idx):
        # We use manual optimization here
        # get the optimizer and set the lr / wd rate
        optimizer = self.optimizers(use_pl_optimizer=True)
        step_idx = min(self.global_step, len(self.lr_schedule) - 1)

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] =self.lr_schedule[step_idx]                    #= self.lr_schedule[self.global_step]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = self.wd_schedule[step_idx]            #= self.wd_schedule[self.global_step]

        # self.log("learning_rate", self.lr_schedule[self.global_step])
        # self.log("weight_decay", self.wd_schedule[self.global_step])
        self.log("learning_rate", self.lr_schedule[step_idx])
        self.log("weight_decay",  self.wd_schedule[step_idx])


        #train,val acc graph 출력하기 위해 주석 처리
        '''
        if type(batch) is dict:
            # We average the loss across all batches for all data loaders
            current_ratio = float(batch_idx) / self.cfg.num_batches_original
            for key, single_batch in batch.items():
                loss, logits, labels = self.train_single_batch(single_batch)
                break
            else:
                loss = self.train_single_batch(batch)
        else:
            loss = self.train_single_batch(batch)
        '''
        if isinstance(batch, dict):
            # 여러 데이터로더를 쓸 때 batch가 {'name': (imgs, covs), …} 형태인 경우
            _, single_batch = next(iter(batch.items()))
            loss, logits, labels = self.train_single_batch(single_batch)
        else:
            loss, logits, labels = self.train_single_batch(batch)


        self.log(f"train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, rank_zero_only=True)

        with torch.no_grad():
            pred = torch.argmax(logits, dim=1)
            labels = batch[1][self.target]
            train_acc = (pred == labels).float().mean()
        self.log("train_acc", train_acc, on_step=False, on_epoch=True, prog_bar=True, rank_zero_only=True)

        return loss

    def inference(self, batch, batch_idx):
        """Make predictions from the trained segmentation model"""
        imgs, covariates = batch
        B = imgs.shape[0]

        if "channels" not in covariates:
            covariates["channels"] = [list(range(imgs.shape[1]))] * B # ChannelViT

        output = self.backbone(imgs, covariates)  # (B, D, H, W) or (B, N, D), etc.
        print("output.shape:", output.shape)

        # output shape 변환 (예전 코드 그대로)
        if output.dim() == 2:
            output = output[:, :, None, None]
        elif output.dim() == 3:
            B, N, D = output.shape
            C = imgs.shape[1]
            patch_size = self.cfg.backbone.patch_size
            H_patch = imgs.shape[2] // patch_size
            W_patch = imgs.shape[3] // patch_size
            print(f"imgs.shape: {imgs.shape}, patch_size: {patch_size}")
            assert N == C * H_patch * W_patch, f"N({N}) != C({C})*H_patch({H_patch})*W_patch({W_patch})"
            # output = output.permute(0, 2, 1).reshape(B, D, C*H_patch, W_patch)
            feat = output.permute(0, 2, 1).view(B, D, C, H_patch, W_patch)

            #*channel_fuser
            output = feat.mean(dim=2)
            # feat = feat.reshape(B, D * C, H_patch, W_patch)
            # output = self.channel_fuser(feat)

        elif output.dim() == 4:
            pass  # output = (B, D, H, W)
        else:
            raise ValueError(f"[ERROR] Unexpected output shape: {output.shape}")
        print("output.shape =", output.shape)

        #*decoderunet
        # logits = self.classifier(output)
        # logits = F.interpolate(logits, size=(imgs.shape[2], imgs.shape[3]), mode='bilinear', align_corners=False)  # (B, 2, 192, 192)
        logits = self.decoder(output)  # (B,2,192,192)

        pred = torch.argmax(logits, dim=1)
        label = covariates[self.target]  # (B, 192, 192)이어야 함
        # label이 (B, H, W)여야 하며, 추가 전처리 필요 없음

        unique_labels = torch.unique(label, return_counts=True)
        print(f"[DEBUG] true label unique: {unique_labels}")

        return {"logit": logits, "pred": pred, "true": label}
    

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self.inference(batch, batch_idx)
        imgs, covariates = batch
        labels = covariates["label"]
        print(f"[VAL STEP] Input shape:  {imgs.shape}")
        print(f"[VAL STEP] Target shape: {labels.shape}") 
        output = self.backbone(imgs, covariates) 
        print(f"[VAL STEP] Output (logit) shape: {output.shape}")
        
        outputs["dataloader_idx"] = dataloader_idx
        self.validation_step_outputs.append(outputs)
        return outputs
    

    def on_validation_epoch_end(self):
        val_outputs_per_loader = {}

        for outputs in self.validation_step_outputs:
            idx = outputs["dataloader_idx"]
            if idx not in val_outputs_per_loader:
                val_outputs_per_loader[idx] = {"logit": [], "true": []}

            val_outputs_per_loader[idx]["logit"].append(outputs["logit"])
            val_outputs_per_loader[idx]["true"].append(outputs["true"])

        for loader_idx, outputs_dict in val_outputs_per_loader.items():
            # Concatenate along batch dimension
            all_preds = torch.cat(outputs_dict["logit"], dim=0)  # (B, C, H, W)
            all_trues = torch.cat(outputs_dict["true"], dim=0)   # (B, H, W)

            # Compute loss
            loss = self.compute_loss(all_preds, all_trues) #all_trues.long()

            # Compute pixel-wise accuracy
            pred_labels = torch.argmax(all_preds, dim=1)  # (B, H, W)
            acc = (pred_labels == all_trues).float().mean()
            iou1 = self.compute_IoU_for_class(pred_labels, all_trues, class_idx=1)
            if dist.is_initialized():
                dist.all_reduce(iou1)
                iou1 = iou1 / dist.get_world_size()
            print(f"[DEBUG] val_{loader_idx} IoU for class 1: {iou1.item()}")
            self.log(
                f"val_{loader_idx}_IoU_class1",
                iou1,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                rank_zero_only=False,
            )
          


            # Distributed sync
            if dist.is_initialized():
                dist.all_reduce(loss)
                dist.all_reduce(acc)
                world_size = dist.get_world_size()
                loss = loss / world_size
                acc = acc / world_size
            '''
            self.log(
                f"val_{loader_idx}_acc",
                acc,
                on_epoch=True,
                prog_bar=True,
                rank_zero_only=True,
            )
            self.log(
                f"val_{loader_idx}_loss",
                loss,
                on_epoch=True,
                prog_bar=False,
                rank_zero_only=True,
            )
            '''
            self.log(f"val_{loader_idx}_loss", loss, on_epoch=True, prog_bar=False, rank_zero_only=True)
            self.log(f"val_{loader_idx}_acc",  acc,  on_epoch=True, prog_bar=True,  rank_zero_only=True)
        self.log("val_loss", loss, on_epoch=True, rank_zero_only=True)
        self.log("val_acc",  acc,  on_epoch=True, rank_zero_only=True)
        self.val_losses.append(loss.item())
        self.validation_step_outputs.clear()  # free memory


    def compute_mIoU(self, pred, target, num_classes):
        ious = []
        device = pred.device

        for cls in range(num_classes):
            pred_inds = (pred == cls)
            target_inds = (target == cls)

            intersection = (pred_inds & target_inds).sum().float()
            union = (pred_inds | target_inds).sum().float()

            if union == 0:
                ious.append(torch.tensor(1.0 if intersection == 0 else 0.0, device=device))
            else:
                ious.append(intersection / union)
        print("[DEBUG] compute_mIoU pred unique:", torch.unique(pred))
        print("[DEBUG] compute_mIoU true unique:", torch.unique(target))
        values, counts = torch.unique(pred, return_counts=True)
        for v, c in zip(values.tolist(), counts.tolist()):
            print(f"Class {v}: {c} pixels")

        return torch.mean(torch.stack(ious))
    
    def compute_IoU_for_class(self, pred: torch.Tensor, target: torch.Tensor, class_idx: int) -> torch.Tensor:
        """
        pred, target: shape [B, H, W], dtype=torch.int
        class_idx: IoU를 구할 클래스 인덱스 (여기서는 1)
        return: scalar IoU tensor
        """
        device = pred.device
        pred_i = (pred == class_idx)
        target_i = (target == class_idx)

        intersection = (pred_i & target_i).sum().float()
        union        = (pred_i | target_i).sum().float()

        if union == 0:
            # 해당 클래스가 전혀 없던 경우: 완벽 매칭이면 1.0, 아니면 0.0
            return torch.tensor(1.0 if intersection == 0 else 0.0, device=device)
        else:
            return intersection / union


    def predict_step(self, batch, batch_idx):
        return self.inference(batch, batch_idx)

    def configure_optimizers(self):
        """Loading optimizer and learning rate / weight decay schedulers"""

        params_groups = utils.get_params_groups(self.backbone)
        params_groups[0]["params"] += self.classifier.parameters()
        #*channel_fuser
        # params_groups[0]["params"] += list(self.channel_fuser.parameters())
        #decoderunet
        params_groups[0]["params"] += list(self.decoder.parameters())



        print("[DEBUG] classifier parameters added to optimizer?")
        for name, param in self.classifier.named_parameters():
            print(f"  - {name}, requires_grad: {param.requires_grad}, mean: {param.data.mean().item()}")

        print("Creating optimizer.")
        if self.cfg.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
        else:
            raise ValueError("DINO only supports optimizer adaw, sgd and lars.")

        return optimizer

    def configure_scheduler(self) -> None:
        print("Creating learning rate, weight decay and momentum scheduler.")
        # Note that these schedulers are not the typical pytorch schedulers. they are
        # just simple np arrays. The length of each array equals to the total number of
        # steps.
        print(self.cfg.lr)
        print(self.cfg.total_batch_size)
        self.lr_schedule = utils.cosine_scheduler(
            self.cfg.lr * self.cfg.total_batch_size / 256.0,  # linear scaling rule
            self.cfg.min_lr,
            self.cfg.max_epochs,
            self.cfg.num_batches,
            warmup_epochs=self.cfg.warmup_epochs,
        )

        self.wd_schedule = utils.cosine_scheduler(
            self.cfg.weight_decay,
            self.cfg.weight_decay_end,
            self.cfg.max_epochs,
            self.cfg.num_batches,
        )


# 그래프 출력 함수

class MetricsPlotCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses   = []
        self.train_accs   = []
        self.val_accs     = []

    def on_train_epoch_end(self, trainer, pl_module):
        # Lightning이 내부에 모아둔 train_loss, train_acc를 가져와서 리스트에 추가
        metrics = trainer.callback_metrics
        # callback_metrics 에는 'train_loss' 와 'train_acc' 가 on_epoch=True 로 기록되어 있어야 합니다
        self.train_losses.append(metrics["train_loss"].item())
        self.train_accs.append( metrics["train_acc"].item() )

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        val_l = metrics.get("val_loss") or metrics.get("val_0_loss")
        val_a = metrics.get("val_acc")  or metrics.get("val_0_acc")
        if val_l is not None and val_a is not None:
            self.val_losses.append(val_l.item())
            self.val_accs .append(val_a.item())
        # self.val_losses.append(metrics["val_loss"].item())
        #self.val_accs.append( metrics["val_acc"].item() )


    def on_train_end(self, trainer, pl_module):
        out_dir = trainer.default_root_dir
        os.makedirs(out_dir, exist_ok=True)
        epochs_train = range(1, len(self.train_losses) + 1)
        epochs_val   = range(1, len(self.val_losses) + 1)
        
        # ——— Loss 그래프 ———
        loss_path = os.path.join(out_dir, "metrics_loss.png")
        plt.figure()
        plt.plot(epochs_train, self.train_losses, '-o', label="Train Loss")
        plt.plot(epochs_val, self.val_losses,   '-o', label="Val   Loss")
        plt.title("Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        plt.savefig(loss_path)
        plt.close()

        # ——— Accuracy 그래프 ———
        acc_path = os.path.join(out_dir, "metrics_acc.png")
        plt.figure()
        plt.plot(epochs_train, self.train_accs, '-o', label="Train Acc")
        plt.plot(epochs_val, self.val_accs,   '-o', label="Val   Acc")
        plt.title("Accuracy per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(acc_path)
        plt.close()

        print(f"\nMetrics plots saved:\n   - {loss_path}\n   - {acc_path}\n")