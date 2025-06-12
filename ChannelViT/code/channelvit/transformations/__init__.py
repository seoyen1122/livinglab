# from channelvit.transformations.camelyon_dino import CamelyonAugmentationDino
# from channelvit.transformations.cell import CellAugmentation
# from channelvit.transformations.cell_dino import CellAugmentationDino
# from channelvit.transformations.imagenet_dino import ImagenetAugmentationDino
# from channelvit.transformations.rgb import RGBAugmentation
# from channelvit.transformations.so2sat import So2SatAugmentation
# from channelvit.transformations.basic import basic


def basic(is_train=True, **kwargs):
    # transform 정의 예시
    import torch
    import torchvision.transforms as T

    if is_train:
        return T.Compose([
            T.RandomHorizontalFlip(),
        ])
    else:
        return T.Compose([])

# transformations/basic.py

# def basic(is_train=False, per_channel_min=None, per_channel_max=None, **kwargs):
#     def _transform(x):
#         # x: (C, H, W)
#         if per_channel_min is not None and per_channel_max is not None:
#             import torch
#             mins = torch.tensor(per_channel_min, device=x.device)[:, None, None]
#             maxs = torch.tensor(per_channel_max, device=x.device)[:, None, None]
#         else:
#             mins = x.amin(dim=(1,2), keepdim=True)
#             maxs = x.amax(dim=(1,2), keepdim=True)
#         x = (x - mins) / (maxs - mins + 1e-8)
#         return x
#     return _transform



