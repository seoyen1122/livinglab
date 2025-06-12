import os
import numpy as np
import xarray as xr

'''
# 설정
nc_txt_path = "/data0/aix23606/limseoyen/climatenet_txts/test.txt"
nc_root = "/data0/aix23606/limseoyen/climatenet_txts"
patch_save_root = "/data0/aix23606/limseoyen/climatenet_patches/test"
patch_size = 192  
stride = 192

# # ─── train.txt 안의 각 NetCDF(.nc) 상대경로를 읽어들임 ─────────────────────────
with open(nc_txt_path, "r") as f:
 nc_paths = [line.strip() for line in f if line.strip()]

for rel_path in nc_paths:
    abs_path = os.path.join(nc_root, rel_path)
    base     = os.path.splitext(os.path.basename(rel_path))[0]  # 예: "data-2005-06-03-01-1_0"

    try:
        # 1) NetCDF 열기
        ds = xr.open_dataset(abs_path)

        # 2) 4개 채널 변수 → (1, 768, 1152) → 첫 번째 시점만 취해서 (768, 1152)
        TMQ  = ds["TMQ"].values
        if TMQ.ndim == 3:
            TMQ = TMQ[0]   # (768, 1152)
        U850 = ds["U850"].values
        if U850.ndim == 3:
            U850 = U850[0]
        V850 = ds["V850"].values
        if V850.ndim == 3:
            V850 = V850[0]
        PSL  = ds["PSL"].values
        if PSL.ndim == 3:
            PSL = PSL[0]

#         # 3) LABELS는 이미 (768, 1152)
        labels_orig = ds["LABELS"].values  # (768, 1152)

#         # 4) 이진 마스크로 매핑: AR(2) → 1, 나머지(0,1) → 0
        labels_bin = np.where(labels_orig == 2, 1, 0).astype(np.uint8)  # (768, 1152)

#         # 5) (4, 768, 1152) 형태로 쌓기
        data_arr = np.stack([TMQ, U850, V850, PSL], axis=0)  # (4, 768, 1152)

        _, H, W = data_arr.shape
        patch_id = 0

#         # ─── 6) 슬라이딩 윈도우로 패치 자르기 ─────────────────────────────────
        for i in range(0, H - patch_size + 1, stride):
            for j in range(0, W - patch_size + 1, stride):
                # 6-1) 4채널 패치 (4, 192, 192)
                img_patch = data_arr[:, i : i + patch_size, j : j + patch_size]

#                 # 6-2) 이진 레이블 패치 (192, 192)
                lbl_patch = labels_bin[i : i + patch_size, j : j + patch_size]

#                 # ─── 7) 하나의 .npz 파일에 묶어서 저장 ────────────────────────────
                save_name = f"{base}_patch_{patch_id}.npz"
                save_path = os.path.join(patch_save_root, save_name)

                np.savez_compressed(
                    save_path,
                    TMQ   = img_patch[0],  # (192, 192)
                    U850  = img_patch[1],  # (192, 192)
                    V850  = img_patch[2],  # (192, 192)
                    PSL   = img_patch[3],  # (192, 192)
                    LABEL = lbl_patch      # (192, 192), {0,1}
                )
                patch_id += 1

        print(f"✅ {base}: {patch_id} patches saved → {patch_save_root}")

    except Exception as e:
        print(f"❌ Failed to process {abs_path}: {e}")
os.makedirs(patch_save_root, exist_ok=True)

with open(nc_txt_path, "r") as f:
    nc_paths = [line.strip() for line in f if line.strip()]

for rel_path in nc_paths:
    abs_path = os.path.join(nc_root, rel_path)
    base = os.path.splitext(os.path.basename(rel_path))[0]

    try:
        ds = xr.open_dataset(abs_path)
        data = np.stack([ds[var].values[0] for var in ["TMQ", "U850", "V850", "PSL"]], axis=0)  # (C, H, W)
        C, H, W = data.shape

        patch_id = 0
        for i in range(0, H - patch_size + 1, stride):
            for j in range(0, W - patch_size + 1, stride):
                patch = data[:, i:i+patch_size, j:j+patch_size]
                save_path = os.path.join(patch_save_root, f"{base}_patch_{patch_id}.npy")
                np.save(save_path, patch)
                patch_id += 1

        print(f"✅ {base}: {patch_id} patches saved.")
    except Exception as e:
        print(f"❌ Failed to process {abs_path}: {e}")

'''

'''
from glob import glob

for split in ["train", "val"]:
    patch_dir = f"/data0/aix23606/limseoyen/climatenet_patches/{split}"
    txt_path = f"/data0/aix23606/limseoyen/climatenet_txts/{split}_patches.txt"
    patch_files = sorted(glob(os.path.join(patch_dir, "*.npz")))

    with open(txt_path, "w") as f:
        for p in patch_files:
            f.write(p + "\n")

    print(f"📄 {split}: {len(patch_files)}개 경로 저장 완료 → {txt_path}")
'''


from glob import glob

patch_dir = f"/data0/aix23606/limseoyen/climatenet_patches/test"
txt_path = f"/data0/aix23606/limseoyen/climatenet_txts/val_patches1.txt"
patch_files = sorted(glob(os.path.join(patch_dir, "*.npz")))

with open(txt_path, "w") as f:
    for p in patch_files:
        f.write(p + "\n")

print(f"📄 test: {len(patch_files)}개 경로 저장 완료 → {txt_path}")