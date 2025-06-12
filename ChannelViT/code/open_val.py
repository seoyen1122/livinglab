import os
import glob

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

test_list = "/data0/aix23606/limseoyen/climatenet_txts/climatenet_full_test/"
out_folder = "/path/to/save/labels_png"
for file in os.listdir(val_file_list):
    ds = xr.open_dataset(os.path.join(val_file_list, file))
    print(file, np.unique(ds["LABELS"].values))


# 데이터셋을 직접 확인하는 코드 예시
for file in os.listdir(train_file_list):
    ds = xr.open_dataset(os.path.join(train_file_list, file))
    label = ds["LABELS"].values
    print("GT unique values:", np.unique(label))  # → [0, 2] 또는 [1]