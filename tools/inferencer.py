from opencd.apis import OpenCDInferencer
from typing import Union
import cv2
import numpy as np

config_path = 'D:/git/open-cd/configs/FCCDN/FCCDN_512x512_40k_levircd.py'
checkpoint_path = 'D:/git/open-cd/logs/fccdn/4/iter_40000.pth'

# config_path = 'D:/git/open-cd-main/configs/FCCDN/FCCDN_256x256_40k_my-data.py'
# checkpoint_path = 'D:/git/open-cd-main/logs/fccdn/3/best_mIoU_iter_33000.pth'

inferencer = OpenCDInferencer(
    model = config_path,
    weights = checkpoint_path,
    dataset_name='my_Dataset',
    device='cuda:0')


# img = [['E:/changeDectect/train_with_seg/val/A/3017.tif', 'E:/changeDectect/train_with_seg/val/B/3017.tif']]
img = [['E:/LEVIR_CD/test/A/test_7.png', 'E:/LEVIR_CD/test/B/test_7.png']]
# print(type(img))
a = inferencer(img, show=True)
# print(a)
# print(np.unique(a['predictions']))

cv2.imwrite("D:/git/open-cd-main/outputs/1.png", a['predictions'] * 255)
cv2.imwrite("D:/git/open-cd-main/outputs/seg1.png", a['i_seg1_pred'] * 255)
cv2.imwrite("D:/git/open-cd-main/outputs/seg2.png", a['i_seg2_pred'] * 255)
# cv2.waitKey(0)
