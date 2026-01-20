import numpy as np
import cv2


for i in range(60):
    img_path = f"/media/hc/DataDisk/data/Rally/detect/int8_calinrator_data_640/{i:04d}.png"
    save_path = f"/media/hc/DataDisk/data/Rally/detect/int8_calinrator_data_224/{i:04d}.png"
    img = cv2.imread(img_path)
    img = cv2.resize(img,(224, 224))
    cv2.imwrite(save_path, img)