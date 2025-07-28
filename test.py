import torch
import cv2
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# 配置路徑
checkpoint = "./sam2.1_hiera_large.pt"
model_cfg  = "./sam2.1/sam2.1_hiera_l.yaml"
# -*- coding: utf-8 -*-

import os
import cv2

from ultralytics import YOLO

# 載入分割模型
model = YOLO("yolov8m-seg.pt")

predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

def extract_mask_compare(image_path):
    image_name = os.path.basename(image_path)
    # 推論圖片
    results = model(image_path, conf=0.15)

    # 如果想儲存結果圖：
    box = None
    for result in results:
        for obj in result.summary():
            if obj["name"] == "bed":
                result.save(filename= "results/" + image_name.replace('.webp', "_output1.webp"))
                box = obj["box"]
    if box != None:
        input_point = np.array([[(box["x1"] + box["x2"])//2, (box["y1"]+box["y2"])//2]])
        input_label = np.array([1])
        # 載入圖像
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"無法載入圖像: {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True # 會自動選擇分數最高的
        )
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]
        best_score = scores[best_mask_idx]

        # save the best mask to file
        mask_filename = image_name.replace('.webp', "_mask.webp")
        cv2.imwrite("results/" + mask_filename, (best_mask * 255).astype(np.uint8))

img_files = os.listdir("bed-images")
for img_file in img_files:
    if img_file.endswith('.webp'):
        image_path = os.path.join("bed-images", img_file)
        extract_mask_compare(image_path)
        print(f"Processed {img_file}")