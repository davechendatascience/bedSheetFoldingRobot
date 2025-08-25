import numpy as np
import os
from scipy.ndimage import label
import cv2

def depth_map_to_image(depth_map):
    """
    Convert a raw depth map (np.ndarray, any dtype, possibly >255) into an 8-bit image.
    Handles missing/zero values gracefully.
    Returns an uint8 grayscale image (0=far, 255=near).
    """
    valid = depth_map > 0
    if np.any(valid):
        dmin = np.min(depth_map[valid])
        dmax = np.max(depth_map)
        # Avoid division by zero
        if dmax > dmin:
            depth_img = ((depth_map - dmin) / (dmax - dmin) * 255)
        else:
            depth_img = np.zeros_like(depth_map)
    else:
        depth_img = np.zeros_like(depth_map)
    depth_img = np.clip(depth_img, 0, 255).astype(np.uint8)
    return depth_img

def thresholded_locations(data_2d, threshold):
    """
    Find centroids of connected components above a threshold.
    Args:
        data_2d: 2D numpy array (can be logits or probabilities).
        threshold: threshold for segmentation.
        from_logits: If True, applies sigmoid to convert logits to probabilities.
    Returns:
        List of centroid coordinates (as np arrays).
    """
    # If input is logits, convert to probability first!
    probs = data_2d

    # Create a binary mask
    thresholded_2d = (probs >= threshold).astype(np.uint8)

    structure = np.ones((3, 3), dtype=int)  # 8-connectivity
    labeled, num_features = label(thresholded_2d, structure=structure)
    centroids = []
    for mesh_label in range(1, num_features + 1):
        positions = np.argwhere(labeled == mesh_label)
        centroid = positions.mean(axis=0)
        centroids.append(centroid)
    return centroids

def extract_mask_compare(image, yolo_model_finetuned, allowed_classes):
    # image_name = os.path.basename(image_path)
    # 推論
    results = yolo_model_finetuned(image, task="segment")[0]

    # 原圖
    orig_img = image.copy()
    h, w = orig_img.shape[:2]

    # 空白遮罩
    mask_all = np.zeros((h, w), dtype=np.uint8)
    for r in results:
        if r.masks is None:
            continue
        masks = r.masks.data.cpu().numpy()     # [N, H_pred, W_pred]
        classes = r.boxes.cls.cpu().numpy()    # [N] 物件的類別 ID
        for m, cls_id in zip(masks, classes):
            if int(cls_id) not in allowed_classes:
                continue  # 跳過不在清單內的類別
            m = (m * 255).astype(np.uint8)
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_all = cv2.bitwise_or(mask_all, m)
    masked_image = orig_img.copy()
    masked_image[mask_all==0] = 0
    return mask_all