import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from timm import create_model
from .utils import *

# ViT "diffusion" block adapter for fused features (assuming timm ViT backbone)
class DiffusionViTLayer(nn.Module):
    def __init__(self, in_channels, fuse_channels, img_size=128, patch_size=16, vit_type='vit_base_patch16_224'):
        super().__init__()
        self.pre_conv = nn.Conv2d(fuse_channels, 3, kernel_size=1)  # Project fused features to 3 channels
        self.vit = create_model(vit_type, pretrained=True, img_size=(128, 128))
        self.vit.head = nn.Identity()  # Remove classifier head
        self.img_size = img_size

    def forward(self, x):
        # x: (B, fuse_channels, H, W) -> (B, 3, H, W)
        x = self.pre_conv(x)
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        # timm ViT expects (B, 3, img_size, img_size)
        # forward_features returns patch tokens (B, N, D)
        vit_tokens = self.vit.forward_features(x)  # (B, N, D)
        # patch_tokens = vit_tokens[:, 1:, :]  # exclude cls token, now (B, N-1, D)
        return vit_tokens

# Decode ViT tokens into 4 spatial keypoint heatmaps
class KeypointHeatmapHead(nn.Module):
    def __init__(self, token_dim, num_patches, num_keypoints=4, heatmap_size=8):
        super().__init__()
        self.fc = nn.Linear(token_dim * num_patches, num_keypoints * heatmap_size * heatmap_size)
        self.num_keypoints = num_keypoints
        self.heatmap_size = heatmap_size

    def forward(self, x):
        # x: (B, N_patches, D)
        B = x.shape[0]
        flat = x.flatten(1)  # (B, N_patches * D)
        y = self.fc(flat)    # (B, num_keypoints * heatmap_size*heatmap_size)
        y = y.view(B, self.num_keypoints, self.heatmap_size, self.heatmap_size)
        return y

# Full model
class YoloViTDiffusionKeypointNet(nn.Module):
    def __init__(self, backbone, in_channels_list, num_keypoints=4, output_shape=(128,128), vit_img_size=128, vit_patch_size=16):
        super().__init__()
        self.backbone = backbone
        self.fusion = MultiScaleFusion(in_channels_list, 128)      # use your existing MultiScaleFusion
        self.diffusion = DiffusionViTLayer(in_channels_list, 128, img_size=vit_img_size, patch_size=vit_patch_size)
        self.num_patches = (vit_img_size // vit_patch_size) ** 2
        token_dim = 768
        heatmap_size = 8
        self.head = KeypointHeatmapHead(token_dim, self.num_patches, num_keypoints=num_keypoints, heatmap_size=heatmap_size)
        self.output_shape = output_shape

    def forward(self, x):
        features = self.backbone(x)
        fused_features = self.fusion(features)
        vit_tokens = self.diffusion(fused_features)
        patch_tokens = vit_tokens[:, 1:, :]
        heatmaps = self.head(patch_tokens)
        if heatmaps.shape[-2:] != self.output_shape:
            heatmaps = F.interpolate(heatmaps, size=self.output_shape, mode='bilinear', align_corners=False)
        return heatmaps


# # Example usage (commented out)
# if __name__ == "__main__":
#     yolo_model = YOLO('yolov8l.pt')
#     backbone_seq = yolo_model.model.model[:10]
#     backbone = YoloBackbone(backbone_seq, selected_indices=[0,1,2,3,4,5,6,7,8,9])
#     input_dummy = torch.randn(1, 3, 128, 128)
#     feats = backbone(input_dummy)
#     in_channels_list = [f.shape[1] for f in feats]
#     net = YoloViTDiffusionKeypointNet(backbone, in_channels_list)
#     x = torch.randn(2, 3, 128, 128)
#     out = net(x)
#     print('Heatmaps shape:', out.shape)
#     kp = soft_argmax(out)
#     print("keypoint shape:", kp.shape)
