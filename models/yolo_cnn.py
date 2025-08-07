import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

class YoloBackbone(nn.Module):
    def __init__(self, backbone_seq, selected_indices=[4, 7, 9]):
        super().__init__()
        self.backbone = backbone_seq
        self.selected_indices = selected_indices

    def forward(self, x):
        feats = []
        out = x
        for i, layer in enumerate(self.backbone):
            out = layer(out)
            if i in self.selected_indices:
                feats.append(out)
        return feats

class MultiScaleFusion(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.reduce_convs = nn.ModuleList([
            nn.Conv2d(in_c, out_channels, 1) for in_c in in_channels_list
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(len(in_channels_list) * out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        target_size = features[0].shape[-2:]
        upsampled = []
        for conv, feat in zip(self.reduce_convs, features):
            out = conv(feat)
            if out.shape[-2:] != target_size:
                out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
            upsampled.append(out)
        fused = torch.cat(upsampled, dim=1)
        return self.fuse(fused)

class EnhancedYoloKeypointNet(nn.Module):
    def __init__(self, backbone, in_channels_list, num_keypoints=4, output_shape=(128,128)):
        super().__init__()
        self.backbone = backbone
        self.fusion = MultiScaleFusion(in_channels_list, 256)
        self.head = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_keypoints, 3, padding=1)
        )
        self.output_shape = output_shape

    def forward(self, x):
        features = self.backbone(x)
        fused_features = self.fusion(features)
        y = self.head(fused_features)  # (B, 4, H, W)
        if y.shape[-2:] != self.output_shape:
            y = F.interpolate(y, size=self.output_shape, mode='bilinear', align_corners=False)
        return y  # (B, 4, H, W)


import torch.nn.functional as F

def soft_argmax(heatmap, beta=100):
    # heatmap: (B, K, H, W)
    *rest, H, W = heatmap.shape
    heatmap_flat = heatmap.view(-1, H*W)  # (B*K, H*W)
    y_soft = F.softmax(heatmap_flat * beta, dim=1)

    # Create meshgrid of coordinates
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=heatmap.device),
        torch.arange(W, device=heatmap.device),
        indexing='ij'
    )  # each shape (H, W)
    x_coords = x_coords.reshape(-1).float()  # (H*W,)
    y_coords = y_coords.reshape(-1).float()  # (H*W,)

    x = (y_soft * x_coords).sum(dim=1)
    y = (y_soft * y_coords).sum(dim=1)
    coords = torch.stack((x, y), dim=1)  # (B*K, 2)
    return coords.view(*rest, 2)


# if __name__ == "__main__":
#     # Load YOLOv8l and slice the first 10 layers
#     yolo11 = YOLO('yolo11l-seg.pt')  # Or yolo11m-seg.pt, yolo11x-seg.pt, etc.
#     backbone_seq = yolo11.model.model[:12]
#     # yolo_model = YOLO('yolov8l.pt')
#     # backbone_seq = yolo_model.model.model[:10]
#     # Initialize the backbone with selected indices for multi-scale features
#     backbone = YoloBackbone(backbone_seq, selected_indices=[0,1,2,3])
#     input_dummy = torch.randn(1, 3, 128, 128)
#     with torch.no_grad():
#         feats = backbone(input_dummy)
#     print("Feature shapes:", [f.shape for f in feats])
#     in_channels_list = [f.shape[1] for f in feats]

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     keypoint_net = EnhancedYoloKeypointNet(backbone, in_channels_list)
#     keypoint_net = keypoint_net.to(device)

#     x = torch.randn(2, 3, 128, 128).to(device)
#     out = keypoint_net(x)
#     out = soft_argmax(out)
#     print("Final output shape:", out.shape)
