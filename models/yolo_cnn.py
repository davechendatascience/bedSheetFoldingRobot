import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

class YOLOv8TenLayerBackbone(nn.Module):
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

class EnhancedYOLOv8KeypointNet(nn.Module):
    def __init__(self, backbone, in_channels_list, out_channels=1):
        super().__init__()
        self.backbone = backbone
        self.fusion = MultiScaleFusion(in_channels_list, 256)
        self.head = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, padding=1)
        )

    def forward(self, x):
        features = self.backbone(x)
        fused_features = self.fusion(features)
        y = self.head(fused_features)
        if y.shape[-2:] != (128, 128):
            y = F.interpolate(y, size=(128, 128), mode='bilinear', align_corners=False)
        return y.squeeze(1)  # (B, 128, 128) if out_channels=1

if __name__ == "__main__":
    # Load YOLOv8l and slice the first 10 layers
    yolo_model = YOLO('yolov8l.pt')
    backbone_seq = yolo_model.model.model[:10]
    # Initialize the backbone with selected indices for multi-scale features
    backbone = YOLOv8TenLayerBackbone(backbone_seq, selected_indices=[0,1,2,3,4,5,6,7,8,9])
    input_dummy = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        feats = backbone(input_dummy)
    print("Feature shapes:", [f.shape for f in feats])
    in_channels_list = [f.shape[1] for f in feats]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    keypoint_net = EnhancedYOLOv8KeypointNet(backbone, in_channels_list)
    keypoint_net = keypoint_net.to(device)

    x = torch.randn(2, 3, 128, 128).to(device)
    out = keypoint_net(x)
    print("Final output shape:", out.shape)
