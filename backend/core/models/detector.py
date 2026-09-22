try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class nn:
        class Module:
            pass

class TinyObjectDetector(nn.Module if TORCH_AVAILABLE else object):
    """
    Lightweight Single-Shot Grid Object Detector:
    Predicts (num_classes + 4 bounding box offsets + 1 objectness) per grid cell.
    """
    def __init__(self, num_classes: int = 4, grid_size: int = 7):
        super().__init__()
        if not TORCH_AVAILABLE:
            return
        self.grid_size = grid_size
        self.num_classes = num_classes
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d((grid_size, grid_size))
        )
        
        # Out channels per cell: 1 (objectness) + 4 (cx, cy, w, h) + num_classes
        out_channels = 5 + num_classes
        self.head = nn.Conv2d(256, out_channels, kernel_size=1)

    def forward(self, x):
        feat = self.backbone(x)
        out = self.head(feat)
        # Permute to (batch, grid_size, grid_size, out_channels)
        return out.permute(0, 2, 3, 1)

def build_detector(backbone: str = "tiny_yolo", num_classes: int = 4):
    if not TORCH_AVAILABLE:
        return None
    return TinyObjectDetector(num_classes=num_classes)
