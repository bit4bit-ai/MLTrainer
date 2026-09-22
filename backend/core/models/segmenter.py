try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class nn:
        class Module:
            pass

class DoubleConv(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        if not TORCH_AVAILABLE:
            return
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNetSegmenter(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, in_channels: int = 3, num_classes: int = 4):
        super().__init__()
        if not TORCH_AVAILABLE:
            return
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_up1 = DoubleConv(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up2 = DoubleConv(128, 64)
        
        self.up3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv_up3 = DoubleConv(64, 32)
        
        self.outc = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        u1 = self.up1(x4)
        m1 = torch.cat([u1, x3], dim=1)
        c1 = self.conv_up1(m1)
        
        u2 = self.up2(c1)
        m2 = torch.cat([u2, x2], dim=1)
        c2 = self.conv_up2(m2)
        
        u3 = self.up3(c2)
        m3 = torch.cat([u3, x1], dim=1)
        c3 = self.conv_up3(m3)
        
        return self.outc(c3)

def build_segmenter(backbone: str = "unet", num_classes: int = 4):
    if not TORCH_AVAILABLE:
        return None
    return UNetSegmenter(num_classes=num_classes)
