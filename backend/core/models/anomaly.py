try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class nn:
        class Module:
            pass

class ConvAutoencoder(nn.Module if TORCH_AVAILABLE else object):
    """
    Convolutional Autoencoder for unsupervised visual defect anomaly detection.
    Trained on 'Normal' parts to reconstruct pristine surfaces.
    Defects yield high L2/L1 reconstruction residuals -> anomaly heatmap.
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        if not TORCH_AVAILABLE:
            return
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),  # 112
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),           # 56
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),          # 28
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),         # 14
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, 4, stride=2, padding=1), # 224
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon

    def compute_anomaly_map(self, x):
        with torch.no_grad():
            recon = self.forward(x)
            diff = torch.mean(torch.abs(x - recon), dim=1, keepdim=True)
            return diff

def build_anomaly_detector(backbone: str = "autoencoder"):
    if not TORCH_AVAILABLE:
        return None
    return ConvAutoencoder()
