import torch.nn as nn

class ConvUpscaler(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Upscale 2×: 224 → 448
        self.up1 = nn.Sequential(
            nn.Conv2d(128, 128 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
        )

        # Upscale to 512 and refine
        self.up2 = nn.Sequential(
            nn.Conv2d(128, 64 * 4, 3, padding=1),
            nn.PixelShuffle(2),  # 448 → 896 (we'll crop or pad to 512 later)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_channels, 3, padding=1),
        )

    def forward(self, x):  # x: [B, 3, 224, 224]
        x = self.encoder(x)
        x = self.up1(x)
        x = self.up2(x)
        return x[:, :, :512, :512]  # crop to exactly 512×512
