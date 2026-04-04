"""
U-Net architectures for composite von Mises stress field prediction.

Classes
-------
UNet
    Standard U-Net encoder-decoder (6 blocks, as in the reference paper).
    Adapted from biomedical segmentation to scalar field regression.

AttentionGate
    Soft attention gate applied to skip connections.
    Highlights fiber-matrix interface regions where stress concentrations occur.

AttentionUNet
    U-Net with attention gates on every skip connection.
    This is the primary enhanced architecture for the project.

Factory
-------
build_model(arch, in_channels, out_channels) -> nn.Module
    Convenience factory used by train_unet.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import resnet34, ResNet34_Weights
except Exception:  # pragma: no cover - optional dependency at runtime
    resnet34 = None
    ResNet34_Weights = None


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    """Two (Conv → BN → ReLU) layers."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Transposed-convolution upsampling followed by ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch * 2, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if sizes mismatch (e.g. odd-sized inputs)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear",
                              align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UpNoSkipBlock(nn.Module):
    """Transposed-convolution upsampling without skip connections."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


# ---------------------------------------------------------------------------
# Standard U-Net
# ---------------------------------------------------------------------------
class UNet(nn.Module):
    """
    Standard U-Net encoder-decoder for stress field regression.

    Architecture mirrors the reference paper:
      Encoder : 4 ConvBlocks + MaxPool  (features: 64→128→256→512)
      Bottleneck : 1024 channels
      Decoder : 4 UpBlocks with skip connections
      Head : 1×1 Conv → out_channels

    Parameters
    ----------
    in_channels  : 1  (binary microstructure image)
    out_channels : 1  (von Mises stress map)
    base_features: starting channel count (default 64)
    dropout      : dropout probability in decoder blocks (default 0.0)
    """

    def __init__(self,
                 in_channels:   int   = 1,
                 out_channels:  int   = 1,
                 base_features: int   = 64,
                 dropout:       float = 0.0):
        super().__init__()
        f = base_features

        # Encoder
        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = ConvBlock(f,   f * 2)
        self.enc3 = ConvBlock(f*2, f * 4)
        self.enc4 = ConvBlock(f*4, f * 8)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(f * 8, f * 16)

        # Decoder
        self.dec4 = UpBlock(f * 16, f * 8,  dropout=dropout)
        self.dec3 = UpBlock(f * 8,  f * 4,  dropout=dropout)
        self.dec2 = UpBlock(f * 4,  f * 2,  dropout=dropout)
        self.dec1 = UpBlock(f * 2,  f,      dropout=dropout)

        self.head = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b  = self.bottleneck(self.pool(e4))

        d4 = self.dec4(b,  e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        return self.head(d1)


# ---------------------------------------------------------------------------
# Attention Gate
# ---------------------------------------------------------------------------
class AttentionGate(nn.Module):
    """
    Additive soft-attention gate (Oktay et al., 2018).

    Selectively amplifies skip-connection features at regions flagged by the
    decoder's gating signal — critical for focusing on fiber-matrix interfaces.

    Parameters
    ----------
    F_g : channels in gating signal  (from decoder)
    F_l : channels in skip connection (from encoder)
    F_int : intermediate channels (typically F_l // 2)
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        g : gating signal  (B, F_g, H', W') — from decoder (smaller spatial)
        x : skip features  (B, F_l, H,  W)  — from encoder
        """
        # Upsample gating signal to match encoder resolution if needed
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear",
                              align_corners=False)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        alpha = self.psi(self.relu(g1 + x1))  # (B, 1, H, W) ∈ [0, 1]
        return x * alpha                       # attended skip features


# ---------------------------------------------------------------------------
# Attention U-Net
# ---------------------------------------------------------------------------
class AttentionUNet(nn.Module):
    """
    U-Net with attention gates on every skip connection.

    Structural improvements over standard U-Net:
    • AttentionGate on each skip connection → focuses decoder on high-stress zones
    • Dropout2d in decoder → regularisation
    """

    def __init__(self,
                 in_channels:   int   = 1,
                 out_channels:  int   = 1,
                 base_features: int   = 64,
                 dropout:       float = 0.1):
        super().__init__()
        f = base_features

        # Encoder
        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = ConvBlock(f,   f * 2)
        self.enc3 = ConvBlock(f*2, f * 4)
        self.enc4 = ConvBlock(f*4, f * 8)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(f * 8, f * 16)

        # Attention gates
        # F_g = channels of the gating signal (output of ConvTranspose2d up layers)
        # up4: f*16 → f*8, up3: f*8 → f*4, up2: f*4 → f*2, up1: f*2 → f
        self.ag4 = AttentionGate(F_g=f*8,  F_l=f*8,  F_int=f*4)
        self.ag3 = AttentionGate(F_g=f*4,  F_l=f*4,  F_int=f*2)
        self.ag2 = AttentionGate(F_g=f*2,  F_l=f*2,  F_int=f)
        self.ag1 = AttentionGate(F_g=f,    F_l=f,    F_int=f//2)

        # Decoder
        self.up4   = nn.ConvTranspose2d(f*16, f*8, kernel_size=2, stride=2)
        self.conv4 = ConvBlock(f*16, f*8,  dropout=dropout)

        self.up3   = nn.ConvTranspose2d(f*8, f*4, kernel_size=2, stride=2)
        self.conv3 = ConvBlock(f*8,  f*4,  dropout=dropout)

        self.up2   = nn.ConvTranspose2d(f*4, f*2, kernel_size=2, stride=2)
        self.conv2 = ConvBlock(f*4,  f*2,  dropout=dropout)

        self.up1   = nn.ConvTranspose2d(f*2, f,   kernel_size=2, stride=2)
        self.conv1 = ConvBlock(f*2,  f,    dropout=dropout)

        self.head  = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with attention gates
        d4 = self.up4(b)
        if d4.shape[2:] != e4.shape[2:]:
            d4 = F.interpolate(d4, size=e4.shape[2:], mode="bilinear",
                               align_corners=False)
        e4_att = self.ag4(g=d4, x=e4)
        d4 = self.conv4(torch.cat([d4, e4_att], dim=1))

        d3 = self.up3(d4)
        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode="bilinear",
                               align_corners=False)
        e3_att = self.ag3(g=d3, x=e3)
        d3 = self.conv3(torch.cat([d3, e3_att], dim=1))

        d2 = self.up2(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode="bilinear",
                               align_corners=False)
        e2_att = self.ag2(g=d2, x=e2)
        d2 = self.conv2(torch.cat([d2, e2_att], dim=1))

        d1 = self.up1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode="bilinear",
                               align_corners=False)
        e1_att = self.ag1(g=d1, x=e1)
        d1 = self.conv1(torch.cat([d1, e1_att], dim=1))

        return self.head(d1)


# ---------------------------------------------------------------------------
# ResNet34 Attention U-Net
# ---------------------------------------------------------------------------
class ResNet34AttentionUNet(nn.Module):
    """
    Attention U-Net decoder on top of a ResNet34 encoder backbone.

    Notes
    -----
    - Input is single-channel (binary microstructure). The first conv in ResNet
      is adapted from 3 channels to 1 by averaging pretrained RGB filters.
    - A final upsampling stage restores output to full input resolution.
    """

    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 base_features: int = 64,
                 dropout: float = 0.1,
                 pretrained: bool = True):
        super().__init__()
        if base_features != 64:
            raise ValueError("ResNet34AttentionUNet currently requires base_features=64.")
        if resnet34 is None:
            raise ImportError(
                "torchvision is required for arch='resnet34_attention_unet'. "
                "Install torchvision and retry."
            )

        weights = None
        if pretrained:
            if ResNet34_Weights is not None:
                weights = ResNet34_Weights.IMAGENET1K_V1
            else:
                weights = "IMAGENET1K_V1"

        enc = resnet34(weights=weights)

        # Stem: adapt first conv to user-specified input channels.
        old_conv1 = enc.conv1
        if in_channels != old_conv1.in_channels:
            new_conv1 = nn.Conv2d(
                in_channels,
                old_conv1.out_channels,
                kernel_size=old_conv1.kernel_size,
                stride=old_conv1.stride,
                padding=old_conv1.padding,
                bias=False,
            )
            with torch.no_grad():
                if old_conv1.weight.shape[1] == 3 and in_channels == 1:
                    new_conv1.weight.copy_(old_conv1.weight.mean(dim=1, keepdim=True))
                else:
                    nn.init.kaiming_normal_(new_conv1.weight, mode="fan_out", nonlinearity="relu")
            enc.conv1 = new_conv1

        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)   # 64x64
        self.maxpool = enc.maxpool                                 # 32x32
        self.layer1 = enc.layer1                                   # 32x32, 64ch
        self.layer2 = enc.layer2                                   # 16x16, 128ch
        self.layer3 = enc.layer3                                   # 8x8,  256ch
        self.layer4 = enc.layer4                                   # 4x4,  512ch

        self.ag4 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.ag3 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.ag2 = AttentionGate(F_g=64,  F_l=64,  F_int=32)
        self.ag1 = AttentionGate(F_g=64,  F_l=64,  F_int=32)

        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.cv4 = ConvBlock(512, 256, dropout=dropout)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.cv3 = ConvBlock(256, 128, dropout=dropout)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.cv2 = ConvBlock(128, 64, dropout=dropout)

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.cv1 = ConvBlock(128, 64, dropout=dropout)

        self.up0 = UpNoSkipBlock(64, 32, dropout=dropout)
        self.head = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)
        p0 = self.maxpool(s0)

        e1 = self.layer1(p0)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        b = self.layer4(e3)

        d4 = self.up4(b)
        if d4.shape[2:] != e3.shape[2:]:
            d4 = F.interpolate(d4, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d4 = self.cv4(torch.cat([d4, self.ag4(d4, e3)], dim=1))

        d3 = self.up3(d4)
        if d3.shape[2:] != e2.shape[2:]:
            d3 = F.interpolate(d3, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.cv3(torch.cat([d3, self.ag3(d3, e2)], dim=1))

        d2 = self.up2(d3)
        if d2.shape[2:] != e1.shape[2:]:
            d2 = F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.cv2(torch.cat([d2, self.ag2(d2, e1)], dim=1))

        d1 = self.up1(d2)
        if d1.shape[2:] != s0.shape[2:]:
            d1 = F.interpolate(d1, size=s0.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.cv1(torch.cat([d1, self.ag1(d1, s0)], dim=1))

        d0 = self.up0(d1)
        out = self.head(d0)
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return out


# ---------------------------------------------------------------------------
# Simple CNN baseline (no skip connections)
# ---------------------------------------------------------------------------
class SimpleCNN(nn.Module):
    """Compact encoder-decoder CNN baseline without skip connections."""

    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 base_features: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        f = max(16, base_features // 2)

        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = ConvBlock(f, f * 2)
        self.enc3 = ConvBlock(f * 2, f * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(f * 4, f * 8, dropout=dropout)

        self.up3 = UpNoSkipBlock(f * 8, f * 4, dropout=dropout)
        self.up2 = UpNoSkipBlock(f * 4, f * 2, dropout=dropout)
        self.up1 = UpNoSkipBlock(f * 2, f, dropout=dropout)
        self.head = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.enc1(x)
        x1 = self.enc2(self.pool(x0))
        x2 = self.enc3(self.pool(x1))
        b = self.bottleneck(self.pool(x2))

        y = self.up3(b)
        y = self.up2(y)
        y = self.up1(y)
        out = self.head(y)
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_model(arch: str = "attention_unet",
                in_channels: int = 1,
                out_channels: int = 1,
                base_features: int = 64,
                dropout: float = 0.1,
                resnet_pretrained: bool = True) -> nn.Module:
    """
    Parameters
    ----------
    arch : "unet" | "attention_unet" | "resnet34_attention_unet" | "cnn"

    Returns
    -------
    nn.Module
    """
    if arch == "unet":
        return UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_features=base_features,
            dropout=dropout,
        )
    if arch == "attention_unet":
        return AttentionUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_features=base_features,
            dropout=dropout,
        )
    if arch == "resnet34_attention_unet":
        return ResNet34AttentionUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_features=base_features,
            dropout=dropout,
            pretrained=resnet_pretrained,
        )
    if arch == "cnn":
        return SimpleCNN(
            in_channels=in_channels,
            out_channels=out_channels,
            base_features=base_features,
            dropout=dropout,
        )
    raise ValueError(
        f"Unknown architecture '{arch}'. Choose from: "
        f"['unet', 'attention_unet', 'resnet34_attention_unet', 'cnn']"
    )
