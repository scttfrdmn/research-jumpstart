"""
Deep learning model architectures for medical imaging.

Includes:
- ResNet-50 for chest X-ray classification
- 3D CNN for CT scan analysis
- U-Net for MRI segmentation
"""


import torch
import torch.nn as nn
import torchvision.models as models


class ChestXrayClassifier(nn.Module):
    """
    ResNet-50 based multi-label classifier for chest X-rays.

    Pre-trained on ImageNet, fine-tuned for 14-class disease classification.
    """

    def __init__(self, num_classes: int = 14, pretrained: bool = True):
        """
        Args:
            num_classes: Number of disease classes
            pretrained: Use ImageNet pre-trained weights
        """
        super().__init__()

        # Load pre-trained ResNet-50
        self.resnet = models.resnet50(pretrained=pretrained)

        # Replace final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(num_features, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Logits of shape (B, num_classes)
        """
        return self.resnet(x)


class CTNoduleDetector(nn.Module):
    """
    3D ResNet-18 for CT nodule detection and classification.

    Processes 3D CT volumes to detect and classify lung nodules.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        """
        Args:
            in_channels: Number of input channels (1 for CT)
            num_classes: Number of output classes (benign/malignant)
        """
        super().__init__()

        # 3D convolutional layers
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # ResNet blocks
        self.layer1 = self._make_layer(64, 64, num_blocks=2)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(
        self, in_channels: int, out_channels: int, num_blocks: int, stride: int = 1
    ) -> nn.Sequential:
        """Create a residual layer."""
        layers = []

        # First block may downsample
        layers.append(ResBlock3D(in_channels, out_channels, stride))

        # Subsequent blocks
        for _ in range(1, num_blocks):
            layers.append(ResBlock3D(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, 1, D, H, W)

        Returns:
            Logits of shape (B, num_classes)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class ResBlock3D(nn.Module):
    """3D Residual Block for CTNoduleDetector."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.relu(out)

        return out


class MRISegmenter(nn.Module):
    """
    3D U-Net for brain tumor segmentation in MRI.

    Multi-modal input (T1, T1ce, T2, FLAIR) → 4-class segmentation output.
    """

    def __init__(self, in_channels: int = 4, num_classes: int = 4):
        """
        Args:
            in_channels: Number of MRI modalities (4 for BraTS)
            num_classes: Number of segmentation classes
        """
        super().__init__()

        # Encoder
        self.enc1 = self._conv_block(in_channels, 32)
        self.enc2 = self._conv_block(32, 64)
        self.enc3 = self._conv_block(64, 128)
        self.enc4 = self._conv_block(128, 256)

        # Bottleneck
        self.bottleneck = self._conv_block(256, 512)

        # Decoder
        self.dec4 = self._upconv_block(512, 256)
        self.dec3 = self._upconv_block(256, 128)
        self.dec2 = self._upconv_block(128, 64)
        self.dec1 = self._upconv_block(64, 32)

        # Output
        self.out = nn.Conv3d(32, num_classes, kernel_size=1)

        self.pool = nn.MaxPool3d(2)

    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Double convolution block."""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _upconv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Upsampling and convolution block."""
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, 4, D, H, W)

        Returns:
            Segmentation logits of shape (B, num_classes, D, H, W)
        """
        # Encoder with skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.dec4(b)
        d4 = torch.cat([d4, e4], dim=1)  # Skip connection

        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e3], dim=1)

        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e2], dim=1)

        d1 = self.dec1(d2)
        d1 = torch.cat([d1, e1], dim=1)

        # Output
        out = self.out(d1)

        return out


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """
    Count total and trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


if __name__ == "__main__":
    # Test model instantiation
    print("Medical Imaging Model Architectures")
    print("=" * 50)

    # X-ray classifier
    xray_model = ChestXrayClassifier(num_classes=14)
    total, trainable = count_parameters(xray_model)
    print("\\nChestXrayClassifier (ResNet-50):")
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print("  Input: (B, 3, 224, 224)")
    print("  Output: (B, 14)")

    # CT detector
    ct_model = CTNoduleDetector(in_channels=1, num_classes=2)
    total, trainable = count_parameters(ct_model)
    print("\\nCTNoduleDetector (3D ResNet-18):")
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print("  Input: (B, 1, 64, 64, 64)")
    print("  Output: (B, 2)")

    # MRI segmenter
    mri_model = MRISegmenter(in_channels=4, num_classes=4)
    total, trainable = count_parameters(mri_model)
    print("\\nMRISegmenter (3D U-Net):")
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print("  Input: (B, 4, 240, 240, 155)")
    print("  Output: (B, 4, 240, 240, 155)")

    print("\\n" + "=" * 50)
    print("Models ready for training!")
