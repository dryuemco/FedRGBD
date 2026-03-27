"""FedRGBD — MobileNetV3-Small for fire classification."""

import torch
import torch.nn as nn
import torchvision


def create_model(num_classes=2, in_channels=3, pretrained=True):
    """Create MobileNetV3-Small with configurable input channels.
    
    Args:
        num_classes: Number of output classes (2 for fire/nofire)
        in_channels: Input channels (3=RGB, 4=RGB+D, 5=RGB+D+IR)
        pretrained: Use ImageNet pretrained weights
    """
    weights = "IMAGENET1K_V1" if pretrained else None
    model = torchvision.models.mobilenet_v3_small(
        weights=weights, num_classes=1000
    )
    
    # Modify first conv if not 3 channels
    if in_channels != 3:
        old_conv = model.features[0][0]
        new_conv = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        # Copy pretrained weights for first 3 channels
        if pretrained:
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                # Initialize extra channels with mean of RGB weights
                for i in range(3, in_channels):
                    new_conv.weight[:, i] = old_conv.weight.mean(dim=1)
        model.features[0][0] = new_conv
    
    # Replace classifier head
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    
    return model


def get_model_size(model):
    """Get model size in MB."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    return param_size / 1e6


if __name__ == "__main__":
    for ch, name in [(3, "RGB"), (4, "RGB+D"), (5, "RGB+D+IR")]:
        model = create_model(num_classes=2, in_channels=ch)
        params = sum(p.numel() for p in model.parameters())
        size = get_model_size(model)
        x = torch.randn(1, ch, 224, 224)
        out = model(x)
        print(f"{name} ({ch}ch): {params/1e6:.2f}M params, {size:.1f}MB, output={out.shape}")
