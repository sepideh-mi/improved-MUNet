import torch
import torch.nn as nn
import torch.nn.functional as F

class SCConv2D(nn.Module):
    """Spatial and Channel Reconstruction Convolution (2D)"""
    def __init__(self, in_channels, out_channels, dropout_rate=0.0, device=None):
        super(SCConv2D, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.spatial_reconstruction = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels
        ).to(self.device)
        
        self.channel_reconstruction = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1).to(self.device),
            nn.BatchNorm2d(out_channels).to(self.device),
            nn.SiLU().to(self.device),
            nn.Dropout2d(dropout_rate).to(self.device)
        )

    def forward(self, x):
        x = x.to(self.device)
        x = self.spatial_reconstruction(x)
        x = self.channel_reconstruction(x)
        return x


class DWConv2D(nn.Module):
    """Depthwise Separable Convolution (2D)"""
    def __init__(self, in_channels, out_channels, dropout_rate=0.0, device=None):
        super(DWConv2D, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels
        ).to(self.device)
        
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1).to(self.device)
        self.dropout = nn.Dropout2d(dropout_rate).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.dropout(x)
        return x


class SDConv2D(nn.Module):
    """SD-Conv (2D): Combines SCConv and DWConv with dilation"""
    def __init__(self, in_channels, out_channels, dilation=1, dropout_rate=0.0, device=None):
        super(SDConv2D, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.scconv = SCConv2D(in_channels, out_channels, dropout_rate, device=self.device)
        self.dwconv = DWConv2D(out_channels, out_channels, dropout_rate, device=self.device)
        self.dilated_conv = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation
        ).to(self.device)
        self.dropout = nn.Dropout2d(dropout_rate).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.scconv(x)
        x = self.dwconv(x)
        x = self.dilated_conv(x)
        x = self.dropout(x)
        return x


class SDSSMBlock(nn.Module):
    """SD-SSM Block (2D): Local + Global Feature Fusion with Residual"""
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 2, 3], dropout_rate=0.0, debug=False, device=None):
        super(SDSSMBlock, self).__init__()
        self.debug = debug
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Split input into two branches (x1 and x2)
        self.split_channels = in_channels // 2

        # X1 Branch (Local Features)
        self.bn1 = nn.BatchNorm2d(self.split_channels).to(self.device)
        self.sdconv1 = SDConv2D(self.split_channels, out_channels, dilation=dilation_rates[0], dropout_rate=dropout_rate, device=self.device)
        self.sdconv2 = SDConv2D(out_channels, out_channels, dilation=dilation_rates[1], dropout_rate=dropout_rate, device=self.device)
        self.sdconv3 = SDConv2D(out_channels, out_channels, dilation=dilation_rates[2], dropout_rate=dropout_rate, device=self.device)

        # X2 Branch (Global Features)
        self.instance_norm = nn.InstanceNorm2d(self.split_channels).to(self.device)
        self.linear1 = nn.Linear(self.split_channels, self.split_channels).to(self.device)
        self.silu = nn.SiLU().to(self.device)
        self.sdconv4 = SDConv2D(self.split_channels, out_channels, dilation=2, dropout_rate=dropout_rate, device=self.device)

        # Fusion
        self.final_norm = nn.InstanceNorm2d(out_channels).to(self.device)
        self.output_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1).to(self.device),
            nn.Dropout2d(dropout_rate).to(self.device)
        )

        # Residual Projection
        if in_channels != out_channels:
            self.residual_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1).to(self.device)
        else:
            self.residual_proj = nn.Identity().to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        if self.debug:
            print(f"Input Shape: {x.shape}")

        residual = x  # Save for residual connection

        # Split input into two branches
        x1, x2 = torch.chunk(x, 2, dim=1)

        # X1 Branch (Local Features)
        x1 = self.bn1(x1)
        x1 = self.sdconv1(x1)
        x1 = self.sdconv2(x1)
        x1 = self.sdconv3(x1)

        # X2 Branch (Global Features)
        x2 = self.instance_norm(x2)
        x2 = x2.permute(0, 2, 3, 1)  # (B, C, H, W) → (B, H, W, C)
        x2 = self.linear1(x2)
        x2 = self.silu(x2).permute(0, 3, 1, 2)  # (B, H, W, C) → (B, C, H, W)
        x2 = self.sdconv4(x2)

        # Fusion
        fused = x1 + x2
        fused = self.final_norm(fused)
        output = self.output_conv(fused)

        # Residual Connection
        output += self.residual_proj(residual)

        if self.debug:
            print(f"Output Shape: {output.shape}")
        return output
