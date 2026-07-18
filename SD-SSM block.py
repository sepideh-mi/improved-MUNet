
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== Spatial + Channel Reconstruction Conv ==========
class SCConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.spatial = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)   # depthwise spatial
        self.channel = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),      # 1x1 projection
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
            nn.Dropout2d(dropout)
        )
    def forward(self, x):
        return self.channel(self.spatial(x))

# ========== Depthwise Separable Conv ==========
class DWConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)
        self.dropout = nn.Dropout2d(dropout)
    def forward(self, x):
        return self.dropout(self.pointwise(self.depthwise(x)))

# ========== SD‑Conv = SCConv + DWConv + dilated Conv ==========
class SDConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1, dropout=0.0):
        super().__init__()
        self.scconv = SCConv2D(in_ch, out_ch, dropout)
        self.dwconv = DWConv2D(out_ch, out_ch, dropout)
        self.dilated = nn.Conv2d(out_ch, out_ch, 3, padding=dilation, dilation=dilation)
        self.dropout = nn.Dropout2d(dropout)
    def forward(self, x):
        return self.dropout(self.dilated(self.dwconv(self.scconv(x))))

# ========== SD‑SSM Block (split‑merge + residual) ==========
class SDSSMBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilations=[1,2,3], dropout=0.0):
        super().__init__()
        split = in_ch // 2
        # Local branch (x1)
        self.bn1 = nn.BatchNorm2d(split)
        self.sd1 = SDConv2D(split, out_ch, dilation=dilations[0], dropout=dropout)
        self.sd2 = SDConv2D(out_ch, out_ch, dilation=dilations[1], dropout=dropout)
        self.sd3 = SDConv2D(out_ch, out_ch, dilation=dilations[2], dropout=dropout)
        # Global branch (x2)
        self.inst_norm = nn.InstanceNorm2d(split)
        self.linear = nn.Linear(split, split)
        self.silu = nn.SiLU()
        self.sd4 = SDConv2D(split, out_ch, dilation=2, dropout=dropout)
        # Fusion
        self.final_norm = nn.InstanceNorm2d(out_ch)
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1),
            nn.Dropout2d(dropout)
        )
        # Residual shortcut
        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = x
        x1, x2 = torch.chunk(x, 2, dim=1)          # split

        # Local path
        x1 = self.sd3(self.sd2(self.sd1(self.bn1(x1))))

        # Global path: instance norm → permute → linear → SiLU → permute back
        x2 = self.sd4(self.silu(self.linear(self.inst_norm(x2).permute(0,2,3,1))).permute(0,3,1,2))

        fused = self.out_conv(self.final_norm(x1 + x2))
        return fused + self.residual(residual)      # residual connection
```
