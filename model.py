
import torch
import torch.nn as nn
from sd_ssm_block import SDSSMBlock

# Linear projection + BN
class LinearEmbedding(nn.Module):
    def __init__(self, in_ch, emb_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, emb_dim, 1)
        self.norm = nn.BatchNorm2d(emb_dim)
    def forward(self, x):
        return self.norm(self.proj(x))

# Downsample by 2
class PatchMerging(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.norm = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        return self.norm(self.conv(x))

# Upsample by 2
class PatchExpanding(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.norm = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        return self.norm(self.deconv(x))

# Main UNet with SDSSM blocks
class MUNet(nn.Module):
    def __init__(self, in_ch=4, out_ch=4, emb_dim=64, num_layers=4):
        super().__init__()
        self.embed = LinearEmbedding(in_ch, emb_dim)
        # Encoder: SDSSM blocks + merging
        self.enc = nn.ModuleList([SDSSMBlock(emb_dim, emb_dim) for _ in range(num_layers)])
        self.merge = nn.ModuleList([PatchMerging(emb_dim*(2**i), emb_dim*(2**(i+1))) for i in range(num_layers-1)])
        # Bottleneck
        self.bottle = SDSSMBlock(emb_dim*(2**(num_layers-1)), emb_dim*(2**(num_layers-1)))
        # Decoder: expanding, skip‑adjustment, SDSSM blocks
        self.dec = nn.ModuleList()
        self.expand = nn.ModuleList()
        self.adjust = nn.ModuleList()
        for i in range(num_layers-1, -1, -1):
            d = emb_dim * (2**i)
            self.dec.append(SDSSMBlock(d, d))
            if i < num_layers-1:
                self.expand.append(PatchExpanding(d*2, d))
                self.adjust.append(nn.Conv2d(d*2, d, 1))
        self.final = nn.Conv2d(emb_dim, out_ch, 1)

    def forward(self, x):
        x = self.embed(x)
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x)
            skips.append(x)
            if i < len(self.merge):
                x = self.merge[i](x)
        x = self.bottle(x)
        skips = skips[::-1]
        for i, blk in enumerate(self.dec):
            if i > 0:
                x = self.expand[i-1](x)
            skip = skips[i]
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            x = torch.cat([x, skip], dim=1)
            if i < len(self.adjust):
                x = self.adjust[i](x)
            x = blk(x)
        return self.final(x)
