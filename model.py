import torch
import torch.nn as nn
from sd_ssm_block import SDSSMBlock  # Ensure this file exists

class LinearEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, dropout_rate=0.0):
        super(LinearEmbedding, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # ✅ Ensure GPU compatibility
        self.linear_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1).to(self.device)
        self.norm = nn.BatchNorm2d(embed_dim).to(self.device)
        self.dropout = nn.Dropout2d(dropout_rate).to(self.device)

    def forward(self, x):
        x = x.to(self.device)  # ✅ Move input to GPU
        x = self.linear_proj(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x

class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super(PatchMerging, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1).to(self.device)
        self.norm = nn.BatchNorm2d(out_channels).to(self.device)
        self.dropout = nn.Dropout2d(dropout_rate).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.conv(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x

class PatchExpanding(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super(PatchExpanding, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2).to(self.device)
        self.norm = nn.BatchNorm2d(out_channels).to(self.device)
        self.dropout = nn.Dropout2d(dropout_rate).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.deconv(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x

class MUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, embed_dim=64, num_layers=4, dropout_rate=0.0, debug=False):
        super(MUNet, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # ✅ Store device attribute
        self.debug = debug

        # Move model to GPU
        self.to(self.device)

        # Linear Embedding
        self.linear_embedding = LinearEmbedding(in_channels, embed_dim, dropout_rate).to(self.device)

        # Encoder: SDSSM Blocks + Patch Merging
        self.encoder_blocks = nn.ModuleList()
        self.patch_merging_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.encoder_blocks.append(SDSSMBlock(embed_dim, embed_dim).to(self.device))
            if _ != num_layers - 1:
                self.patch_merging_layers.append(PatchMerging(embed_dim, embed_dim * 2, dropout_rate).to(self.device))
                embed_dim *= 2

        # Bottleneck: SDSSM Block
        self.bottleneck = SDSSMBlock(embed_dim, embed_dim).to(self.device)

        # Decoder: Patch Expanding + SDSSM Blocks + Channel Adjustment
        self.decoder_blocks = nn.ModuleList()
        self.patch_expanding_layers = nn.ModuleList()
        self.channel_adjust_layers = nn.ModuleList()

        for _ in range(num_layers - 1, -1, -1):
            if _ != num_layers - 1:
                embed_dim //= 2
                self.patch_expanding_layers.append(PatchExpanding(embed_dim * 2, embed_dim, dropout_rate).to(self.device))
            self.decoder_blocks.append(SDSSMBlock(embed_dim, embed_dim).to(self.device))
            self.channel_adjust_layers.append(nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1).to(self.device))

        # Final Conv Layer (outputs logits)
        self.final_conv = nn.Conv2d(embed_dim, out_channels, kernel_size=1).to(self.device)

    def forward(self, x):
        x = x.to(self.device)  # ✅ Ensure input is moved to GPU
        x = self.linear_embedding(x)

        # Encoder
        skip_connections = []
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            skip_connections.append(x)
            if i < len(self.patch_merging_layers):
                x = self.patch_merging_layers[i](x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1]
        for i, block in enumerate(self.decoder_blocks):
            if i > 0:
                x = self.patch_expanding_layers[i - 1](x)
            skip_connection = skip_connections[i]
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:])
            x = torch.cat((x, skip_connection), dim=1)
            x = self.channel_adjust_layers[i](x)
            x = block(x)

        # Final Conv Layer (returns logits)
        x = self.final_conv(x)  # ✅ Softmax removed, outputs raw logits
        return x

#
