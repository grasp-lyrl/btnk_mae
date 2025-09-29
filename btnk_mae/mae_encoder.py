import torch
import torch.nn as nn

from functools import partial
from timm.models.vision_transformer import PatchEmbed, Block


class MAEEncoder(nn.Module):
    """
    Encoder of Masked Autoencoder ViT (MAE).
    """
    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int, 
                 depth: int, num_heads: int, mlp_ratio: float, norm_layer: callable):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        # Encoder blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer
            )
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Weight initialization
        self.initialize_weights()

    @property
    def num_patches(self):
        return self.patch_embed.num_patches
    
    def initialize_weights(self, module=None):
        """
        Initialize learnable parameters
        """
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def forward(self, x):
        """
        Forward pass for the encoder.
        Args:
            x: (B, num_patches, embed_dim)
        Returns:
            x: (B, num_patches, embed_dim)
        """
        # Pass through Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x
