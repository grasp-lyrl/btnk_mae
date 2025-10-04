import torch
import torch.nn as nn

from timm.models.vision_transformer import Block
from btnk_mae.utils.pos_embed import get_2d_sincos_pos_embed


class MAEDecoder(nn.Module):
    """
    Decoder portion of Masked Autoencoder ViT (MAE).
    """
    def __init__(self, img_size: int, patch_size: int, in_chans: int, 
                 embed_dim: int, decoder_embed_dim: int, decoder_depth: int, 
                 decoder_num_heads: int, mlp_ratio: float, norm_layer: callable):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans

        # ------------------------------------------------
        # Decoder projection
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Positional embedding for the decoder (fixed sin-cos)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim),
            requires_grad=False
        )

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer
            )
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)

        # Final projection to pixel values (predict each patch)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)

    @property
    def num_patches(self):
        return (self.img_size // self.patch_size) ** 2

    def initialize_weights(self):
        # --------------------------------------------------------------------------
        # Initialize decoder_pos_embed by sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(
            embed_dim=self.decoder_pos_embed.shape[-1],
            grid_size=int(self.num_patches**0.5),
            cls_token=True
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # --------------------------------------------------------------------------
        # Mask token
        nn.init.normal_(self.mask_token, std=0.02)

        # --------------------------------------------------------------------------
        # Initialize linear weights
        nn.init.xavier_uniform_(self.decoder_embed.weight)
        nn.init.constant_(self.decoder_embed.bias, 0)
        nn.init.xavier_uniform_(self.decoder_pred.weight)
        nn.init.constant_(self.decoder_pred.bias, 0)

    def forward(self, x):
        """
        Forward pass for the decoder without unshuffling or mask token insertion.
        
        Args:
            x (torch.Tensor): (B, N, embed_dim), 
                output tokens from the encoder (including CLS token) or the bottleneck layers
        Returns:
            torch.Tensor: (B, N, patch_size^2 * in_chans), 
                predicted pixel values for each patch (excluding CLS)
        """
        # Map from encoder dimension to decoder dimension
        x = self.decoder_embed(x)  # (B, 1 + N, decoder_embed_dim)
        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        x = self.decoder_pred(x)

        x = x[:, 1:, :]  # (B, N, patch_size^2 * in_chans)

        return x
