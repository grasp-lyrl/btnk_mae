import os
import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import Block
from btnk_mae.utils.pos_embed import get_2d_sincos_pos_embed
from btnk_mae.utils.patchify import unpatchify
from btnk_mae.patch_reconstruct import PatchReconstruction
from btnk_mae.mae_decoder import MAEDecoder


PATCH_16_BASE_CFG = {
    "patch_size": 16, "embed_dim": 768,
    "decoder_embed_dim": 512, "decoder_depth": 8, "decoder_num_heads": 16, 
}

PATCH_16_LARGE_CFG = {
    "patch_size": 16, "embed_dim": 1024,
    "decoder_embed_dim": 512, "decoder_depth": 8, "decoder_num_heads": 16, 
}

PATCH_14_HUGE_CFG = {
    "patch_size": 14, "embed_dim": 1280,
    "decoder_embed_dim": 512, "decoder_depth": 8, "decoder_num_heads": 16, 
}
model_cfg_dict = {
    "base": PATCH_16_BASE_CFG,
    "large": PATCH_16_LARGE_CFG,
    "large_gan": PATCH_16_LARGE_CFG,
    "huge": PATCH_14_HUGE_CFG
}


class BtnkMAEDecoder(nn.Module):
    """
    Bottleneck MAE Decoder that combines additional bottleneck layers with a pretrained MAE decoder.
    """
    
    def __init__(self, model_size: str, img_size: int = 224, in_chans: int = 3, 
                 mlp_ratio: float = 4., norm_layer: callable = partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()

        # Get model configuration and set parameters
        self.img_size = img_size
        self.in_chans = in_chans
        
        # Extract decoder parameters from model config
        self.model_size = model_size
        self.__dict__.update(model_cfg_dict[self.model_size])

        # Initialize the learnable patch embeddings and its positional embedding
        self.learned_embed = torch.randn(1, self.num_patches, self.embed_dim)
        self.learned_embed = nn.Parameter(self.learned_embed)
        
        # Positional embedding
        self.pos_embed = torch.zeros(1, self.num_patches, self.embed_dim)
        self.pos_embed = nn.Parameter(self.pos_embed, requires_grad=False)

        # Cross-Attention
        self.patch_recon = PatchReconstruction(
            encoder_dim = self.embed_dim,  # The dimension of the encoder embeddings
            num_heads = self.decoder_num_heads  # Use the same number of heads as the decoder
        )
        # self.projection = nn.Sequential(
        #     nn.LayerNorm(self.embed_dim),
        #     nn.Linear(self.embed_dim, self.decoder_embed_dim),
        #     nn.GELU(),
        #     nn.LayerNorm(self.decoder_embed_dim),
        # )

        # Decoder blocks
        self.mae_decoder = MAEDecoder(
            img_size=self.img_size, patch_size=self.patch_size, 
            in_chans=self.in_chans, embed_dim=self.embed_dim, 
            decoder_embed_dim=self.decoder_embed_dim, decoder_depth=self.decoder_depth, 
            decoder_num_heads=self.decoder_num_heads, mlp_ratio=mlp_ratio, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )

        # Weight initialization
        self.initialize_weights()

    def load_pretrained(self):
        """
        This will load the pretrained MAE weights. It will automatically handle
        the weight downloading if its not already downloaded.
        """
        model_dict = {
            "base": "mae_visualize_vit_base.pth",
            "large": "mae_visualize_vit_large.pth",
            "large_gan": "mae_visualize_vit_large_ganloss.pth",
            "huge": "mae_visualize_vit_huge.pth"
        }
        save_dir = os.path.join(os.path.dirname(__file__), "../.cache/pretrained/")
        ckpt_filename = os.path.join(save_dir, f"{model_dict[self.model_size]}")

        # We will assume that the checkpoint will be download by the encoder part
        # Therefore we don't manage the downloading here, if file not exist just raise an error
        if not os.path.isfile(ckpt_filename):
            raise FileNotFoundError(f"Pretrained MAE decoder checkpoint {ckpt_filename} not found")

        # Load the weights
        self.mae_decoder.load_state_dict(torch.load(ckpt_filename)["model"], strict=False)

    @property
    def num_patches(self):
        return (self.img_size // self.patch_size) ** 2

    def initialize_weights(self):
        """
        Initialize btnk layers and projection layers. Also copy in fixed sin-cos position embeddings.
        """
        # Initialize (and freeze) pos_embed by sin-cos embedding
        num_patches = self.num_patches
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim=self.pos_embed.shape[-1],
            grid_size=int(num_patches**0.5),
            cls_token=False
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )

        # Initialize decoder queries
        nn.init.trunc_normal_(self.learned_embed, std=0.02)

        # Call the initialize weights method for the mae decoder
        self.patch_recon.initialize_weights()
        self.mae_decoder.initialize_weights()

        # Apply normal init to all Linear/LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        Forward pass for the BtnkMAEDecoder.
        Args:
            x: (B, embed_dim)
        Returns:
            x: (B, N, embed_dim), patch predictions
        """
        # Expand the decoder queries to match the number of patches
        B = x.size(0)
        learned_embed = self.learned_embed.expand(B, -1, -1) + self.pos_embed  # (B, N, embed_dim)
        x = self.patch_recon(learned_embed, x)  # (B, N, embed_dim)

        # Map from encoder dimension to decoder dimension
        x = self.mae_decoder.decoder_embed(x)  # (B, N, decoder_embed_dim)

        # Pass through mae decoder
        for blk in self.mae_decoder.decoder_blocks:
            x = blk(x)

        x = self.mae_decoder.decoder_norm(x)
        x = self.mae_decoder.decoder_pred(x)

        # img = unpatchify(x, self.patch_size)

        return x


    def full_mae_forward(self, x):
        """
        Forward pass for the full MAE decoder.
        Args:
            x: (B, N+1, embed_dim)
        Returns:
            x: (B, N, embed_dim) patch predictions
        """
        return self.mae_decoder(x)
