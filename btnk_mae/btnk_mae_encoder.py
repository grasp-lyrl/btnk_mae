import os
import torch
import torch.nn as nn
from typing import Optional
from functools import partial
from btnk_mae.utils.patchify import patchify
from .mae_encoder import MAEEncoder
from btnk_mae.utils.pos_embed import get_2d_sincos_pos_embed, interpolate_pos_embed


COMMON_CFG = {
    "mlp_ratio": 4, "norm_pix_loss": False, "img_size": 224, "in_chans": 3,
}
PATCH_16_BASE_CFG = {
    "patch_size": 16, "embed_dim": 768, "depth": 12, "num_heads": 12, 
}
PATCH_16_LARGE_CFG = {
    "patch_size": 16, "embed_dim": 1024, "depth": 24, "num_heads": 16, 
}
PATCH_14_HUGE_CFG = {
    "patch_size": 14, "embed_dim": 1280, "depth": 32, "num_heads": 16, 
}
model_cfg_dict = {
    "base": {**COMMON_CFG, **PATCH_16_BASE_CFG},
    "large": {**COMMON_CFG, **PATCH_16_LARGE_CFG},
    "large_gan": {**COMMON_CFG, **PATCH_16_LARGE_CFG},
    "huge": {**COMMON_CFG, **PATCH_14_HUGE_CFG}
}


class BtnkMAEEncoder(nn.Module):
    """
    Bottleneck MAE Encoder that combines a pretrained MAE encoder with additional bottleneck layers.
    """
    def __init__(self, model_size: str, act_fn: Optional[str] = None):
        super().__init__()

        # Extract encoder parameters from model config
        self.model_size = model_size
        self.__dict__.update(model_cfg_dict[self.model_size])

        # Initialize the mae encoder
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.mae_encoder = MAEEncoder(
            img_size=self.img_size, patch_size=self.patch_size, in_chans=self.in_chans,
            embed_dim=self.embed_dim, depth=self.depth, num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio, norm_layer=norm_layer
        )

        # Class token & positional embedding (fixed sin-cos)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.mae_encoder.num_patches + 1, self.embed_dim),
            requires_grad=False
        )

        # Final activation
        activation_fn = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }
        self.act = activation_fn[act_fn] if act_fn is not None else None

        # ------------------------------------------------
        # Weight initialization
        self.initialize_weights()

    def set_img_size(self, img_size: int):
        """
        Set the image size for the encoder.
        """
        if img_size == self.img_size: 
            return

        assert isinstance(img_size, int), "Image size must be an integer"
        assert img_size % self.patch_size == 0, "Image size must be divisible by patch size"

        # Record the original settings
        device, dtype = self.pos_embed.device, self.pos_embed.dtype
        requires_grad = self.pos_embed.requires_grad

        # Set the image size
        self.img_size = img_size
        self.mae_encoder.patch_embed.set_input_size(img_size)
        new_num_patches = self.mae_encoder.num_patches
        
        # If the pos_embed does not require_grad, use direct sin-cos computation
        if not requires_grad:
            _pos_embed = get_2d_sincos_pos_embed(
                embed_dim=self.embed_dim,
                grid_size=int(self.mae_encoder.num_patches**0.5),
                cls_token=True
            )
            self.pos_embed = nn.Parameter(
                torch.from_numpy(_pos_embed).float().unsqueeze(0).to(device=device, dtype=dtype),
                requires_grad=False
            )

        # If the pos_embed requires_grad, use interpolation
        else:
            new_pos_embed = nn.Parameter(
                torch.zeros(1, self.mae_encoder.num_patches + 1, self.embed_dim),
                requires_grad=False
            )
            interp_pos_embed = interpolate_pos_embed(self.pos_embed, new_pos_embed, new_num_patches)
            if interp_pos_embed is not None:
                interp_pos_embed = interp_pos_embed.to(device=device, dtype=dtype)
                self.pos_embed = nn.Parameter(interp_pos_embed, requires_grad=True)

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
        os.makedirs(save_dir, exist_ok=True)
        ckpt_filename = os.path.join(save_dir, f"{model_dict[self.model_size]}")

        # If the checkpoint is not found, download it
        if not os.path.isfile(ckpt_filename):
            download_path = f"https://dl.fbaipublicfiles.com/mae/visualize/{model_dict[self.model_size]}"
            os.system(f"wget {download_path} -O {ckpt_filename}")

        # Load the weights
        self.mae_encoder.load_state_dict(torch.load(ckpt_filename)["model"], strict=False)

    def initialize_weights(self):
        """
        Initialize btnk layers and projection layers.
        The module parameter is kept for HuggingFace compatibility but not used.
        """
        # Initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim=self.pos_embed.shape[-1],
            grid_size=int(self.mae_encoder.num_patches**0.5),
            cls_token=True
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )

        # Initialize class token
        nn.init.normal_(self.cls_token, std=0.02)

        # Call the initialize weights method for the mae encoder
        self.mae_encoder.initialize_weights()

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
        Forward pass for the BtnkMAEEncoder.
        Args:
            x: input images (B, 3, H, W)
        Returns:
            latent: (B, D) only the cls token is used
        """
        # Patchify the input images
        x = self.mae_encoder.patch_embed(x)  # (B, N, D)

        # Add positional embedding
        x = x + self.pos_embed[:, 1:, :]  # (B, N, D)

        # Replace the CLS token with the learnable cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Pass through mae encoder
        x = self.mae_encoder(x)  # (B, N, D)

        # Optionally apply activation function
        return self.act(x) if self.act is not None else x

    def mse_loss(self, imgs, pred):
        """
        imgs: (B, 3, H, W)
        pred: (B, N, D)
        """
        patch_size = self.mae_encoder.patch_embed.patch_size[0]
        target = patchify(imgs, patch_size)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        return ((pred - target) ** 2).mean()
