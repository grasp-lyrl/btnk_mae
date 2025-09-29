import torch.nn as nn


class PatchReconstruction(nn.Module):
    """
    Cross-attention layer that reconstruct the image representation from 
    (B, encoder_dim) to (B, N, encoder_dim).

    The rec_patches are learned embeddings for efficient image reconstruction. 
    These learned embeddings are cross attend with the img_token
    """
    def __init__(self, encoder_dim: int, num_heads: int, dropout: float = 0.1, mlp_ratio: float = 4.):
        """
        Initialize the PatchReconstruction module.

        Args:
            encoder_dim (int): Dimension of the encoder embeddings
            num_heads (int): Number of attention heads for multi-head attention
            dropout (float): Dropout rate for the attention and MLP
            mlp_ratio (int): Ratio of MLP hidden dimension to embedding dimension
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=encoder_dim, num_heads=num_heads, batch_first=True
        )
        self.q_norm = nn.LayerNorm(encoder_dim)   # norm for queries (rec_patches)
        self.kv_norm = nn.LayerNorm(encoder_dim)  # norm for key/value (img_token)

        mlp_hidden_dim = int(encoder_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(encoder_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, encoder_dim),
        )
        self.x_norm = nn.LayerNorm(encoder_dim)   # norm before MLP

        self.drop_attn = nn.Dropout(dropout)
        self.drop_res = nn.Dropout(dropout)

    def initialize_weights(self):
        # Initialize MultiheadAttention's in_proj weights/bias
        nn.init.xavier_uniform_(self.attn.in_proj_weight)
        if self.attn.in_proj_bias is not None:
            nn.init.constant_(self.attn.in_proj_bias, 0)

    def forward(self, rec_patches, img_token):
        """
        Forward pass of the cross-attention layer.

        Cross-attend patch embeddings (queries) and the image token (keys/values), 
        then project dimension up with MLP, with residual connections and layer normalization.

        Args:
            rec_patches (torch.Tensor): Learnable patch embeddings that serve as queries.
                                      Shape: [B, N, encoder_dim]
            img_token (torch.Tensor): Compact image representation from encoder (cls token).
                                   Shape: [B, encoder_dim]

        Returns:
            torch.Tensor: Cross-attended and MLP-processed patch embeddings with residual 
                         connections and layer normalization applied. Shape: [B, N, encoder_dim]
        """
        # Cross attention block
        q = self.q_norm(rec_patches)  # (B, N, D)
        kv = self.kv_norm(img_token).unsqueeze(1)  # (B, 1, D)
        attn_out, _ = self.attn(q, kv, kv)  # (B, N, D)
        x = rec_patches + self.drop_attn(attn_out)  # residual

        # MLP block (pre-norm)
        x = x + self.drop_res(self.mlp(self.x_norm(x)))  # residual
        return x
