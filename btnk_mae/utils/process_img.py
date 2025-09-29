import torch
import numpy as np
from PIL import Image
from btnk_mae.utils.patchify import unpatchify

# ImageNet normalization constants
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def load_image(path, device="cpu"):
    """Load and normalize an image."""
    img = Image.open(path).convert("RGB")
    img_array = np.array(img)[..., :3] / 255.0  # shape: (H, W, 3), float64
    img_tensor = torch.tensor(img_array, dtype=torch.float32)
    img_tensor = (img_tensor - IMAGENET_MEAN) / IMAGENET_STD
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 3, H, W)
    return img, img_tensor


def decode_image(patch_size, img_feat, device="cpu"):
    pred_img = unpatchify(img_feat, patch_size)
    pred_img = torch.einsum('nchw->nhwc', pred_img)  # (N, H, W, C)

    # Make sure stats are broadcastable to (H, W, 3)
    imagenet_mean = IMAGENET_MEAN.view(1, 1, 3).to(device)
    imagenet_std = IMAGENET_STD.view(1, 1, 3).to(device)

    pred_img = pred_img * imagenet_std + imagenet_mean
    pred_img = torch.clamp(pred_img * 255.0, 0, 255).to(torch.uint8)
    return pred_img.to(device)


def process_one_image(encoder, decoder, img_path, device="cuda"):
    # Load image
    img, img_tensor = load_image(img_path, device=device)

    # Model forward
    feat = encoder(img_tensor)
    y = decoder(feat)

    # Postprocess
    rec_img = decode_image(decoder.patch_size, y)

    return img, rec_img
