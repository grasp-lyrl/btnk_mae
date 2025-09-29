from omegaconf import OmegaConf, DictConfig
from .btnk_mae_encoder import BtnkMAEEncoder
from .btnk_mae_decoder import BtnkMAEDecoder
from .utils.process_img import process_one_image


def get_encoder_decoder(cfg: DictConfig):
    if cfg.model.type == "btnk_mae":
        encoder = BtnkMAEEncoder(model_size=cfg.model.model_size, act_fn=cfg.model.act_fn)
        decoder = BtnkMAEDecoder(model_size=cfg.model.model_size)
        return encoder, decoder
    else:
        raise ValueError(f"Model {cfg.model.type} not found")

__all__ = [
    "get_encoder_decoder",
    "BtnkMAEEncoder", 
    "BtnkMAEDecoder", 
    "process_one_image"
]
