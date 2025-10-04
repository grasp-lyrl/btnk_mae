# train.py
import sys
import os
import time
import datetime
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.distributed.elastic.multiprocessing.errors import record

import wandb
import hydra
from omegaconf import OmegaConf, DictConfig

import btnk_mae
import btnk_mae.utils.misc as misc
from btnk_mae.utils.datasets import get_dataset
from btnk_mae.utils.misc import NativeScalerWithGradNormCount as NativeScaler
from train_engine import train_one_epoch


@record
@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    """
    Main training loop for CLS-MAE.

    Args:
        cfg: Hydra config object, contains model/training hyperparameters
    """
    # Allow new key
    OmegaConf.set_struct(cfg, False)

    # If no run_name is set, use datatime str as the run name
    ds = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.run_name = cfg.get("run_name") or ds
    cfg.output_dir = f"ckpts/{cfg.run_name}"
    dcfg = cfg.distributed

    # -------------------------------------------------------------------------
    # Initialize distributed mode if needed
    # -------------------------------------------------------------------------
    misc.init_distributed_mode(cfg)
    if cfg.distributed is not None:
        is_valid_rank = dcfg.local_rank >= 0
        dcfg.local_rank = dcfg.local_rank if is_valid_rank else misc.get_rank()

    # -------------------------------------------------------------------------
    # Initialize W&B for the main process and create output_dir
    # -------------------------------------------------------------------------
    assert cfg.run_name is not None, "Run name not set properly, aborted."
    if misc.is_main_process():
        print(f"Initializing W&B with project {cfg.wandb_project} and run name {cfg.run_name}")
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.run_name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
        os.makedirs(cfg.output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Fix seed for reproducibility
    # -------------------------------------------------------------------------
    seed = cfg.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # -------------------------------------------------------------------------
    # Create dataset & DataLoader
    # -------------------------------------------------------------------------
    train_dataset = get_dataset(cfg)

    if cfg.distributed is not None:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)

    data_loader_train = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler_train,
        batch_size=cfg.train_cfg.batch_size,
        num_workers=cfg.train_cfg.num_workers,
        pin_memory=cfg.train_cfg.pin_mem,
        drop_last=True
    )

    # -------------------------------------------------------------------------
    # Load model and optimizer
    # -------------------------------------------------------------------------
    encoder, decoder = btnk_mae.get_encoder_decoder(cfg)
    if cfg.distributed is not None:
        torch.cuda.set_device(dcfg.local_rank)
        device = torch.device(f"cuda:{dcfg.local_rank}")
        rank = misc.get_rank()
        device_id = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_id)
        print(f"[Rank {rank}] using device {device_id} ({device_name})", flush=True)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder.to(device)
    decoder.to(device)

    # -------------------------------------------------------------------------
    # Create optimizer for the bottleneck only
    # -------------------------------------------------------------------------
    # If lr is null in YAML, compute it from blr and batch size
    if cfg.train_cfg.lr is None:
        eff_batch_size = cfg.train_cfg.batch_size * cfg.train_cfg.accum_iter * misc.get_world_size()
        cfg.train_cfg.lr = cfg.train_cfg.blr * eff_batch_size / 256

    tune_params = list(encoder.parameters())
    train_params = list(decoder.parameters())
    param_groups = [
        {"params": tune_params, "lr": cfg.train_cfg.lr * 0.1, "weight_decay": cfg.train_cfg.weight_decay},
        {"params": train_params,  "lr": cfg.train_cfg.lr,  "weight_decay": cfg.train_cfg.weight_decay},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=cfg.train_cfg.blr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    # Optionally resume from a checkpoint that includes the bottleneck
    misc.load_checkpoint(
        cfg=cfg,
        enc_without_ddp=encoder,
        dec_without_ddp=decoder,
        optimizer=optimizer,
        loss_scaler=loss_scaler
    )

    # After (optional) resuming from a checkpoint, set the image size if it is different from the original image size
    if cfg.model.img_size != encoder.img_size or cfg.model.img_size != decoder.img_size:
        encoder.set_img_size(cfg.model.img_size)
        decoder.set_img_size(cfg.model.img_size)

    if cfg.distributed is not None:
        encoder = torch.nn.parallel.DistributedDataParallel(
            encoder, device_ids=[dcfg.local_rank], find_unused_parameters=False
        )
        decoder = torch.nn.parallel.DistributedDataParallel(
            decoder, device_ids=[dcfg.local_rank], find_unused_parameters=False
        )

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    for epoch in range(cfg.train_cfg.start_epoch, cfg.train_cfg.epochs):
        if cfg.distributed is not None:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(
            encoder=encoder,
            decoder=decoder,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            args=cfg
        )

        if cfg.distributed is not None:
            torch.distributed.barrier()

        if cfg.output_dir and misc.is_main_process():
            enc_to_save = encoder.module if hasattr(encoder, "module") else encoder
            dec_to_save = decoder.module if hasattr(decoder, "module") else decoder
            misc.save_enc_dec_model(
                cfg=cfg,
                epoch=epoch,
                enc_without_ddp=enc_to_save,
                dec_without_ddp=dec_to_save,
                optimizer=optimizer,
                loss_scaler=loss_scaler
            )

        if cfg.distributed is not None:
            torch.distributed.barrier()

    if misc.is_main_process():
        wandb.finish()


if __name__ == "__main__":
    main()
