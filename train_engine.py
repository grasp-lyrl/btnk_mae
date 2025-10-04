# train_engine.py
import math
import sys
from typing import Iterable

import torch
import wandb
import btnk_mae.utils.misc as misc
import btnk_mae.utils.lr_sched as lr_sched
from btnk_mae.utils.patchify import unpatchify

def train_one_epoch(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    args=None
):
    """
    Train only the bottleneck, while encoder and decoder are frozen.
    """
    # encoder & decoder in eval mode, bottleneck in train mode
    enc = encoder.module if isinstance(encoder, torch.nn.parallel.DistributedDataParallel) else encoder
    dec = decoder.module if isinstance(decoder, torch.nn.parallel.DistributedDataParallel) else decoder
    enc.train()
    dec.train()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('train_lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 20

    accum_iter = args.train_cfg.accum_iter
    optimizer.zero_grad()

    for data_iter_step, (samples, _) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # Adjust LR per iteration
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer,
                data_iter_step / len(data_loader) + epoch,
                args
            )

        samples = samples.to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            # Forward pass through encoder and decoder
            cls_latent = enc(samples)
            pred_latent = dec(cls_latent[:, 0, :])
            btnk_mae_loss = enc.mse_loss(samples, pred_latent) * args.train_cfg.btnk_mae_loss_coeff
            
            # Compute full MAE loss if mae_loss_coeff > 0
            full_mae_loss = 0
            if args.train_cfg.full_mae_loss_coeff > 0:
                full_mae_latent = dec.full_mae_forward(cls_latent)
                full_mae_loss = enc.mse_loss(samples, full_mae_latent) * args.train_cfg.full_mae_loss_coeff

            # Combine losses
            loss = btnk_mae_loss + full_mae_loss

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            if misc.is_main_process():

                wandb.log({"error": f"Loss is {loss_value}, stopping training."})
            sys.exit(1)

        # Gradient scaling/accumulation
        loss /= accum_iter
        loss_scaler(
                loss, optimizer, parameters=list(enc.parameters()) + list(dec.parameters()),
            update_grad=((data_iter_step + 1) % accum_iter == 0)
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        train_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(train_lr=train_lr, mse_loss=loss)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if misc.is_main_process() and ((data_iter_step + 1) % accum_iter == 0):
            updates_per_epoch = len(data_loader) // accum_iter
            global_step = epoch * updates_per_epoch + (data_iter_step + 1) // accum_iter
            wandb.log({
                'train/loss': loss_value_reduce,
                'train/train_lr': train_lr,
                # 'train/l1_loss': l1_loss,
                'train/mse_loss': loss,
            }, step=global_step)

            if global_step % 100 == 0:
                with torch.no_grad():
                    # pick one sample from this local batch (arbitrary)
                    idx = (data_iter_step * samples.shape[0]) % samples.shape[0]
                    orig_img = samples[idx].detach().cpu()
                    pred_img = unpatchify(pred_latent[idx].unsqueeze(0), dec.patch_size)
                    pred_img = pred_img.squeeze(0).detach().cpu()

                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    orig_img = torch.clamp(orig_img * std + mean, 0, 1)
                    pred_img = torch.clamp(pred_img * std + mean, 0, 1)

                    wandb.log({
                        "visualization/original": wandb.Image(orig_img.permute(1, 2, 0).numpy()),
                        "visualization/reconstructed": wandb.Image(pred_img.permute(1, 2, 0).numpy())
                    }, step=global_step)

    # gather stats
    metric_logger.synchronize_between_processes()
    if misc.is_main_process():
        wandb.log({"epoch_stats": str(metric_logger)})
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
