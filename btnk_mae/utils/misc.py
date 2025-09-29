# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.distributed as dist
from torch import inf
import wandb


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    log_message = log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB)
                    
                    if is_main_process() and wandb.run is not None:
                        # Log iteration metrics to wandb
                        wandb.log({
                            "iter/progress": i / len(iterable),
                            "iter/eta_seconds": eta_seconds,
                            "iter/time": iter_time.global_avg,
                            "iter/data_time": data_time.global_avg,
                            "iter/memory_MB": torch.cuda.max_memory_allocated() / MB,
                            **{f"iter/{k}": meter.global_avg for k, meter in self.meters.items()}
                        })
                else:
                    log_message = log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time))
                
                if is_main_process():
                    print(log_message)
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        if is_main_process() and wandb.run is not None:
            print('{} Total time: {} ({:.4f} s / it)'.format(
                header, total_time_str, total_time / len(iterable)))
            wandb.log({
                "iter/total_time": total_time,
                "iter/time_per_it": total_time / len(iterable)
            })
        return total_time


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def init_distributed_mode(cfg):
    dcfg = cfg.distributed
    if dcfg.dist_on_itp:
        dcfg.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        dcfg.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        dcfg.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        dcfg.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(dcfg.gpu)
        os.environ['RANK'] = str(dcfg.rank)
        os.environ['WORLD_SIZE'] = str(dcfg.world_size)
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dcfg.rank = int(os.environ["RANK"])
        dcfg.world_size = int(os.environ['WORLD_SIZE'])
        dcfg.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        dcfg.rank = int(os.environ['SLURM_PROCID'])
        dcfg.gpu = dcfg.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)
        cfg.distributed = None
        return

    # ─── DEBUG: bind temporarily to LOCAL_RANK so we can query memory stats
    temp_device = dcfg.gpu
    torch.cuda.set_device(temp_device)
    alloc   = torch.cuda.memory_allocated(temp_device)
    reserved = torch.cuda.memory_reserved(temp_device)
    print(f"[Rank {dcfg.rank}] BEFORE real bind → device={temp_device}, "
          f"alloc={alloc/1024**2:.1f} MiB, reserved={reserved/1024**2:.1f} MiB",
          flush=True)

    # ─── Now do the real binding for NCCL:
    torch.cuda.set_device(dcfg.gpu)
    dcfg.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        dcfg.rank, dcfg.dist_url, dcfg.gpu), flush=True)

    torch.distributed.init_process_group(
        backend=dcfg.dist_backend,
        init_method=dcfg.dist_url,
        world_size=dcfg.world_size,
        rank=dcfg.rank,
        timeout=datetime.timedelta(seconds=600)
    )
    torch.distributed.barrier()
    setup_for_distributed(dcfg.rank == 0)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.amp.GradScaler('cuda')

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def save_enc_dec_model(cfg, epoch, enc_without_ddp, dec_without_ddp, optimizer, loss_scaler):
    # Only save on main process to avoid DDP race conditions
    if not is_main_process():
        return
        
    output_dir = Path(cfg.output_dir)
    epoch_name = str(epoch)

    print("*" * 100)
    print(f"[RANK {get_rank()}] Saving encoder and decoder models at epoch {epoch}")
    print("*" * 100)

    # Define to_save dictionary once, outside the conditional block
    to_save = {
        'enc_model': enc_without_ddp.state_dict(),
        'dec_model': dec_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'scaler': loss_scaler.state_dict(),
        'cfg': cfg,
    }

    checkpoint_paths = []
    if epoch % 10 == 0:
        checkpoint_path = output_dir / ('checkpoint-%s.pth' % epoch_name)
        torch.save(to_save, checkpoint_path)
        checkpoint_paths.append(checkpoint_path)

    # Save latest checkpoint separately
    checkpoint_path = output_dir / 'checkpoint-latest.pth'
    torch.save(to_save, checkpoint_path)
    checkpoint_paths.append(checkpoint_path)

    print(f"[RANK {get_rank()}] Finished saving encoder and decoder models", flush=True)


def load_checkpoint(cfg, enc_without_ddp, dec_without_ddp, optimizer, loss_scaler):
    # If want to resume from a checkpoint
    if cfg.train_cfg.resume:
        print("="*10 + f" Resuming checkpoint {cfg.train_cfg.resume} " + "="*10)
        checkpoint = torch.load(cfg.train_cfg.resume, map_location='cpu', weights_only=False)

        enc_without_ddp.load_state_dict(checkpoint['enc_model'], strict=False)
        dec_without_ddp.load_state_dict(checkpoint['dec_model'], strict=False)
        
        resume_optimizer = cfg.train_cfg.resume_optimizer if hasattr(cfg.train_cfg, 'resume_optimizer') else True
        if 'optimizer' in checkpoint \
            and 'epoch' in checkpoint \
            and not (hasattr(cfg.train_cfg, 'eval') and cfg.train_cfg.eval) \
            and resume_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
            cfg.train_cfg.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    # If want to start from scratch, we will still load the pretrained MAE encoder and decoder
    else:
        enc_without_ddp.load_pretrained()
        dec_without_ddp.load_pretrained()
        cfg.train_cfg.start_epoch = 0


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x
