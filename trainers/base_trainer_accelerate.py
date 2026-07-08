import argparse
import datetime
import fnmatch
import itertools
import os
import random
import traceback
from accelerate import Accelerator
import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
import yaml
from tqdm import tqdm
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict
from copy import deepcopy
from easydict import EasyDict
import time
import json
import math
import sys
from PIL import Image
import shutil
from safetensors.torch import load_file as load_safetensors
from utils.basic import seed_anything, count_parameters

from datasets import create_dataloader
# from model.network import Network
from utils.misc import get_logger, is_logging_process, pretty_print_hydra_config, move_to_device, get_rank
from utils.basic import seed_anything
from utils.optimizer import build_optimizer
from utils.scheduler import build_scheduler
from utils.dist import (
    MetricLogger,
    SmoothedValue,
    init_distributed_mode,
    setup_for_distributed,
)
from accelerate import DistributedDataParallelKwargs
from transformers.trainer_pt_utils import get_model_param_count
from accelerate import (
    DistributedType,
)
from accelerate.utils import (
    DataLoaderConfiguration,
    DynamoBackend,
    GradientAccumulationPlugin,
    ProjectConfiguration,
    TorchDynamoPlugin,
    set_seed,
)
import numpy as np


class _LimitedIterable:
    def __init__(self, iterable, max_items):
        self.iterable = iterable
        self.max_items = max(0, int(max_items))

    def __iter__(self):
        return itertools.islice(iter(self.iterable), self.max_items)

    def __len__(self):
        return self.max_items


class BaseTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self._no_grad_loss_warn_count = 0

        with open_dict(cfg):
            cfg.job_logging_cfg = HydraConfig.get().job_logging

        # random seed
        if cfg.random_seed is None:
            cfg.random_seed = random.randint(1, 10000)
        seed_anything(cfg.random_seed, deterministic=False)              # deterministic=True for reproduction

        ## 1. Build accelerator
        self.build_accelerator()

        if is_logging_process():
            pretty_print_hydra_config(cfg)

        ## 2. Prepare model
        self.log_info("Preparing model...")
        self.model = self.prepare_model()
        self.n_learnable_parameters = get_model_param_count(
            self.model, trainable_only=True
        )
        self.n_fix_parameters = get_model_param_count(
            self.model, trainable_only=False
        )
        self.accelerator.wait_for_everyone()

        ## 3. Prepare dataloader
        self.log_info("Making train dataloader...")
        self.train_loader = create_dataloader(cfg, 'train')
        self.log_info("Making test dataloader...")
        self.test_loader = create_dataloader(cfg, 'test')
        self.accelerator.wait_for_everyone()

        ## 5. Prepare optimizer and scheduler (fsdp should after preparing the model using accelerate)
        if self._using_fsdp():
            self.model = self._maybe_wrap_model_for_hsdp(self.model)
            self.model = self.accelerator.prepare(self.model)
            self.accelerator.wait_for_everyone()

            self.optimizer = self.build_optimizer(self.cfg.train.optimizer, self.model)
            self.log_info(f"optimizer: {self.optimizer}")
        else:
            self.optimizer = self.build_optimizer(self.cfg.train.optimizer, self.model)
            self.log_info(f"optimizer: {self.optimizer}")

            self.model = self.accelerator.prepare(self.model)
            self.accelerator.wait_for_everyone()

        # Create the LR scheduler
        self.iters_per_epoch = self.cfg.train.iters_per_epoch if self.cfg.train.iters_per_epoch > 0 else len(self.train_loader)
        self.iters_per_test = self.cfg.test.iters_per_test if self.cfg.test.iters_per_test > 0 else len(self.test_loader)
        # self.cfg.train.lr_scheduler.total_steps = self.cfg.train.num_epoch * self.iters_per_epoch
        # 【修改点】：只有当 yaml 中存在 total_steps 时，才动态计算并赋值，兼容 OneCycleLR
        if 'total_steps' in self.cfg.train.lr_scheduler:
            # 临时解除 OmegaConf 的结构锁定
            with open_dict(self.cfg.train.lr_scheduler):
                self.cfg.train.lr_scheduler.total_steps = self.cfg.train.num_epoch * self.iters_per_epoch
            self.log_info(f"Total step for lr scheduler: {self.cfg.train.lr_scheduler.total_steps} ({self.cfg.train.num_epoch} * {self.iters_per_epoch})")
        else:
            self.log_info(f"Using scheduler {self.cfg.train.lr_scheduler.type} without dynamic total_steps.")
        # self.log_info(f"Total step for lr scheduler: {self.cfg.train.lr_scheduler.total_steps} ({self.cfg.train.num_epoch} * {self.iters_per_epoch})")
        self.lr_scheduler = build_scheduler(
            self.cfg.train.lr_scheduler, optimizer=self.optimizer
        )
        self.log_info(f"LRScheduler: {self.lr_scheduler}")

        ## 6. Prepare accelerate training
        self.prepare_training()

    def build_optimizer(self, cfg_optimizer, model, param_group_fn=None):
        return build_optimizer(cfg_optimizer, model, param_group_fn=param_group_fn)

    def _zero_grad_anchor_from_trainable_params(self):
        anchor = None
        for param in self.model.parameters():
            if param.requires_grad and param.numel() > 0:
                term = param.reshape(-1)[0] * 0.0
                anchor = term if anchor is None else anchor + term
        return anchor

    def _format_no_grad_loss_context(self, batch_output, max_items=12):
        parts = []
        for key, value in batch_output.items():
            if not isinstance(value, torch.Tensor) or value.numel() != 1:
                continue
            try:
                scalar = float(value.detach().float().item())
            except Exception:
                scalar = float("nan")
            parts.append(f"{key}: value={scalar:.6g}, requires_grad={value.requires_grad}")
            if len(parts) >= max_items:
                break
        return "; ".join(parts)

    def prepare_training(self):
        # report model details
        self.log_info(
            f"total number of learnable params: {self.n_learnable_parameters / 1e6} M"
        )
        self.log_info(
            f"total number of fixed params: {self.n_fix_parameters / 1e6} M"
        )

        # Wrap the model, optmizer, and scheduler with accelerate
        self.log_info("before accelerator.prepare")

        # (
        #     self.model,
        #     self.train_loader,
        #     self.test_loader,
        #     self.optimizer,
        #     self.lr_scheduler,
        # ) = self.accelerator.prepare(
        #     self.model, self.train_loader, self.test_loader, self.optimizer, self.lr_scheduler
        # )

        # don't wrap dataloader
        (
            self.optimizer,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.optimizer, self.lr_scheduler
        )

        if self.accelerator.is_main_process and len(self.accelerator.trackers) == 0:
            self.accelerator.init_trackers(os.path.basename(self.cfg.log.output_dir))

        # Report the training info
        self.total_batch_size = (
            self.cfg.train.batch_size
            * self.accelerator.num_processes
            * self.cfg.train.gradient_accumulation_steps
        )
        self.log_info("***** Running training *****")
        self.log_info(f"LR = {self.cfg.train.optimizer.lr:.8f}")
        self.log_info(f"Weigth Decay = {self.cfg.train.optimizer.weight_decay:.8f}")
        self.log_info(f"Instantaneous batch size per device = {self.cfg.train.batch_size}")
        self.log_info(f"Total Batch size = {self.total_batch_size}")
        self.log_info(
            f"Gradient Accumulation steps = {self.accelerator.gradient_accumulation_steps}"
        )
        self.log_info(f"Number of epochs = {self.cfg.train.num_epoch}")
        self.log_info(
            f"Number of training steps per epoch = {self.iters_per_epoch}"
        )
        self.log_info(
            f"Number of total training steps = {self.iters_per_epoch * self.cfg.train.num_epoch}"
        )
        # self.log_info(f"Number of training examples per epoch = {len(self.dataloader.dataset)}")
        self.log_info(
            f"Number of model parameters = {self.n_fix_parameters / 1e6:.2f}M"
        )
        self.log_info(
            f"Number of model trainable parameters = {self.n_learnable_parameters / 1e6:.2f}M"
        )

        # Auto resume the checkpoint
        latest_epoch = self.auto_resume()
        # self.initial_global_step = self.iters_per_epoch * latest_epoch
        self.initial_global_step = (self.iters_per_epoch * latest_epoch) // self.cfg.train.gradient_accumulation_steps
        self.first_epoch = latest_epoch

        os.makedirs(self.cfg.log.ckpt_dir, exist_ok=True)

    def _fsdp_requested(self):
        return bool(self.cfg.get("fsdp_plugin")) or (
            os.environ.get("ACCELERATE_USE_FSDP", "false").lower() == "true"
        )

    def _using_fsdp(self):
        return self._fsdp_requested() or (
            getattr(self.accelerator, "distributed_type", None) == DistributedType.FSDP
        )

    def _get_hsdp_cfg(self):
        hsdp_cfg = self.cfg.get("hsdp")
        if hsdp_cfg and hsdp_cfg.get("enabled", False):
            return hsdp_cfg
        return None

    def _build_hsdp_process_groups(self, shard_group_size):
        if hasattr(self, "_hsdp_process_group_pair"):
            return self._hsdp_process_group_pair

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("HSDP requires torch.distributed to be initialized before wrapping the model.")

        world_size = dist.get_world_size()
        rank = dist.get_rank()
        if shard_group_size <= 1:
            raise ValueError(f"hsdp.shard_group_size must be > 1, got {shard_group_size}.")
        if world_size % shard_group_size != 0:
            raise ValueError(
                f"world_size ({world_size}) must be divisible by hsdp.shard_group_size ({shard_group_size})."
            )

        num_shard_groups = world_size // shard_group_size
        shard_rank_groups = [
            list(range(group_idx * shard_group_size, (group_idx + 1) * shard_group_size))
            for group_idx in range(num_shard_groups)
        ]
        replicate_rank_groups = [
            list(range(offset, world_size, shard_group_size))
            for offset in range(shard_group_size)
        ]

        shard_pg = None
        shard_ranks = None
        for ranks in shard_rank_groups:
            pg = dist.new_group(ranks=ranks)
            if rank in ranks:
                shard_pg = pg
                shard_ranks = ranks

        replicate_pg = None
        replicate_ranks = None
        for ranks in replicate_rank_groups:
            pg = dist.new_group(ranks=ranks)
            if rank in ranks:
                replicate_pg = pg
                replicate_ranks = ranks

        if shard_pg is None or replicate_pg is None:
            raise RuntimeError(
                f"Rank {rank} could not be assigned to HSDP groups. "
                f"Shard groups: {shard_rank_groups}, replicate groups: {replicate_rank_groups}"
            )

        self.log_info(
            f"HSDP rank {rank}: shard_group={shard_ranks}, replicate_group={replicate_ranks}"
        )
        self._hsdp_process_group_pair = (shard_pg, replicate_pg)
        return self._hsdp_process_group_pair

    def _maybe_wrap_model_for_hsdp(self, model):
        hsdp_cfg = self._get_hsdp_cfg()
        if hsdp_cfg is None:
            return model

        from torch.distributed.fsdp import (
            CPUOffload,
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            ShardingStrategy,
        )

        if isinstance(model, FSDP):
            return model

        shard_group_size = int(hsdp_cfg.get("shard_group_size", 2))
        process_group = self._build_hsdp_process_groups(shard_group_size)
        sharding_strategy_name = str(hsdp_cfg.get("sharding_strategy", "HYBRID_SHARD")).upper()
        if not hasattr(ShardingStrategy, sharding_strategy_name):
            raise ValueError(f"Unsupported hsdp.sharding_strategy: {sharding_strategy_name}")

        cpu_offload = None
        if hsdp_cfg.get("cpu_offload", False):
            cpu_offload = CPUOffload(offload_params=True)

        mixed_precision = MixedPrecision(
            param_dtype=self.weight_dtype,
            reduce_dtype=self.weight_dtype,
            buffer_dtype=self.weight_dtype,
            cast_forward_inputs=True,
            cast_root_forward_inputs=True,
        )

        self.log_info(
            f"Wrapping model with HSDP: strategy={sharding_strategy_name}, shard_group_size={shard_group_size}"
        )
        return FSDP(
            model,
            process_group=process_group,
            sharding_strategy=getattr(ShardingStrategy, sharding_strategy_name),
            cpu_offload=cpu_offload,
            auto_wrap_policy=None,
            mixed_precision=mixed_precision,
            device_id=self.accelerator.device,
            sync_module_states=bool(hsdp_cfg.get("sync_module_states", True)),
            forward_prefetch=bool(hsdp_cfg.get("forward_prefetch", False)),
            limit_all_gathers=bool(hsdp_cfg.get("limit_all_gathers", True)),
            use_orig_params=bool(hsdp_cfg.get("use_orig_params", True)),
        )

    def prepare_model(self):
        model = hydra.utils.instantiate(self.cfg.model)
        count_parameters(model)
        # model.encoder = torch.compile(model.encoder)
        # model.decoder = torch.compile(model.decoder)
        return model

    # def prepare_model(self):
    #     model = hydra.utils.instantiate(self.cfg.model)
    #     count_parameters(model)
        
    #     # 使用 try-except 包装，避免低版本 PyTorch 报错
    #     try:
    #         import torch._dynamo
    #         torch._dynamo.config.suppress_errors = True 
            
    #         self.log_info("Applying torch.compile to pure PyTorch sub-modules...")
            
    #         # 推荐使用 "default" 模式。
    #         # "max-autotune" 编译时间极长，且在显存边缘游走时容易触发 OOM。
    #         compile_mode = "default" 
            
    #         # 1. 编译特征提取器
    #         if hasattr(model, 'encoder'):
    #             model.encoder = torch.compile(model.encoder, mode=compile_mode, dynamic=True)
    #         # if hasattr(model, 'decoder'):
    #         #     model.decoder = torch.compile(model.decoder, mode=compile_mode, dynamic=True)
    #         # 2. 编译并行的 Decoder Heads
    #         # 将纯 Transformer 结构的头部进行编译，可以大幅加速 Attention 和 MLP 计算
    #         if hasattr(model, 'point_decoder'):
    #             model.point_decoder = torch.compile(model.point_decoder, mode=compile_mode, dynamic=True)
    #         if hasattr(model, 'gs_decoder'):
    #             model.gs_decoder = torch.compile(model.gs_decoder, mode=compile_mode, dynamic=True)
    #         if hasattr(model, 'camera_decoder'):
    #             model.camera_decoder = torch.compile(model.camera_decoder, mode=compile_mode, dynamic=True)
    #         if hasattr(model, 'conf_decoder'):
    #             model.conf_decoder = torch.compile(model.conf_decoder, mode=compile_mode, dynamic=True)
    #         if hasattr(model, 'gs_head'):
    #             model.gs_head = torch.compile(model.gs_head, mode=compile_mode, dynamic=True)
    #         if hasattr(model, 'point_head'):
    #             model.point_head = torch.compile(model.point_head, mode=compile_mode, dynamic=True)
    #         if hasattr(model, 'camera_head'):
    #             model.camera_head = torch.compile(model.camera_head, mode=compile_mode, dynamic=True)
    #         if hasattr(model, 'conf_head'):
    #             model.conf_head = torch.compile(model.conf_head, mode=compile_mode, dynamic=True)
    #         # (注：刻意跳过了光栅化头和共享的 model.decoder(ModuleList)，
    #         # 因为 ModuleList 逐块编译容易产生碎片化的 Graph，得不偿失)
            
    #     except Exception as e:
    #         self.log_info(f"torch.compile failed or skipped. Error: {e}")

    #     return model
    # def prepare_model(self):
    #     model = hydra.utils.instantiate(self.cfg.model)
    #     count_parameters(model)
        
    #     try:
    #         import torch._dynamo
    #         torch._dynamo.config.suppress_errors = True 
            
    #         self.log_info("Applying torch.compile to decoder modules...")
            
    #         # 【修改点】：取消 max-autotune，改用 default，防止极端融合导致 NaN
    #         compile_mode = "default" 
            
    #         if hasattr(model, 'decoder') and isinstance(model.decoder, torch.nn.ModuleList):
    #             for i in range(len(model.decoder)):
    #                 model.decoder[i] = torch.compile(model.decoder[i], mode=compile_mode)
            
    #         if hasattr(model, 'point_decoder'):
    #             model.point_decoder = torch.compile(model.point_decoder, mode=compile_mode)
    #         if hasattr(model, 'gs_decoder'):
    #             model.gs_decoder = torch.compile(model.gs_decoder, mode=compile_mode)
    #         if hasattr(model, 'camera_decoder'):
    #             model.camera_decoder = torch.compile(model.camera_decoder, mode=compile_mode)
    #         if hasattr(model, 'conf_decoder'):
    #             model.conf_decoder = torch.compile(model.conf_decoder, mode=compile_mode)
                
    #     except Exception as e:
    #         self.log_info(f"torch.compile failed or skipped. Error: {e}")

    #     return model
    
    def before_epoch(self, epoch):
        pass

    def train(self):
        # Start Train!
        start_time = time.time()
        self.accelerator.wait_for_everyone()

        # Initialize variable to track the best validation metric
        best_val_metric = float('inf')  # For metrics like loss; use -float('inf') for accuracy
        best_model_path = None

        max_checkpoints = self.cfg.log.max_checkpoints  # Maximum number of recent checkpoints to keep
        saved_checkpoints = []  # List to track saved checkpoint paths

        for epoch in range(self.first_epoch, self.cfg.train.num_epoch):
            torch.cuda.reset_peak_memory_stats()

            self.before_epoch(epoch)

            train_stats = self.train_one_epoch(epoch)

            # Perform validation at the end of each epoch
            val_stats = self.validate(epoch)

            save_checkpoints = bool(self.cfg.log.get("save_checkpoints", True))
            save_best_model = bool(self.cfg.log.get("save_best_model", save_checkpoints))

            current_val_metric = val_stats.get("loss", float('inf'))  # Replace "val_loss" with your metric key
            if save_best_model and current_val_metric < best_val_metric:
                best_val_metric = current_val_metric
                best_model_path = os.path.join(
                    self.cfg.log.ckpt_dir,
                    "best_model",
                )
                self.accelerator.save_state(best_model_path, safe_serialization=True)
                self.log_info(f"Saved best model at epoch {epoch} with val_metric: {best_val_metric:.4f}")

            self.accelerator.wait_for_everyone()

            if save_checkpoints and (
                (epoch + 1) % self.cfg.log.ckpt_interval == 0
                or epoch + 1 == self.cfg.train.num_epoch
            ):
                if self.accelerator.sync_gradients:
                    self.global_step = (self.iters_per_epoch * (epoch + 1)) // self.cfg.train.gradient_accumulation_steps
                    save_path = os.path.join(
                        self.cfg.log.ckpt_dir,
                        f"checkpoint_{epoch}",
                    )
                    self.accelerator.save_state(save_path, safe_serialization=True)
                    self.log_info(
                        f"Saved state for global step {self.global_step}"
                    )

                    # Manage saved checkpoints
                    saved_checkpoints.append(save_path)
                    if self.accelerator.is_main_process and len(saved_checkpoints) > max_checkpoints:
                        oldest_checkpoint = saved_checkpoints.pop(0)
                        if os.path.exists(oldest_checkpoint):
                            shutil.rmtree(oldest_checkpoint)
                            self.log_info(f"Removed old checkpoint: {oldest_checkpoint}")

                self.accelerator.wait_for_everyone()

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_{k}": v for k, v in val_stats.items()},
                "epoch": epoch,
                "n_parameters": self.n_learnable_parameters,
            }

            if self.accelerator.is_main_process:
                with open(
                    os.path.join(self.cfg.log.ckpt_dir, "log.txt"),
                    mode="a",
                    encoding="utf-8",
                ) as f:
                    f.write(json.dumps(log_stats) + "\n")

                self.log_all(log_stats, step=self.global_step)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.log_info("Training time {}".format(total_time_str))

        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()

    def validate(self, epoch):
        self.model.eval()
        metric_logger = MetricLogger(delimiter="  ")
        header = f"Validation Epoch: [{epoch}]"

        val_loss = 0.0
        total_samples = 0

        self.log_info(f"Start validation for epoch {epoch}")
        with torch.no_grad():
            val_iterable = _LimitedIterable(self.test_loader, self.iters_per_test)
            for it, batch in enumerate(metric_logger.log_every(
                val_iterable, self.cfg.train.print_freq, header
            )):
                batch = move_to_device(batch, self.accelerator.device)

                # Forward pass
                outputs = self.forward_batch(batch, mode='test')
                
                outputs = self.calculate_loss(
                    outputs, 
                    batch, 
                    mode='test', 
                    current_epoch=epoch, 
                    total_epochs=self.cfg.train.num_epoch,
                    batch_idx=it  # <--- [修改点 2]: 传入局部 batch_idx
                )
                loss = outputs.loss
                # Gather statistics
                loss_value = loss.item()
                val_loss += loss_value * len(batch)
                total_samples += len(batch)

                # self.log_all(outputs, self.global_step, prefix='val')
                # scalar_metrics = {
                #         k: v for k, v in outputs.items() 
                #         if not (isinstance(v, torch.Tensor) and v.numel() > 1)
                #     }
                # metric_logger.update(**scalar_metrics)
                scalar_metrics = {
                    k: v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v 
                    for k, v in outputs.items() 
                    if not (isinstance(v, torch.Tensor) and v.numel() > 1)
                }
                metric_logger.update(**scalar_metrics)
                # metric_logger.update(**outputs)

        # Average the validation loss
        val_loss /= total_samples

        # Gather the stats from all processes
        metric_logger.synchronize_between_processes()
        self.log_info(f"Validation results: {metric_logger}")

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    def train_one_epoch(self, epoch):
        self.model.train()
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter(
            "min_lr", SmoothedValue(window_size=1, fmt="{value:.6f}")
        )
        # metric_logger.add_meter(
        #     "dataloader", SmoothedValue(window_size=1, fmt="{value:.6f}")
        # )
        header = "Epoch: [{}]".format(epoch)
        loss_details_dict = {}
        # start_steps = epoch * self.iters_per_epoch
        start_steps = (epoch * self.iters_per_epoch) // self.cfg.train.gradient_accumulation_steps
        self.global_step = start_steps

        self.log_info(
            "Start training epoch {}, {} iters per inner epoch. Training dtype {}".format(
                epoch, self.iters_per_epoch, self.cfg.train.model_dtype
            )
        )

        for it, batch in enumerate(metric_logger.log_every(
            self.train_loader, self.cfg.train.print_freq, header
        )):
            if it >= self.iters_per_epoch:
                break

            with self.accelerator.accumulate(self.model):
                # Perform the forward using the accerlate
                batch = move_to_device(batch, device=self.accelerator.device)
                with self.accelerator.autocast():
                    forward_output = self.forward_batch(batch, mode='train',global_step=self.global_step)
                
                # [修改点 2] 传递 current_epoch 和 total_epochs 到 calculate_loss
                batch_output = self.calculate_loss(
                    forward_output, 
                    batch, 
                    mode='train', 
                    current_epoch=epoch, 
                    total_epochs=self.cfg.train.num_epoch,
                    batch_idx=self.global_step
                )
                
                loss = batch_output.loss
                if not isinstance(loss, torch.Tensor):
                    raise TypeError(
                        f"calculate_loss must return a tensor loss, got {type(loss).__name__} "
                        f"at epoch {epoch}, iter {it}, global step {self.global_step}."
                    )
                if loss.numel() != 1:
                    raise ValueError(
                        f"calculate_loss must return a scalar tensor loss, got shape {tuple(loss.shape)} "
                        f"at epoch {epoch}, iter {it}, global step {self.global_step}."
                    )

                if bool((loss.detach() > self.cfg.train.clip_loss).item()):
                    loss = loss * 0.0

                # Check if the loss is nan
                loss_value = float(loss.detach().float().item())
                if not math.isfinite(loss_value):
                    rank = get_rank()
                    print(
                        f"Rank {rank}: Loss is {loss_value}, stopping training at iter {it} (epoch {epoch}, global step {self.global_step}).",
                        force=True,
                    )
                    sys.exit(1)

                if not loss.requires_grad:
                    zero_grad_anchor = self._zero_grad_anchor_from_trainable_params()
                    no_grad_context = self._format_no_grad_loss_context(batch_output)
                    if zero_grad_anchor is None:
                        raise RuntimeError(
                            "Loss does not require grad and no trainable model parameter was found. "
                            f"epoch={epoch}, iter={it}, global_step={self.global_step}. "
                            f"Scalar details: {no_grad_context}"
                        )
                    if self._no_grad_loss_warn_count < 8:
                        self.log_info(
                            "Loss does not require grad; adding a zero-gradient anchor so this "
                            f"degenerate batch contributes zero gradients. epoch={epoch}, iter={it}, "
                            f"global_step={self.global_step}. Scalar details: {no_grad_context}"
                        )
                    self._no_grad_loss_warn_count += 1
                    loss = loss + zero_grad_anchor
                    batch_output.loss = loss

                self.accelerator.backward(loss)

                # for item in batch_output:
                #     if 'loss' in item:
                #         batch_output[item] = self.accelerator.gather(batch_output[item]).mean().item()
                #         if item in loss_details_dict:
                #             loss_details_dict[item] += batch_output[item] / self.cfg.train.gradient_accumulation_steps if loss_value != 0 else 0.0
                #         else:
                #             loss_details_dict[item] = batch_output[item] / self.cfg.train.gradient_accumulation_steps if loss_value != 0 else 0.0
                # ------------------- [修改开始: 收集所有数值标量] -------------------
                for item in batch_output:
                    val = batch_output[item]
                    
                    # 如果是 tensor 标量，跨多卡 gather 取平均并转为 float
                    if isinstance(val, torch.Tensor) and val.numel() == 1:
                        # 使用 .view(-1) 防止 0-dim tensor 在 gather 时报错
                        val = self.accelerator.gather(val.view(-1)).mean().item()
                    elif not isinstance(val, (int, float)):
                        continue # 跳过图片、字符串等非数值项
                        
                    if item in loss_details_dict:
                        loss_details_dict[item] += val / self.cfg.train.gradient_accumulation_steps if loss_value != 0 else 0.0
                    else:
                        loss_details_dict[item] = val / self.cfg.train.gradient_accumulation_steps if loss_value != 0 else 0.0
                # ------------------- [修改结束] -------------------
                # clip the gradient
                if self.accelerator.sync_gradients:
                    params_to_clip = self.model.parameters()
                    self.accelerator.clip_grad_norm_(
                        params_to_clip, self.cfg.train.clip_grad
                    )

                    def get_gradient_norm(parameters):
                        norm = 0
                        for param in parameters:
                            if param.grad is None:
                                continue
                            local_norm = param.grad.detach().data.norm(2)
                            norm += local_norm.item() ** 2
                        norm = norm**0.5
                        return norm

                    grad_norm = get_gradient_norm(self.model.parameters())

                if self.accelerator.state.deepspeed_plugin is None:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                self.lr_scheduler.step()

                if self.accelerator.sync_gradients:
                    start_steps += 1

                    # Report to tensorboard / WandB
                    batch_output.update(loss_details_dict)
                    loss_details_dict = {}
                    
                    # ------------------- [修改开始: 移除 40 步限制] -------------------
                    # 每步都传给 WandB，保证没有任何 loss 数据遗漏。WandB 后台会异步上传，不卡训练。
                    self.log_all(batch_output, start_steps, prefix='train')
                    # ------------------- [修改结束] -------------------
                    
                    # 控制台日志依然保留平滑打印（依赖你 config 中的 print_freq）
                    # 过滤掉多元素的张量（如图像），只保留标量用于平滑计算和终端打印
                    scalar_metrics = {
                        k: v for k, v in batch_output.items() 
                        if not (isinstance(v, torch.Tensor) and v.numel() > 1)
                    }
                    metric_logger.update(**scalar_metrics)
                    # metric_logger.update(**batch_output)

                    min_lr = 10.0
                    max_lr = 0.0
                    for group in self.optimizer.param_groups:
                        min_lr = min(min_lr, group["lr"])
                        max_lr = max(max_lr, group["lr"])

                    metric_logger.update(lr=max_lr)
                    metric_logger.update(min_lr=min_lr)
                    self.log_scalar_metrics({"lr": max_lr, "min_lr": min_lr}, step=start_steps)

                    weight_decay_value = None
                    for group in self.optimizer.param_groups:
                        if group["weight_decay"] > 0:
                            weight_decay_value = group["weight_decay"]
                    metric_logger.update(weight_decay=weight_decay_value)
                    metric_logger.update(grad_norm=grad_norm)
                    self.log_scalar_metrics(
                        {"weight_decay": weight_decay_value, "grad_norm": grad_norm},
                        step=start_steps,
                    )

                    self.global_step = start_steps
                    del forward_output, batch_output, loss, batch

        # # gather the stats from all processes
        # metric_logger.synchronize_between_processes()
        # print("Averaged stats:", metric_logger)

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


    def _format_log_key(self, key, prefix=""):
        return f"{prefix}/{key}" if prefix else str(key)

    def _metric_allowed(self, key):
        allowlist = self.cfg.log.get("metric_allowlist", None)
        if allowlist is None:
            return True
        key = str(key).lstrip("/")
        for pattern in allowlist:
            pattern = str(pattern).lstrip("/")
            if pattern == "*" or fnmatch.fnmatchcase(key, pattern):
                return True
        return False

    def log_scalar_metrics(self, metrics, step, prefix=""):
        log_scaler = {}
        for key, value in metrics.items():
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                value = value.item()
            if not (np.isscalar(value) or isinstance(value, (int, float))):
                continue
            log_key = self._format_log_key(key, prefix=prefix)
            if self._metric_allowed(log_key):
                log_scaler[log_key] = value
        if log_scaler:
            self.accelerator.log(log_scaler, step)

    def log_all(self, output, step, prefix=""):
        if 'log_keys' in output:
            log_keys = output.log_keys
        else:
            log_keys = list(output.keys())

        log_scaler = {}
        log_img = {}
        log_images = bool(self.cfg.log.get("log_images", False))
        for k in log_keys:
            v = output[k]
            # ------------------- [修改开始: 增加对 Tensor 标量的兼容] -------------------
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                v = v.item()
                
            if np.isscalar(v) or isinstance(v, (int, float)):
                log_key = self._format_log_key(k, prefix=prefix)
                if self._metric_allowed(log_key):
                    log_scaler[log_key] = v
                continue
            # ------------------- [修改结束] -------------------
            if log_images and isinstance(v, Image.Image):
                log_img[prefix+'/'+k] = v

        if log_scaler:
            self.accelerator.log(log_scaler, step)
        if log_images and log_img:
            for tracker in self.accelerator.trackers:
                tracker.log_images(log_img, step)

    def forward_batch(self, batch, mode='train'):
        output = self.model(batch)
        assert isinstance(output, EasyDict)
        return output

    # [修改点 3] 更新函数签名，接收 current_epoch 和 total_epochs
    # [修改点 4] 更新函数签名，增加 batch_idx
    def calculate_loss(self, output, batch, mode='train', current_epoch=None, total_epochs=None, batch_idx=0):
        pass

    def build_accelerator(self):
        accelerator_project_config = ProjectConfiguration(
            project_dir=self.cfg.log.output_dir,
            logging_dir=self.cfg.log.output_dir,
            total_limit=4,      # self.cfg.save_total_limit = 4
            # automatic_checkpoint_naming=True,
        )

        # Initialize the Environment variables throught MPI run
        init_distributed_mode(
            self.cfg.train, init_pytorch_ddp=False
        )  # set `init_pytorch_ddp` to False, since the accelerate will do later

        if self.cfg.log.use_wandb:
            log_with = 'wandb'
        elif self.cfg.log.use_tensorboard:
            log_with = 'tensorboard'
        else:
            log_with = 'all'

        mixed_precision = 'no' if self.cfg.train.model_dtype not in ['fp8', 'fp16', 'bf16'] else self.cfg.train.model_dtype

        # For mixed precision training we cast all non-trainable weights to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        self.weight_dtype = torch.float32
        if mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # dynamic complie
        if self.cfg.train.get("dynamo_backend"):
            if isinstance(self.cfg.train.dynamo_backend, str) and hasattr(
                DynamoBackend, self.cfg.train.dynamo_backend.upper()
            ):
                dynamo_backend = getattr(DynamoBackend, self.cfg.train.dynamo_backend.upper())
            elif isinstance(self.cfg.train.dynamo_backend, DynamoBackend):
                dynamo_backend = self.cfg.train.dynamo_backend
            else:
                print(
                    f"Invalid dynamo_backend {self.cfg.train.dynamo_backend}, using default. Please refer to "
                    "https://huggingface.co/docs/accelerate/v1.2.1/en/package_reference/utilities#accelerate.utils.DynamoBackend for available names."
                )
        else:
            dynamo_backend = DynamoBackend.NO

        print(f"Using dynamo backend: {dynamo_backend}")

        torch._inductor.config.reorder_for_compute_comm_overlap = True

        dynamo_plugin = TorchDynamoPlugin(
            backend=dynamo_backend,
            mode="max-autotune-no-cudagraphs",
            dynamic=self.cfg.train.get("dynamic_compile", True),
        )
        
        accelerate_config = dict(
            gradient_accumulation_steps=self.cfg.train.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=log_with,
            project_config=accelerator_project_config,
            dataloader_config=DataLoaderConfiguration(
                non_blocking=True,
                split_batches=False,
                dispatch_batches=None,
                even_batches=True,
                use_seedable_sampler=False,
            ),
            step_scheduler_with_optimizer=False,             # not to step n_gpus times per step.
            dynamo_plugin=dynamo_plugin,
        )

        # fsdp
        if self._fsdp_requested():
            if self.cfg.get("fsdp_plugin"):
                from torch.distributed.fsdp import MixedPrecision

                fsdp_plugin = hydra.utils.instantiate(self.cfg.fsdp_plugin)
                # Keep FSDP input casting aligned with the original bf16/fp16 training path.
                fsdp_plugin.mixed_precision_policy = MixedPrecision(
                    param_dtype=self.weight_dtype,
                    reduce_dtype=self.weight_dtype,
                    buffer_dtype=self.weight_dtype,
                    cast_forward_inputs=True,
                    cast_root_forward_inputs=True,
                )
                accelerate_config["fsdp_plugin"] = fsdp_plugin
        else:
            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=self.cfg.train.find_unused_parameters)
            accelerate_config['kwargs_handlers'] = [ddp_kwargs]

        accelerator = Accelerator(**accelerate_config)

        self.logger = get_logger(self.cfg, os.path.basename(__file__))

        # To block the print on non main process
        setup_for_distributed(accelerator.is_main_process)

        # self.logger.rank_zero_only = False
        self.log_info(accelerator.state)
        # self.logger.rank_zero_only = True
        # ------------------- [修改开始] -------------------
        if accelerator.is_main_process:
            # 准备 WandB 的初始化参数
            init_kwargs = {}
            if self.cfg.log.use_wandb:
                init_kwargs["wandb"] = {
                    "name": self.cfg.log.exp_name if "exp_name" in self.cfg.log else os.path.basename(
                        self.cfg.log.output_dir),
                    "entity": self.cfg.log.wandb_entity if "wandb_entity" in self.cfg.log else None,
                    # 你可以在这里添加更多 wandb.init 的参数
                }

            # 使用项目名称初始化 Trackers
            project_name = self.cfg.log.project_name if "project_name" in self.cfg.log else "Pi3_3DGS"
            accelerator.init_trackers(project_name, config=OmegaConf.to_container(self.cfg, resolve=True),
                                           init_kwargs=init_kwargs)
        # ------------------- [修改结束] -------------------
        if self.cfg.random_seed is not None:
            set_seed(self.cfg.random_seed, device_specific=True)

        self.device = accelerator.device

        self.accelerator = accelerator

    def auto_resume(self):
        if self.cfg.train.resume:
            path = self.cfg.train.resume
        elif os.path.exists(self.cfg.log.ckpt_dir):
            # Get the most recent checkpoint
            dirs = os.listdir(self.cfg.log.ckpt_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint_")]
            dirs = sorted(dirs, key=lambda x: int(x.split("_")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
            if path is not None:
                path = os.path.join(self.cfg.log.ckpt_dir, path)
        else:
            path = None

        if path is None:
            self.log_info("Checkpoint does not exist. Starting a new training run.")
            start_epoch = 0
        else:
            self.log_info(f"Resuming from checkpoint {path}")
            start_epoch = self._get_resume_epoch(path)
            checkpoint_group_sizes = self._get_checkpoint_optimizer_group_sizes(path)
            current_group_sizes = self._get_optimizer_group_sizes()

            if (
                checkpoint_group_sizes is not None
                and current_group_sizes is not None
                and checkpoint_group_sizes != current_group_sizes
            ):
                self.log_info(
                    "Optimizer state is incompatible with the current trainable parameter "
                    f"groups. Checkpoint groups: {checkpoint_group_sizes}, current groups: "
                    f"{current_group_sizes}. Loading model weights only and restarting the "
                    "optimizer / LR scheduler from step 0."
                )
                self._load_model_only(path)
                start_epoch = 0
            else:
                try:
                    self.accelerator.load_state(path)
                except ValueError as exc:
                    if not self._is_optimizer_state_mismatch(exc):
                        raise

                    self.log_info(
                        "Optimizer state could not be restored after a parameter-group change "
                        f"({exc}). Loading model weights only and restarting the optimizer / "
                        "LR scheduler from step 0."
                    )
                    self._load_model_only(path)
                    start_epoch = 0

        return start_epoch

    def _get_resume_epoch(self, path):
        if "checkpoint_" in path:
            checkpoint_name = path.rstrip("/").split("/")[-1]
            return int(checkpoint_name.split("checkpoint_")[-1]) + 1

        # Best-model checkpoints do not encode the epoch in their directory name.
        return 0

    def _get_optimizer_group_sizes(self):
        optimizer = getattr(self.optimizer, "optimizer", self.optimizer)
        if not hasattr(optimizer, "param_groups"):
            return None
        return [len(group.get("params", [])) for group in optimizer.param_groups]

    def _get_checkpoint_optimizer_group_sizes(self, path):
        optimizer_path = os.path.join(path, "optimizer.bin")
        if not os.path.exists(optimizer_path):
            return None

        optimizer_state = self._torch_load_cpu(optimizer_path)
        if not isinstance(optimizer_state, dict):
            return None

        param_groups = optimizer_state.get("param_groups", [])
        return [len(group.get("params", [])) for group in param_groups]

    def _torch_load_cpu(self, path):
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def _load_model_only(self, path):
        state_dict = None
        candidate_loaders = [
            (os.path.join(path, "model.safetensors"), lambda p: load_safetensors(p, device="cpu")),
            (os.path.join(path, "pytorch_model.bin"), self._torch_load_cpu),
            (os.path.join(path, "pytorch_model_fsdp.bin"), self._torch_load_cpu),
        ]
        model_path = None
        for candidate_path, loader in candidate_loaders:
            if os.path.exists(candidate_path):
                model_path = candidate_path
                state_dict = loader(candidate_path)
                break

        if state_dict is None:
            raise FileNotFoundError(
                f"Could not find model weights under checkpoint directory: {path}"
            )

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        result = unwrapped_model.load_state_dict(state_dict, strict=False)
        self.log_info(
            f"Model-only resume from {model_path}: "
            f"{len(result.missing_keys)} missing keys, "
            f"{len(result.unexpected_keys)} unexpected keys."
        )
        self.accelerator.wait_for_everyone()

    def _is_optimizer_state_mismatch(self, exc):
        message = str(exc)
        mismatch_markers = (
            "different number of parameter groups",
            "doesn't match the size of optimizer's group",
        )
        return any(marker in message for marker in mismatch_markers)

    def log_info(self, info):
        if is_logging_process():
            self.logger.info(info)
