from trainers.base_trainer_accelerate import BaseTrainer
from easydict import EasyDict
import torch
import inspect
from datasets.base.base_dataset import sample_resolutions
import hydra

from pi3.models.loss import Pi3Loss
from pi3.models.loss_3dgs import Pi3LossGS

class Pi3Trainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.train_loss = hydra.utils.instantiate(cfg.loss.train_loss)
        self.test_loss = hydra.utils.instantiate(cfg.loss.train_loss)

    # def build_optimizer(self, cfg_optimizer, model):
    #     def param_group_fn(model_):
    #         encoder_params = [param for param in model_.encoder.named_parameters()]
    #         other_params = [
    #             (name, param) for name, param in model_.named_parameters()
    #             if not name.startswith("encoder.") and not '.encoder.' in name
    #         ]

    #         print(f'Number of trainable encoder parameters:', sum(p.numel() for _, p in encoder_params if p.requires_grad))
    #         print(f'Length of trainable others:', sum(p.numel() for _, p in other_params if p.requires_grad))

    #         def handle_weight_decay(params, weight_decay, lr):
    #             decay = []
    #             no_decay = []
    #             for name, param in params:
    #                 if not param.requires_grad:
    #                     continue

    #                 if param.ndim <= 1 or name.endswith(".bias"):
    #                     no_decay.append(param)
    #                 else:
    #                     decay.append(param)

    #             return [
    #                 {"params": no_decay, "weight_decay": 0.0, 'lr': lr},
    #                 {"params": decay, "weight_decay": weight_decay, 'lr': lr},
    #             ]

    #         res = []
    #         res.extend(handle_weight_decay(encoder_params, cfg_optimizer.weight_decay, cfg_optimizer.encoder_lr))
    #         res.extend(handle_weight_decay(other_params, cfg_optimizer.weight_decay, cfg_optimizer.lr))

    #         return res
        
    #     return super().build_optimizer(cfg_optimizer, model, param_group_fn=param_group_fn)
    def build_optimizer(self, cfg_optimizer, model):
        def param_group_fn(model_):
            # 1. 精细化参数分组容器
            encoder_params = []
            point_decoder_params = []
            gs_decoder_params = []
            gs_head_params = []
            camera_decoder_params = []
            other_params = []

            # 2. 遍历并分类参数
            for name, param in model_.named_parameters():
                if not param.requires_grad:
                    continue  # 跳过已经物理冻结的参数（如 encoder）

                if name.startswith("encoder.") or '.encoder.' in name:
                    encoder_params.append((name, param))
                elif 'point_decoder' in name:
                    point_decoder_params.append((name, param))
                elif 'gs_decoder' in name:
                    gs_decoder_params.append((name, param))
                elif 'gs_head' in name:
                    gs_head_params.append((name, param))
                elif 'camera_decoder' in name:
                    camera_decoder_params.append((name, param))
                else:
                    other_params.append((name, param))

            # 打印各组可训练参数的数量，方便你 debug 检查
            print(f'Trainable encoder params:', sum(p.numel() for _, p in encoder_params))
            print(f'Trainable point_decoder params:', sum(p.numel() for _, p in point_decoder_params))
            print(f'Trainable gs_decoder params:', sum(p.numel() for _, p in gs_decoder_params))
            print(f'Trainable gs_head params:', sum(p.numel() for _, p in gs_head_params))
            print(f'Trainable camera_decoder params:', sum(p.numel() for _, p in camera_decoder_params))
            print(f'Trainable other params:', sum(p.numel() for _, p in other_params))

            def handle_weight_decay(params, weight_decay, lr):
                decay = []
                no_decay = []
                for name, param in params:
                    if param.ndim <= 1 or name.endswith(".bias"):
                        no_decay.append(param)
                    else:
                        decay.append(param)

                groups = []
                # 增加非空判断，避免 PyTorch 优化器收到空的 param_group 报错
                if no_decay:
                    groups.append({"params": no_decay, "weight_decay": 0.0, 'lr': lr})
                if decay:
                    groups.append({"params": decay, "weight_decay": weight_decay, 'lr': lr})
                return groups

            res = []
            base_lr = cfg_optimizer.lr
            gs_head_lr_multiplier = float(getattr(cfg_optimizer, 'gs_head_lr_multiplier', 2.0))
            
            # 3. 分配差异化学习率 (核心精进策略)
            
            # Encoder: 如果有解冻层，用专门的 encoder_lr
            if encoder_params:
                res.extend(handle_weight_decay(encoder_params, cfg_optimizer.weight_decay, getattr(cfg_optimizer, 'encoder_lr', base_lr * 0.01)))
            
            # Point Decoder (几何): 极低学习率 (5%)，实现“软冻结”，只允许微小形变
            if point_decoder_params:
                res.extend(handle_weight_decay(point_decoder_params, cfg_optimizer.weight_decay, base_lr * 0.07))
            
            # # Camera Decoder (相机姿态): 收敛后期通常不需要大动 (1%)
            # if camera_decoder_params:
            #     res.extend(handle_weight_decay(camera_decoder_params, cfg_optimizer.weight_decay, base_lr * 0.01))
            
            # GS Decoder (颜色/透明度等): 现阶段的优化主力，保持 100% 基础学习率
            if gs_decoder_params:
                res.extend(handle_weight_decay(gs_decoder_params, cfg_optimizer.weight_decay, base_lr * 1.0))

            # GS Head 直接输出 opacity/color/scale 等渲染属性，给更高学习率以加快光度收敛
            if gs_head_params:
                res.extend(handle_weight_decay(gs_head_params, cfg_optimizer.weight_decay, base_lr * gs_head_lr_multiplier))
            
            # 其他主干网络 (如 Transformer 主体): 压低学习率 (10%)，稳定已有的特征空间
            if other_params:
                res.extend(handle_weight_decay(other_params, cfg_optimizer.weight_decay, base_lr * 1.0))

            return res
        
        return super().build_optimizer(cfg_optimizer, model, param_group_fn=param_group_fn)

    def before_epoch(self, epoch):
        if hasattr(self.train_loader, 'dataset') and hasattr(self.train_loader.dataset, 'set_epoch'):
            self.train_loader.dataset.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
        if hasattr(self.train_loader, 'sampler') and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
            # self.train_loader.sampler.set_epoch(epoch)
        if hasattr(self.train_loader, 'batch_sampler') and hasattr(self.train_loader.batch_sampler, 'batch_sampler') and hasattr(self.train_loader.batch_sampler.batch_sampler, 'sampler') and hasattr(self.train_loader.batch_sampler.batch_sampler.sampler, 'set_epoch'):       # handle acclerate warpped dataloader (more gpu)
            self.train_loader.batch_sampler.batch_sampler.sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
        if hasattr(self.train_loader, 'batch_sampler') and hasattr(self.train_loader.batch_sampler, 'set_epoch'):       # handle acclerate warpped dataloader (more gpu)
            self.train_loader.batch_sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
        

        if hasattr(self.test_loader, 'dataset') and hasattr(self.test_loader.dataset, 'set_epoch'):
            self.test_loader.dataset.set_epoch(0, base_seed=self.cfg.train.base_seed)
        if hasattr(self.test_loader, 'batch_sampler') and hasattr(self.test_loader.batch_sampler, 'batch_sampler') and hasattr(self.test_loader.batch_sampler.batch_sampler, 'sampler') and hasattr(self.test_loader.batch_sampler.batch_sampler.sampler, 'set_epoch'):       # handle acclerate warpped dataloader (more gpu)
            self.test_loader.batch_sampler.batch_sampler.sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
        if hasattr(self.test_loader, 'batch_sampler') and hasattr(self.train_loader.batch_sampler, 'set_epoch'):       # handle acclerate warpped dataloader (more gpu)
            self.test_loader.batch_sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)

        if 'random_reslution' in self.cfg.train and self.cfg.train.random_reslution and self.cfg.train.num_resolution > 0:
            seed = epoch + self.cfg.train.base_seed
            resolutions = sample_resolutions(aspect_ratio_range=self.cfg.train.aspect_ratio_range, pixel_count_range=self.cfg.train.pixel_count_range, patch_size=self.cfg.train.patch_size, num_resolutions=self.cfg.train.num_resolution, seed=seed)
            print('[Pi3 Trainer] Sampled new resolutions:', resolutions)
            datasets = []
            recursive_get_dataset(self.train_loader.dataset, datasets)
            for dataset in datasets:
                dataset._set_resolutions(resolutions)
            
    def forward_batch(self, batch, mode='train',global_step=None):
        imgs = torch.stack([view['img'] for view in batch], dim=1)
        # imgs_paired = torch.stack([view['img_paired'] for view in batch], dim=1) if 'img_paired' in batch[0] else None
        imgs_paired = torch.stack([view['img_paired'] for view in batch], dim=1) if ('img_paired' in batch[0] and isinstance(batch[0]['img_paired'], torch.Tensor)) else None
        intrinsics = torch.stack([view['camera_intrinsics'] for view in batch], dim=1)
        current_step = global_step if global_step is not None else 0

        forward_params = inspect.signature(self.model.forward).parameters
        model_kwargs = {}
        if 'imgs_paired' in forward_params:
            model_kwargs['imgs_paired'] = imgs_paired
        if 'intrinsics' in forward_params:
            model_kwargs['intrinsics'] = intrinsics
        if 'global_step' in forward_params:
            model_kwargs['global_step'] = current_step

        pred = self.model(imgs, **model_kwargs)

        return [pred, batch]
    
    def calculate_loss(self, output, batch, mode='train', current_epoch=None, total_epochs=None, batch_idx=0):
        output, batch = output

        if mode == 'train':
            loss, details = self.train_loss(
                output, batch,
                current_epoch=current_epoch, 
                total_epochs=total_epochs,
                batch_idx=batch_idx  # <--- [修改点 6]: 透传给 train_loss
            )
        else:
            loss, details = self.test_loss(
                output, batch,
                current_epoch=current_epoch, 
                total_epochs=total_epochs,
                batch_idx=batch_idx  # <--- [修改点 7]: 透传给 test_loss
            )

        return EasyDict(
            loss=loss,
            **details
        )


def recursive_get_dataset(dataset, res=[]):
    if hasattr(dataset, 'datasets'):
        for ds in dataset.datasets:
            recursive_get_dataset(ds, res)
    else:
        if hasattr(dataset, 'dataset'):
            res.append(dataset.dataset)
    return res
