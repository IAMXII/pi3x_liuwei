from .base.transforms import *

from utils.misc import get_world_size, get_rank
from torch.utils.data import DataLoader
import hydra
from omegaconf import OmegaConf
from datasets.base.base_dataset import sample_resolutions, unified_collate_fn
from datasets.base.batched_sampler import DynamicBatchSampler, DynamicDistributedSampler

__HIGH_QUALITY_DATASETS__ = ['BlinkVision', 'Game', 'GameNew', 'DynamicStereo', 'FlyingThings3D', 'GTA-sfm', 'Hypersim', 'MatrixCity', 'MidAir', 'Monkaa', 'PointOdyssey', 'Sintel', 'Spring', 'TarTanAir', 'Unreal4k', 'VirtualKitti', 'Habitat']
__MIDDLE_QUALITY_DATASETS__ = ['BlendedMVG', 'BlendedMVS', 'DTU', 'ETH3D', 'ScanNet', 'Scannetpp', 'Taskonomy']
__INDOOR_DATASETS__ = ['Hypersim', 'ScanNet', 'Scannetpp', 'Taskonomy', 'ARKitScenes', 'Habitat']


def _cfg_get(container, key, default=None):
    if container is None:
        return default
    try:
        if isinstance(container, dict) and key in container:
            return container[key]
        if OmegaConf.is_config(container) and key in container:
            return container[key]
        return getattr(container, key)
    except Exception:
        return default


def _find_key_case_insensitive(mapping, requested):
    if requested in mapping:
        return requested
    requested_lower = str(requested).lower()
    for key in mapping:
        if str(key).lower() == requested_lower:
            return key
    return None


def _dataset_enabled(cfg, dataset_name):
    switches = _cfg_get(cfg, "dataset_switches")
    if switches is None:
        return True
    key = str(dataset_name).lower()
    try:
        if key in switches:
            return bool(switches[key])
    except Exception:
        pass
    return True


def _filter_eval_weights(cfg, weights, mode):
    if mode == "train":
        return weights

    eval_only_dataset = _cfg_get(_cfg_get(cfg, "test"), "eval_only_dataset")
    if eval_only_dataset in (None, ""):
        return weights

    dataset_key = _find_key_case_insensitive(weights, str(eval_only_dataset))
    if dataset_key is None:
        available = ", ".join(str(key) for key in weights.keys())
        raise ValueError(
            f"test.eval_only_dataset={eval_only_dataset!r} is not in test_dataset.weights. "
            f"Available datasets: {available}"
        )

    print(f"[Validation] eval_only_dataset={dataset_key}; filtering validation dataset.", flush=True)
    return {dataset_key: weights[dataset_key]}


def _instantiate_weighted_dataset(cfg, cfg_dataset, dataset_name, mode, resolution=None):
    kwargs = {}
    if resolution is not None:
        kwargs["resolution"] = resolution

    eval_only_scene = _cfg_get(_cfg_get(cfg, "test"), "eval_only_scene")
    if mode != "train" and eval_only_scene not in (None, "") and str(dataset_name).lower() == "360_v2":
        kwargs["scene_names"] = eval_only_scene
        print(f"[Validation] 360_v2 scene filter: {eval_only_scene}", flush=True)

    return hydra.utils.instantiate(cfg_dataset[dataset_name], **kwargs)


def _warn_limited_unshuffled_validation(cfg, cfg_dataloader, component_ranges, world_size):
    if not component_ranges or len(component_ranges) < 2:
        return
    if bool(_cfg_get(cfg_dataloader, "shuffle", False)):
        return

    iters_per_test = _cfg_get(_cfg_get(cfg, "test"), "iters_per_test", -1)
    try:
        iters_per_test = int(iters_per_test)
    except Exception:
        return
    if iters_per_test <= 0:
        return

    covered_global_prefix = iters_per_test * max(1, int(world_size))
    unreached = [
        item for item in component_ranges
        if item["start"] >= covered_global_prefix
    ]
    partial = [
        item for item in component_ranges
        if item["start"] < covered_global_prefix < item["end"]
    ]
    if not unreached and not partial:
        return

    ranges = ", ".join(
        f"{item['name']}[{item['start']}:{item['end']})"
        for item in component_ranges
    )
    print(
        "\n[Validation Coverage Warning] test_dataloader.shuffle=false and "
        f"test.iters_per_test={iters_per_test} only cover about the first "
        f"{covered_global_prefix} global validation samples across {world_size} rank(s). "
        f"Dataset ranges: {ranges}.",
        flush=True,
    )
    if partial:
        print(
            "[Validation Coverage Warning] Partially covered dataset(s): "
            + ", ".join(item["name"] for item in partial),
            flush=True,
        )
    if unreached:
        print(
            "[Validation Coverage Warning] Unreached dataset(s): "
            + ", ".join(item["name"] for item in unreached)
            + ". Enable test_dataloader.shuffle=true or set test.eval_only_dataset.",
            flush=True,
        )


def create_dataloader(cfg, mode):
    data_loader = DataLoader
    num_resolution = 1
    component_ranges = []

    # pytorch dataset
    if mode == 'train':
        cfg_dataset = cfg.train_dataset
        cfg_dataloader = cfg.train_dataloader
        batch_size = cfg.train.batch_size
        num_workers = cfg.train.num_workers
    else:
        cfg_dataset = cfg.test_dataset
        cfg_dataloader = cfg.test_dataloader
        batch_size = cfg.test.batch_size
        num_workers = cfg.test.num_workers

    if isinstance(cfg_dataset, str):
        dataset = eval(cfg_dataset) 
    elif 'weights' in cfg_dataset:
        weights = _filter_eval_weights(cfg, cfg_dataset.weights, mode)
        if 'length' in cfg_dataset:
            dataset_length = cfg_dataset.length
            weight_sum = sum([v for k, v in weights.items()])
            new_weights = {}
            for dataset_name, weight in weights.items():
                new_weights[dataset_name] = max(int(weight / weight_sum * dataset_length), 1)
            weights = new_weights
            print(f'New weights for dataset (adjusting to dataset length {dataset_length}): {new_weights}')

        datasets_all = []
        current_start = 0

        def append_weighted_dataset(dataset_name, weight, resolution=None):
            nonlocal current_start
            if not _dataset_enabled(cfg, dataset_name):
                return
            dataset_i = _instantiate_weighted_dataset(cfg, cfg_dataset, dataset_name, mode, resolution=resolution)
            dataset_i.convert_attributes()
            weighted_dataset = weight @ dataset_i
            weighted_length = len(weighted_dataset)
            datasets_all.append(weighted_dataset)
            component_ranges.append({
                "name": str(dataset_name),
                "start": current_start,
                "end": current_start + weighted_length,
                "length": weighted_length,
                "base_length": len(dataset_i),
            })
            current_start += weighted_length

        num_resolution = cfg.train.num_resolution if 'num_resolution' in cfg.train and mode == 'train' else 1
        if mode == 'train' and 'random_reslution' in cfg.train and cfg.train.random_reslution:
            seed = 777 + 0
            resolutions = sample_resolutions(aspect_ratio_range=cfg.train.aspect_ratio_range, pixel_count_range=cfg.train.pixel_count_range, patch_size=cfg.train.patch_size, num_resolutions=num_resolution, seed=seed)
            print('Initialized resolution', resolutions)
            num_resolution = len(resolutions)
            for dataset_name, weight in weights.items():
                append_weighted_dataset(dataset_name, weight, resolution=resolutions)
        elif 'resolution' in cfg.train:
            resolutions = cfg.train.resolution
            print('Setting dataset resolution', resolutions)
            for dataset_name, weight in weights.items():
                append_weighted_dataset(dataset_name, weight, resolution=resolutions)
        else:
            for dataset_name, weight in weights.items():
                append_weighted_dataset(dataset_name, weight)
        if not datasets_all:
            raise ValueError(f"No datasets were enabled for mode={mode}.")
        dataset = datasets_all[0]
        for dataset_ in datasets_all[1:]:
            dataset += dataset_
    else:
        dataset = hydra.utils.instantiate(cfg_dataset)
        dataset.convert_attributes()
    world_size = get_world_size()
    rank = get_rank()
    if mode != 'train':
        _warn_limited_unshuffled_validation(cfg, cfg_dataloader, component_ranges, world_size)

    image_num_range = cfg.train.image_num_range if mode == 'train' else [8, 8]
    print(f'Sampling frame number range from {image_num_range}')
    # adapte from vggt
    max_img_per_gpu = cfg.train.max_img_per_gpu if 'max_img_per_gpu' in cfg.train else image_num_range[0]
    print(f'Max frame number per rank {max_img_per_gpu}')
    if mode == 'train' and cfg.train.iters_per_epoch > 0:
        print('Needed batch number per epoch (per rank):', (max_img_per_gpu // image_num_range[0]) * cfg.train.iters_per_epoch)
        print('Dataset length per rank:', len(dataset) // world_size)
        assert (max_img_per_gpu // image_num_range[0]) * cfg.train.iters_per_epoch < len(dataset) // world_size

    sampler = DynamicDistributedSampler(dataset, seed=cfg.train.base_seed, shuffle=cfg_dataloader.shuffle, rank=rank, drop_last=cfg_dataloader.drop_last)
    batch_sampler = DynamicBatchSampler(
        sampler, 
        num_resolution, 
        image_num_range, 
        seed=cfg.train.base_seed,
        max_img_per_gpu=max_img_per_gpu,
        rank=rank
    )
    print(batch_sampler,flush=True)
    return data_loader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        # sampler=sampler,        # <--- 修改: 直接使用底层的 DistributedSampler
        # batch_size=1,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=unified_collate_fn
    )
