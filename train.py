"""UniReg training entry point.

Two-stage UniReg-RPN training:
    python train.py --config configs/train/single_task/lumir_rpnet.yaml --gpu 0
    python train.py --config configs/train/multi_task/unireg_rpn_6task_from_brain.yaml \
        --init_checkpoint ./logs/lumir_rpnet/best.pth --gpu 0
"""



from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from torch.utils.data import DataLoader, ConcatDataset




def _expand_env(obj):
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expanduser(os.path.expandvars(obj))
    return obj

def parse_args():
    p = argparse.ArgumentParser("UniReg Training")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--resume", default=None, help="Checkpoint to resume from, including optimizer/step")
    p.add_argument("--init_checkpoint", default=None, help="Initialize model weights only; do not restore optimizer/step")
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--val_every", type=int, default=None)
    p.add_argument("--save_every", type=int, default=None)
    p.add_argument("--device", default=None, help="Device: cuda, cuda:0, cuda:1, cpu")
    p.add_argument("--gpu", type=int, default=None, help="GPU id. Overrides --device to cuda:N")
    p.add_argument("--log_dir", default=None)
    p.add_argument("--feat_loss", action="store_true", help="Enable feature loss")
    p.add_argument("--no_feat_loss", action="store_true", help="Disable feature loss")
    return p.parse_args()


def load_config(path: str, overrides: dict) -> dict:
    with open(path) as f:
        cfg = _expand_env(yaml.safe_load(f))
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


def _task_display_name(task_cfg: dict) -> str:
    return str(task_cfg.get("name", task_cfg.get("task_id", task_cfg.get("dataset", "task"))))


class TaggedTaskDataset(torch.utils.data.Dataset):
    """Attach task metadata to every sample returned by a normal dataset."""

    def __init__(self, dataset, task_cfg: dict):
        self.dataset = dataset
        self.task_cfg = dict(task_cfg)
        self.task_name = _task_display_name(task_cfg)
        self.task_type_id = int(task_cfg.get("task_type_id", 0))
        self.task_id_id = int(task_cfg.get("task_id_id", 0))
        self.use_pre_align = bool(task_cfg.get("use_pre_align", False))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = dict(self.dataset[index])
        item["task_name"] = self.task_name
        item["task_type_id"] = torch.tensor(self.task_type_id, dtype=torch.long)
        item["task_id_id"] = torch.tensor(self.task_id_id, dtype=torch.long)
        item["use_pre_align_task"] = torch.tensor(int(self.use_pre_align), dtype=torch.long)
        return item


class BalancedMultiTaskDataset(torch.utils.data.Dataset):
    """
    Round-robin task-level sampling for heterogeneous 3D datasets.

    If sampling_weight is [1,1,1], the cycle is task0 -> task1 -> task2.
    If sampling_weight is [2,2,1], the effective task ratio is 2:2:1.
    """

    def __init__(self, datasets: List[torch.utils.data.Dataset], weights=None, seed: int = 42):
        assert len(datasets) > 0, "BalancedMultiTaskDataset needs at least one task"
        self.datasets = list(datasets)
        self.seed = int(seed)
        weights = weights or [1] * len(self.datasets)
        if len(weights) != len(self.datasets):
            raise ValueError(f"weights length={len(weights)} does not match n_tasks={len(self.datasets)}")
        self.weights = [max(1, int(w)) for w in weights]
        self.order = []
        for task_idx, w in enumerate(self.weights):
            self.order.extend([task_idx] * w)
        self.max_len = max(len(ds) for ds in self.datasets)

    def __len__(self):
        return self.max_len * len(self.order)

    def __getitem__(self, index):
        task_idx = self.order[index % len(self.order)]
        local_round = index // len(self.order)
        ds = self.datasets[task_idx]
        local_idx = local_round % len(ds)
        return ds[local_idx]


def _merged_task_cfg(task_cfg: dict, global_cfg: dict) -> dict:
    merged = dict(global_cfg)
    merged.update(task_cfg)
    # Do not let the global multi_task name overwrite the real task dataset.
    if "tasks" in merged:
        merged.pop("tasks", None)
    return merged


def _build_one_dataset(task_cfg: dict, split: str, global_cfg: dict):
    from unireg.data.dataset import build_dataset

    merged = _merged_task_cfg(task_cfg, global_cfg)
    dataset_name = str(merged["dataset"]).lower()
    compute_size = tuple(merged["compute_size"])
    norm = merged.get("norm_strategy", merged.get("norm", None))
    seed = int(merged.get("seed", global_cfg.get("seed", 42)))
    eval_in_raw = bool(merged.get("eval_in_raw", global_cfg.get("eval_in_raw", False)))

    if split == "train":
        return build_dataset(
            dataset_name, split="train",
            compute_size=compute_size, norm=norm,
            img_dir=merged.get("train_root", None),
            seg_dir=merged.get("train_seg_root", merged.get("train_label_root", None)),
            pair_json=merged.get("pair_json", None),
            use_pre_align=bool(merged.get("use_pre_align", False)),
            num_repeats=merged.get("num_repeats", global_cfg.get("num_repeats", 10)),
            seed=seed,
        )
    if split == "val":
        return build_dataset(
            dataset_name, split="val",
            compute_size=compute_size, norm=norm,
            img_dir=merged.get("val_root", None),
            seg_dir=merged.get("val_seg_root", None),
            native_img_dir=merged.get("raw_val_root", None),
            native_seg_dir=merged.get("raw_val_seg_root", None),
            pair_json=merged.get("pair_json", None),
            use_pre_align=bool(merged.get("use_pre_align", False)),
            seed=seed,
            return_native=eval_in_raw,
        )
    if split == "test":
        return build_dataset(
            dataset_name, split="test",
            compute_size=compute_size, norm=norm,
            img_dir=merged.get("test_root", None),
            seg_dir=merged.get("test_seg_root", None),
            native_img_dir=merged.get("raw_test_root", None),
            native_seg_dir=merged.get("raw_test_seg_root", None),
            pair_json=merged.get("pair_json", None),
            use_pre_align=bool(merged.get("use_pre_align", False)),
            seed=seed,
            return_native=eval_in_raw,
        )
    raise ValueError(f"Unknown split={split}")


def build_loaders(cfg: dict):
    from unireg.data.dataset import build_dataset

    dataset_name = str(cfg["dataset"]).lower()
    batch_size = int(cfg.get("batch_size", 1))
    num_workers = int(cfg.get("num_workers", 4))
    seed = int(cfg.get("seed", 42))

    if dataset_name == "multi_task":
        task_cfgs = cfg.get("tasks", [])
        if not task_cfgs:
            raise ValueError("dataset: multi_task requires a non-empty 'tasks:' list")
        if batch_size != 1:
            raise ValueError("multi_task training requires batch_size: 1 because task volumes have different sizes")

        train_parts = [TaggedTaskDataset(_build_one_dataset(t, "train", cfg), t) for t in task_cfgs]
        val_parts = [TaggedTaskDataset(_build_one_dataset(t, "val", cfg), t) for t in task_cfgs]
        sampling_weights = [int(t.get("sampling_weight", 1)) for t in task_cfgs]
        train_ds = BalancedMultiTaskDataset(train_parts, weights=sampling_weights, seed=seed)
        val_ds = ConcatDataset(val_parts)

        expanded = []
        for t in task_cfgs:
            expanded.extend([_task_display_name(t)] * int(t.get("sampling_weight", 1)))
        print("[train.py] multi_task train cycle: " + " -> ".join(expanded), flush=True)

        return (
            DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True),
            DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True),
        )

    compute_size = tuple(cfg["compute_size"])
    norm = cfg.get("norm_strategy", None)
    num_repeats = cfg.get("num_repeats", 10)
    eval_in_raw = bool(cfg.get("eval_in_raw", False))

    train_ds = build_dataset(
        dataset_name, split="train",
        compute_size=compute_size, norm=norm,
        img_dir=cfg.get("train_root", None),
        seg_dir=cfg.get("train_seg_root", cfg.get("train_label_root", None)),
        pair_json=cfg.get("pair_json", None),
        use_pre_align=bool(cfg.get("use_pre_align", False)),
        num_repeats=num_repeats,
        seed=seed,
    )
    val_ds = build_dataset(
        dataset_name, split="val",
        compute_size=compute_size, norm=norm,
        img_dir=cfg.get("val_root", None),
        seg_dir=cfg.get("val_seg_root", None),
        native_img_dir=cfg.get("raw_val_root", None),
        native_seg_dir=cfg.get("raw_val_seg_root", None),
        pair_json=cfg.get("pair_json", None),
        use_pre_align=bool(cfg.get("use_pre_align", False)),
        seed=seed,
        return_native=eval_in_raw,
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True),
    )


def build_test_loader(cfg: dict):
    from unireg.data.dataset import build_dataset

    dataset_name = str(cfg["dataset"]).lower()
    num_workers = int(cfg.get("num_workers", 4))

    if dataset_name == "multi_task":
        parts = [TaggedTaskDataset(_build_one_dataset(t, "test", cfg), t) for t in cfg.get("tasks", [])]
        return DataLoader(ConcatDataset(parts), batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

    test_ds = build_dataset(
        cfg["dataset"], split="test",
        compute_size=tuple(cfg["compute_size"]),
        norm=cfg.get("norm_strategy", None),
        img_dir=cfg.get("test_root", None),
        seg_dir=cfg.get("test_seg_root", None),
        native_img_dir=cfg.get("raw_test_root", None),
        native_seg_dir=cfg.get("raw_test_seg_root", None),
        return_native=bool(cfg.get("eval_in_raw", False)),
        pair_json=cfg.get("pair_json", None),
        use_pre_align=bool(cfg.get("use_pre_align", False)),
    )
    if "task_type_id" in cfg or "task_id_id" in cfg:
        test_ds = TaggedTaskDataset(test_ds, cfg)
    return DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)


def build_losses(cfg: dict):
    from unireg.losses.registry import build_sim_loss, build_reg_loss
    return build_sim_loss(cfg), build_reg_loss(cfg)


def build_task_losses(cfg: dict):
    """Build per-task loss functions and weights for dataset: multi_task."""
    from unireg.losses.registry import build_sim_loss, build_reg_loss

    task_sim_losses: Dict[str, torch.nn.Module] = {}
    task_reg_losses: Dict[str, torch.nn.Module] = {}
    task_sim_weights: Dict[str, float] = {}
    task_reg_weights: Dict[str, float] = {}

    if str(cfg.get("dataset", "")).lower() != "multi_task":
        return task_sim_losses, task_reg_losses, task_sim_weights, task_reg_weights

    for task_cfg in cfg.get("tasks", []):
        name = _task_display_name(task_cfg)
        merged = _merged_task_cfg(task_cfg, cfg)
        task_sim_losses[name] = build_sim_loss(merged)
        task_reg_losses[name] = build_reg_loss(merged)
        task_sim_weights[name] = float(merged.get("sim_weight", cfg.get("sim_weight", 1.0)))
        task_reg_weights[name] = float(merged.get("reg_weight", cfg.get("reg_weight", 0.01)))
        print(
            f"[train.py] task={name}: sim_loss={merged.get('sim_loss', 'ncc')} "
            f"ncc_win={merged.get('ncc_win', cfg.get('ncc_win', 9))} "
            f"sim_weight={task_sim_weights[name]} reg_weight={task_reg_weights[name]}",
            flush=True,
        )
    return task_sim_losses, task_reg_losses, task_sim_weights, task_reg_weights


def main():
    args = parse_args()
    device_override = f"cuda:{args.gpu}" if args.gpu is not None else args.device
    overrides = {
        "max_steps": args.max_steps,
        "val_every": args.val_every,
        "save_every": args.save_every,
        "device": device_override,
        "log_dir": args.log_dir,
    }
    cfg = load_config(args.config, overrides)

    if args.feat_loss:
        cfg["use_feature_loss"] = True
        cfg["feat_weight"] = cfg.get("feat_weight") or 1.0
    if args.no_feat_loss:
        cfg["use_feature_loss"] = False
        cfg["feat_weight"] = 0.0

    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    log_dir = cfg.get("log_dir", "./logs")
    print(f"[train.py] device={device}", flush=True)

    from unireg.models.registry import build_model
    model = build_model(cfg, device=device)

    lr = cfg.get("lr", 1e-4)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    sim_loss, reg_loss = build_losses(cfg)
    task_sim_losses, task_reg_losses, task_sim_weights, task_reg_weights = build_task_losses(cfg)

    train_loader, val_loader = build_loaders(cfg)
    steps_per_epoch = len(train_loader)
    epochs = cfg.get("epochs", 100)
    val_epochs = cfg.get("val_epochs", 10)
    save_epochs = cfg.get("save_epochs", 10)

    max_steps_cfg = cfg.get("max_steps", None)
    max_steps = int(max_steps_cfg) if max_steps_cfg is not None else int(epochs * steps_per_epoch)
    val_every_cfg = cfg.get("val_every", None)
    save_every_cfg = cfg.get("save_every", None)
    val_every = int(val_every_cfg) if val_every_cfg is not None else max(1, int(val_epochs * steps_per_epoch))
    save_every = int(save_every_cfg) if save_every_cfg is not None else max(1, int(save_epochs * steps_per_epoch))

    scheduler = None
    if cfg.get("lr_scheduler", None) == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_steps, eta_min=cfg.get("lr_min", 1e-6)
        )

    from unireg.engine.trainer import Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        sim_loss=sim_loss,
        reg_loss=reg_loss,
        sim_weight=cfg.get("sim_weight", 1.0),
        reg_weight=cfg.get("reg_weight", 0.01),
        task_sim_losses=task_sim_losses,
        task_reg_losses=task_reg_losses,
        task_sim_weights=task_sim_weights,
        task_reg_weights=task_reg_weights,
        use_feature_loss=bool(cfg.get("use_feature_loss", False)),
        feat_weight=float(cfg.get("feat_weight", cfg.get("feature_loss_weight", 0.0)) or 0.0),
        feat_loss_type=str(cfg.get("feat_loss_type", "cos")).lower(),
        feat_ncc_win=int(cfg.get("feat_ncc_win", 9)),
        feat_level_weights=cfg.get("feat_level_weights", [1.0, 1.0, 1.0, 1.0]),
        device=device,
        log_dir=log_dir,
        amp=cfg.get("amp", cfg.get("use_amp", True)),
        val_every=val_every,
        save_every=save_every,
        max_steps=max_steps,
        lr_scheduler=scheduler,
        select_metric="dice",
    )

    if args.init_checkpoint:
        from unireg.engine.checkpoint import load_checkpoint
        _ = load_checkpoint(args.init_checkpoint, model, optimizer=None, device=device)
        print(f"[train.py] Initialized model weights from {args.init_checkpoint}", flush=True)

    if args.resume:
        trainer.resume(args.resume)

    trainer.run(train_loader, val_loader)

    print("\n[train.py] Training complete. Evaluating test set...", flush=True)
    best_ckpt = Path(log_dir) / "best.pth"
    if best_ckpt.exists():
        from unireg.engine.checkpoint import load_checkpoint
        _ = load_checkpoint(str(best_ckpt), model, optimizer=None, device=device)
        print(f"[train.py] Loaded best checkpoint for test: {best_ckpt}", flush=True)
    else:
        print(f"[train.py] best.pth not found in {log_dir}; using current in-memory model.", flush=True)

    test_loader = build_test_loader(cfg)
    from unireg.engine.evaluator import Evaluator
    evaluator = Evaluator(model, compute_size=tuple(cfg["compute_size"]), device=device, log_dir=log_dir)
    results = evaluator.run(test_loader)

    if results:
        csv_path = os.path.join(log_dir, "test_results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"[train.py] Final test results saved to {csv_path}", flush=True)


if __name__ == "__main__":
    main()
