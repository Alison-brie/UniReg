"""UniReg evaluation script with optional forward-runtime reporting."""
# conda activate samreg4090 && cd code/3_model/

import argparse
import os
import time
import yaml
import torch
from torch.utils.data import DataLoader




def _expand_env(obj):
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expanduser(os.path.expandvars(obj))
    return obj

def parse_args():
    p = argparse.ArgumentParser("UniReg Evaluation")
    p.add_argument("--config",     required=True, help="YAML config")
    p.add_argument("--checkpoint", required=True, help="Model checkpoint .pth")
    p.add_argument("--mode",       default="eval",
                   choices=["infer", "eval", "benchmark"],
                   help="'infer' | 'eval' | 'benchmark'")
    p.add_argument("--device",     default=None, help="Device: cuda, cuda:0, cuda:1, cpu")
    p.add_argument("--gpu",        type=int, default=None, help="GPU id (e.g. 0, 1). Overrides --device to cuda:N")
    p.add_argument("--output_dir", default=None, help="Where to save outputs (infer mode)")
    p.add_argument("--eval_in_raw", type=lambda x: x.lower() in ("1", "true", "yes"),
                   default=None, metavar="BOOL",
                   help="Override eval_in_raw in config.")
    p.add_argument("--disable_timing", action="store_true",
                   help="Disable model-forward runtime measurement.")
    p.add_argument("--runtime_drop_first", type=int, default=0,
                   help="Number of first timed forwards to exclude from the main runtime summary. Default: 0, i.e., include all forwards.")
    return p.parse_args()


def build_test_loader(cfg: dict):
    """Build test loader. Reuses train.py helpers for multi_task or task-tagged single-task eval."""
    if str(cfg.get("dataset", "")).lower() == "multi_task":
        from train import build_test_loader as _build_multi_test_loader
        return _build_multi_test_loader(cfg)

    from unireg.data.dataset import build_dataset
    from train import TaggedTaskDataset

    dataset_type = cfg.get("test_dataset", cfg.get("dataset", "chest_unified")).lower()
    if dataset_type == "dir_qa":
        dataset_type = "dirqa"

    compute_size = tuple(cfg["compute_size"])
    norm_str = cfg.get("norm_strategy", None)
    img_dir = cfg.get("test_root", None)
    seg_dir = cfg.get("test_seg_root", None)
    eval_in_raw = bool(cfg.get("eval_in_raw", False))

    ds = build_dataset(
        dataset=dataset_type,
        split="test",
        compute_size=compute_size,
        norm=norm_str,
        img_dir=img_dir,
        seg_dir=seg_dir,
        pair_json=cfg.get("pair_json", None),
        use_pre_align=bool(cfg.get("use_pre_align", False)),
        pair_mode=cfg.get("test_pair_mode", None),
        seed=cfg.get("seed", 42),
        return_native=eval_in_raw,
    )

    # Pass task-conditioning ids to Evaluator for UniReg checkpoints.
    if "task_type_id" in cfg or "task_id_id" in cfg:
        ds = TaggedTaskDataset(ds, {
            "name": cfg.get("task_id", cfg.get("dataset", "task")),
            "task_type_id": cfg.get("task_type_id", 0),
            "task_id_id": cfg.get("task_id_id", 0),
            "use_pre_align": bool(cfg.get("use_pre_align", False)),
        })

    return DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
    )


class TimingModel(torch.nn.Module):
    """A transparent wrapper that measures pure model forward time.

    The reported time excludes Dice/TRE/Jacobian computation, NIfTI/NumPy saving,
    CSV writing, and other post-processing. CUDA synchronization is used so the
    timing reflects actual GPU execution rather than asynchronous kernel launch.
    """

    def __init__(self, model: torch.nn.Module, device: str = "cuda"):
        super().__init__()
        self.model = model
        self.device = str(device)
        self.forward_times = []

    def forward(self, *args, **kwargs):
        use_cuda_timing = self.device.startswith("cuda") and torch.cuda.is_available()
        if use_cuda_timing:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.model(*args, **kwargs)
        if use_cuda_timing:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        self.forward_times.append(t1 - t0)
        return out

    def runtime_summary(self, drop_first: int = 0):
        times = list(self.forward_times)
        if not times:
            return None

        import numpy as _np
        all_arr = _np.asarray(times, dtype=float)
        if len(times) > drop_first:
            main_arr = _np.asarray(times[drop_first:], dtype=float)
        else:
            main_arr = all_arr

        return {
            "n_all": int(len(all_arr)),
            "n_used": int(len(main_arr)),
            "drop_first": int(drop_first if len(times) > drop_first else 0),
            "mean": float(main_arr.mean()),
            "std": float(main_arr.std()),
            "median": float(_np.median(main_arr)),
            "min": float(main_arr.min()),
            "max": float(main_arr.max()),
            "mean_all": float(all_arr.mean()),
            "std_all": float(all_arr.std()),
        }


def print_runtime_summary(timing_model: TimingModel, drop_first: int = 0):
    summary = timing_model.runtime_summary(drop_first=drop_first)
    if summary is None:
        print("[Runtime] No model forward call was timed.", flush=True)
        return

    print("=" * 50, flush=True)
    print("RUNTIME SUMMARY", flush=True)
    print("  Scope   : pure model forward only", flush=True)
    print(f"  Used    : n={summary['n_used']} / all={summary['n_all']} "
          f"(drop_first={summary['drop_first']})", flush=True)
    print(f"  Mean    : {summary['mean']:.4f} s/pair", flush=True)
    print(f"  Std     : {summary['std']:.4f} s", flush=True)
    print(f"  Median  : {summary['median']:.4f} s", flush=True)
    print(f"  Min/Max : {summary['min']:.4f} / {summary['max']:.4f} s", flush=True)
    print(f"  All mean: {summary['mean_all']:.4f} ± {summary['std_all']:.4f} s", flush=True)
    print("=" * 50, flush=True)


def main():
    args   = parse_args()
    with open(args.config) as f:
        cfg = _expand_env(yaml.safe_load(f))

    if args.gpu is not None:
        device = f"cuda:{args.gpu}"
    else:
        device = args.device or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate.py] device={device}", flush=True)

    # ---- Build model ----
    from unireg.models.registry import build_model
    model = build_model(cfg, device=device)

    # ---- Load checkpoint ----
    from unireg.engine.checkpoint import load_checkpoint
    load_checkpoint(args.checkpoint, model, device=device)

    model.eval()

    # ---- Runtime timing wrapper ----
    # Wrapped after checkpoint loading so state_dict loading remains unchanged.
    timing_model = None
    if not args.disable_timing:
        timing_model = TimingModel(model, device=device)
        model = timing_model
        model.eval()

    # ---- eval_in_raw override (CLI > YAML) ----
    if args.eval_in_raw is not None:
        cfg["eval_in_raw"] = args.eval_in_raw

    # ---- Inference / Evaluation ----
    test_loader = build_test_loader(cfg)

    log_dir = args.output_dir or cfg.get("output_dir", "./eval_output")
    os.makedirs(log_dir, exist_ok=True)

    from unireg.engine.evaluator import Evaluator
    evaluator = Evaluator(
        model        = model,
        compute_size = tuple(cfg["compute_size"]),
        device       = device,
        log_dir      = log_dir,
        iter_cfg     = cfg,
    )

    if args.mode in ("eval", "benchmark"):
        results = evaluator.run(test_loader)
        # Save CSV summary
        import csv, json
        csv_path = os.path.join(log_dir, "results.csv")
        if results:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(results[0].keys()),
                                        extrasaction="ignore")
                writer.writeheader()
                writer.writerows(results)
        print(f"Results saved to {csv_path}", flush=True)
        if timing_model is not None:
            print_runtime_summary(timing_model, drop_first=args.runtime_drop_first)

    else:  # infer
        print("[evaluate.py] Infer mode: saving warped images + flows...", flush=True)
        from unireg.core.spaces import compute_to_raw
        from unireg.data.loaders import save_nifti
        import numpy as np

        with torch.no_grad():
            for idx, batch in enumerate(test_loader):
                moving = batch["moving"].to(device, dtype=torch.float32)
                fixed  = batch["fixed"].to(device, dtype=torch.float32)
                task_type = batch.get("task_type_id", None)
                task_id = batch.get("task_id_id", None)
                if task_type is not None:
                    task_type = task_type.to(device).view(-1).long()
                if task_id is not None:
                    task_id = task_id.to(device).view(-1).long()
                try:
                    warped, flow = model(moving, fixed, task_type=task_type, task_id=task_id)
                except TypeError:
                    warped, flow = model(moving, fixed)
                meta = batch.get("meta", {})
                if isinstance(meta, dict) and "moving_id" in meta:
                    cid = str(meta.get("moving_id", idx))
                elif isinstance(meta, dict) and "moving" in meta and isinstance(meta["moving"], dict):
                    cid = str(meta["moving"].get("case_id", idx))
                else:
                    cid = str(idx)
                prefix = os.path.join(log_dir, cid)
                warped_np = warped.squeeze().cpu().numpy()
                np.save(prefix + "_warped.npy", warped_np)
                np.save(prefix + "_flow.npy", flow.squeeze().cpu().numpy())
                print(f"  Saved {prefix}_warped.npy", flush=True)

        if timing_model is not None:
            print_runtime_summary(timing_model, drop_first=args.runtime_drop_first)


if __name__ == "__main__":
    main()
