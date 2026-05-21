"""
unireg/engine/logger.py
=====================
Structured metric logging to console + JSONL + CSV.
"""

from __future__ import annotations

import sys
import csv
import json
import os
from datetime import datetime
from typing import Dict, Optional


class MetricLogger:
    """
    Logs training/validation metrics to:
      1. Console (formatted)
      2. JSONL file (one JSON object per step)
      3. CSV file (accumulates all numeric metrics)
    """

    def __init__(
        self,
        log_dir: str,
        prefix: str = "metrics",
    ):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir  = log_dir
        self.prefix   = prefix
        self._jsonl_path = os.path.join(log_dir, f"{prefix}.jsonl")
        self._csv_path   = os.path.join(log_dir, f"{prefix}.csv")
        self._csv_writer: Optional[csv.DictWriter] = None
        self._csv_file  = None
        self._fieldnames = None

    def log(self, step: int, metrics: Dict[str, float], phase: str = "train"):
        """
        Log a metric dict for a given step.

        Args:
            step    : global training step
            metrics : dict of metric_name -> float value
            phase   : 'train' | 'val' | 'eval'
        """
        ts = datetime.now().strftime("%H:%M:%S")
        row = {"step": step, "phase": phase, "time": ts, **metrics}

        # Console
        metric_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                                if isinstance(v, (int, float)))
        print(f"[{ts}] [{phase}] step={step}  {metric_str}", flush=True)

        # JSONL
        with open(self._jsonl_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        # CSV
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
            self._csv_file   = open(self._csv_path, "w", newline="")
            self._csv_writer  = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames,
                                               extrasaction="ignore")
            self._csv_writer.writeheader()
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def close(self):
        if self._csv_file:
            self._csv_file.close()

    def __del__(self):
        self.close()
