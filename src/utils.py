"""Small shared utilities: seeding, JSON persistence, timing, logging."""
from __future__ import annotations

import json
import logging
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from src import config


def set_seed(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                              datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def _jsonify(obj):
    """Recursively convert numpy/path types so json.dump never chokes."""
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, np.ndarray):
        return _jsonify(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def save_json(payload: dict, name: str) -> Path:
    """Persist a metrics payload to results/metrics/<name>.json (git-tracked)."""
    path = config.RESULTS / f"{name}.json"
    with open(path, "w") as fh:
        json.dump(_jsonify(payload), fh, indent=2)
    return path


def load_json(name: str) -> dict:
    path = config.RESULTS / f"{name}.json"
    with open(path) as fh:
        return json.load(fh)


def json_exists(name: str) -> bool:
    return (config.RESULTS / f"{name}.json").exists()


@contextmanager
def timer(label: str, logger: logging.Logger | None = None):
    t0 = time.time()
    yield
    msg = f"{label}: {time.time() - t0:,.1f}s"
    (logger.info if logger else print)(msg)
