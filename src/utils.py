import logging
import os
import random

import numpy as np
import torch


def _file_handler_filter(log_record):
    return log_record.levelno == logging.INFO


def configure_logger(save_path: str | None = None) -> None:
    """Configure root logger with console + optional file handlers.

    Idempotent-ish: clears existing handlers to avoid duplicate logs if called
    multiple times in the same process.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    if save_path:
        fh = logging.FileHandler(save_path)
        fh.setLevel(logging.INFO)
        fh.addFilter(_file_handler_filter)
        formatter = logging.Formatter("%(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and Torch for reproducibility.

    If `deterministic=True`, configures cuDNN for deterministic behavior.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    # Safe to call even if CUDA is not available
    torch.cuda.manual_seed(seed)
