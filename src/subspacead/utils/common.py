import argparse
import json
import logging
import os
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve


def setup_logging(outdir: str, save_log: bool = True):
    """Configures the logging for console and file output."""
    log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)

    root_logger = logging.getLogger()
    # Avoid adding handlers multiple times if called elsewhere
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)

    # File handler
    if save_log:
        log_file = os.path.join(outdir, "run.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        root_logger.addHandler(file_handler)

    logging.info("Logging configured.")


def save_config(args: argparse.Namespace):
    """Saves the run configuration to a JSON file."""
    config_path = os.path.join(args.outdir, "config.json")
    try:
        with open(config_path, "w") as f:
            json.dump(vars(args), f, indent=4)
        logging.info(f"Configuration saved to {config_path}")
    except TypeError as e:
        logging.warning(
            f"Could not save config as JSON: {e}. Some args may not be serializable."
        )


def min_max_norm(
    x: torch.Tensor | np.ndarray, eps: float = 1e-8
) -> torch.Tensor | np.ndarray:
    """Performs min-max normalization on a tensor or numpy array."""
    is_torch = torch.is_tensor(x)

    if is_torch:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x_min = torch.amin(x, dim=(-1, -2), keepdim=True)
        x_max = torch.amax(x, dim=(-1, -2), keepdim=True)
        x_norm = (x - x_min) / (x_max - x_min + eps)
        return x_norm.clamp(0.0, 1.0)
    else:
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x_min = np.min(x, axis=(-1, -2), keepdims=True)
        x_max = np.max(x, axis=(-1, -2), keepdims=True)
        x_norm = (x - x_min) / (x_max - x_min + eps)
        return np.clip(x_norm, 0.0, 1.0)


def _best_f1_threshold_from_scores(y_true, y_score):
    """Return threshold maximizing F1 on validation scores."""
    y_true = np.asarray(y_true).astype(np.uint8)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.size == 0 or y_score.size == 0 or (y_true.max() == y_true.min()):
        return None, 0.0

    p, r, t = precision_recall_curve(y_true, y_score)
    if t.size == 0:
        return None, 0.0

    f1 = (2 * p[:-1] * r[:-1]) / np.clip(p[:-1] + r[:-1], 1e-12, None)
    i = int(np.nanargmax(f1))
    return float(t[i]), float(f1[i])


def _quantile_threshold_from_negatives(y_true, y_score, target_fpr=0.01):
    """Fallback: pick threshold so that ~target_fpr of NEGATIVES exceed it."""
    y_true = np.asarray(y_true).astype(np.uint8)
    y_score = np.asarray(y_score, dtype=np.float64)
    neg = y_score[y_true == 0]
    if neg.size == 0:
        return None
    q = np.clip(1.0 - float(target_fpr), 0.0, 1.0)
    return float(np.quantile(neg, q, interpolation="linear"))


def pick_threshold_with_fallback(y_true, y_score, target_fpr):
    """
    Try PR-optimal F1; if degenerate (single-class), fall back to negative-quantile.
    Returns (thr, how), where how ∈ {"pr", "quantile", "none"}.
    """
    thr_pr, _ = _best_f1_threshold_from_scores(y_true, y_score)
    if thr_pr is not None:
        return thr_pr, "pr"

    thr_q = _quantile_threshold_from_negatives(y_true, y_score, target_fpr)
    if thr_q is not None:
        return thr_q, "quantile"

    return None, "none"


def topk_mean(arr, frac=0.01):
    """Computes the mean of the top k% values in a flattened array."""
    flat = arr.ravel()
    k = max(1, int(len(flat) * frac))
    idx = np.argpartition(flat, -k)[-k:]
    return float(np.mean(flat[idx]))


def generate_run_name(args: argparse.Namespace) -> str:
    """Generates a unique run name from the command-line arguments."""
    # Core params
    run_name = f"{args.dataset_name}_{args.agg_method}"
    run_name += f"_layers{''.join(args.layers.split(','))}"
    run_name += f"_res{args.image_res}_docrop{int(args.docrop)}"

    # Optional params
    if args.patch_size:
        run_name += f"_patch{args.patch_size}"
    if args.use_kernel_pca:
        run_name += f"_kpca-{args.kernel_pca_kernel}"
    if args.use_specular_filter:
        run_name += "_spec-filt"
    if args.bg_mask_method:
        run_name += f"_mask-{args.bg_mask_method}_thr-{args.mask_threshold_method}"
        if args.mask_threshold_method == "percentile":
            run_name += f"{args.percentile_threshold}"
        if args.bg_mask_method == "dino_saliency":
            run_name += f"_L{args.dino_saliency_layer}"

    run_name += f"_score-{args.score_method}"
    run_name += f"_clahe{int(args.use_clahe)}"
    run_name += f"_dropk{args.drop_k}"
    run_name += f"_model-{args.model_ckpt.split('/')[-1]}"

    # PCA params
    pca_str = (
        f"pca_ev{args.pca_ev}" if args.pca_ev is not None else f"_pca_dim{args.pca_dim}"
    )
    run_name += f"_{pca_str}"
    if (
        not args.use_kernel_pca
        and getattr(args, "pca_algorithm", "two_pass") == "single_pass"
    ):
        run_name += "_single-pass"
    run_name += f"_i-score{args.img_score_agg}"

    # K-shot params
    if args.k_shot is not None:
        run_name += f"_k{args.k_shot}"
        if args.aug_count > 0 and args.aug_list:
            aug_str = "".join(sorted([a[0] for a in args.aug_list]))
            run_name += f"_aug{args.aug_count}x{aug_str}"

    if args.save_intro_overlays:
        run_name += f"_intro-overlays"

    return run_name
