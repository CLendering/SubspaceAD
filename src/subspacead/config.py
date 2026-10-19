import argparse


def parse_layer_indices(arg_str: str):
    """Parses a comma-separated string of integers."""
    return [int(x.strip()) for x in arg_str.split(",")]


def parse_grouped_layers(arg_str: str):
    """Parses grouped layer indices from format like '-1,-2:-3,-4'."""
    if not arg_str:
        return []
    return [parse_layer_indices(group) for group in arg_str.split(":")]


def get_args():
    """Parses and returns command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified Anomaly Detection Benchmark Framework"
    )
    data_group = parser.add_argument_group("Dataset Arguments")
    model_group = parser.add_argument_group("Model & Feature Extraction Arguments")
    aug_group = parser.add_argument_group("Augmentation Arguments (for k-shot)")
    pca_group = parser.add_argument_group("Anomaly Detection (PCA) Arguments")
    score_group = parser.add_argument_group("Scoring & Evaluation Arguments")
    mask_group = parser.add_argument_group("Background Removal (Saliency) Arguments")
    specular_group = parser.add_argument_group("Specular Reflection Filter Arguments")
    log_group = parser.add_argument_group("Logistics")

    data_group.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    data_group.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["mvtec_ad", "mvtec_ad2", "visa"],
        help="Name of the dataset to use.",
    )
    data_group.add_argument(
        "--dataset_path", type=str, required=True, help="Root path to the dataset."
    )
    data_group.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="Specify categories to run, e.g., 'bottle screw'. If None, runs all.",
    )
    model_group.add_argument(
        "--model_ckpt",
        type=str,
        default="facebook/dinov2-with-registers-large",
        help="HuggingFace model checkpoint for feature extraction.",
    )
    model_group.add_argument(
        "--image_res", type=int, default=256, help="Image resolution for the model."
    )
    model_group.add_argument(
        "--patch_size",
        type=int,
        default=None,
        help="Size of the square patches. If None, process in full resolution.",
    )
    model_group.add_argument(
        "--patch_overlap",
        type=float,
        default=0.0,
        help="Overlap ratio between patches.",
    )
    model_group.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for feature extraction."
    )
    model_group.add_argument(
        "--k_shot",
        type=int,
        default=None,
        help="Number of 'good' training images to use (k-shot). If None, all are used.",
    )
    model_group.add_argument(
        "--agg_method",
        type=str,
        default="mean",
        choices=["concat", "mean", "group"],
        help="Feature aggregation method across layers.",
    )
    model_group.add_argument(
        "--layers",
        type=str,
        default="-12,-13,-14,-15,-16,-17,-18",
        help="Comma-separated layer indices for 'concat' or 'mean' aggregation.",
    )
    model_group.add_argument(
        "--grouped_layers",
        type=str,
        default=None,
        help="Layer groups for 'group' agg. Format: '-1,-2:-3,-4'.",
    )
    model_group.add_argument(
        "--docrop",
        action="store_true",
        help="Apply center cropping during preprocessing.",
    )
    model_group.add_argument(
        "--use_clahe",
        action="store_true",
        help="Apply CLAHE to the images.",
    )

    aug_group.add_argument(
        "--aug_count",
        type=int,
        default=0,
        help="Number of augmented samples to generate per k-shot image. Only active if --k_shot is set.",
    )
    aug_group.add_argument(
        "--aug_list",
        type=str,
        nargs="+",
        default=["rotate"],
        help="List of augmentations to apply. Choices: hflip, vflip, rotate, color_jitter, affine.",
    )
    aug_group.add_argument(
        "--no_aug_categories",
        type=str,
        nargs="+",
        default=["transistor"],
        help="List of categories for which augmentations should be disabled.",
    )

    pca_group.add_argument(
        "--pca_dim",
        type=int,
        default=None,
        help="Number of principal components to keep. Overrides --pca_ev.",
    )
    pca_group.add_argument(
        "--pca_ev",
        type=float,
        default=0.99,
        help="Explained variance to retain for PCA. Used if --pca_dim is None.",
    )
    pca_group.add_argument(
        "--whiten", action="store_true", help="Apply whitening in PCA."
    )
    pca_group.add_argument(
        "--pca_algorithm",
        choices=["two_pass", "single_pass"],
        default="two_pass",
        help="Streaming statistics algorithm for standard PCA (not Kernel PCA).",
    )
    pca_group.add_argument(
        "--use_kernel_pca",
        action="store_true",
        help="Use Kernel PCA instead of standard PCA.",
    )
    pca_group.add_argument(
        "--kernel_pca_kernel",
        type=str,
        default="rbf",
        choices=["rbf", "linear", "poly", "sigmoid", "cosine"],
        help="Kernel to use for Kernel PCA.",
    )
    pca_group.add_argument(
        "--kernel_pca_gamma",
        type=float,
        default=None,
        help="Gamma for rbf, poly and sigmoid kernels. If None, it's set to 1/n_features.",
    )
    score_group.add_argument(
        "--score_method",
        type=str,
        default="reconstruction",
        choices=["reconstruction", "mahalanobis", "cosine", "euclidean"],
        help="Anomaly scoring method.",
    )
    score_group.add_argument(
        "--drop_k",
        type=int,
        default=0,
        help="Number of initial principal components to drop during reconstruction scoring.",
    )
    score_group.add_argument(
        "--img_score_agg",
        type=str,
        default="mtop1p",
        choices=["max", "mean", "p99", "mtop5", "mtop1p"],
        help="Aggregation for image-level scores from pixel maps.",
    )
    score_group.add_argument(
        "--pro_integration_limit",
        type=float,
        default=0.3,
        help="Integration limit for AU-PRO calculation.",
    )
    mask_group.add_argument(
        "--bg_mask_method",
        type=str,
        default=None,
        choices=[None, "dino_saliency", "pca_normality"],
        help="Method to use for background masking.",
    )
    mask_group.add_argument(
        "--mask_threshold_method",
        type=str,
        default="percentile",
        choices=["percentile", "otsu"],
        help="How to binarize the saliency/normality map.",
    )
    mask_group.add_argument(
        "--percentile_threshold",
        type=float,
        default=0.15,
        help="Percentile threshold (0.0-1.0) for 'percentile' method.",
    )
    mask_group.add_argument(
        "--dino_saliency_layer",
        type=int,
        default=6,
        help="Which transformer layer's attention to use for 'dino_saliency' mask (0-indexed).",
    )
    specular_group.add_argument(
        "--use_specular_filter",
        action="store_true",
        help="Enable the specular reflection filter as a post-processing step.",
    )
    specular_group.add_argument(
        "--specular_tau",
        type=float,
        default=0.6,
        help="Binarization threshold for the specular mask.",
    )
    specular_group.add_argument(
        "--specular_size_threshold_factor",
        type=float,
        default=1.5,
        help="Size threshold factor for filtering specular anomalies.",
    )
    log_group.add_argument(
        "--outdir",
        type=str,
        default="./results_full_shot",
        help="Directory to save results, logs, and visualizations.",
    )
    log_group.add_argument(
        "--vis_count",
        type=int,
        default=0,
        help="Number of anomalous examples to visualize per category.",
    )
    log_group.add_argument(
        "--save_intro_overlays",
        action="store_true",
        help="Save clean overlay images for the introductory figure.",
    )
    log_group.add_argument(
        "--no_log_file",
        action="store_true",
        help="Do not save a log file to the output directory.",
    )
    log_group.add_argument(
        "--debug_limit",
        type=int,
        default=None,
        help="Run in debug mode on a subset of N images.",
    )
    log_group.add_argument(
        "--batched_zero_shot",
        action="store_true",
        help="Run in batched zero-shot mode, fitting PCA on the test set.",
    )

    args = parser.parse_args()
    return args
