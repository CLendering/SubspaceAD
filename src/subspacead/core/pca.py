import logging
import torch
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler


class KernelPCAModel:
    """Wraps sklearn.decomposition.KernelPCA for feature collection."""

    def __init__(self, k=None, kernel="rbf", gamma=None, eps=1e-6):
        self.k = k
        self.kernel = kernel
        self.gamma = gamma
        self.eps = eps
        self.scaler = None
        self.kpca = None
        self.pca_params = {}

    def fit(self, features: np.ndarray):
        logging.info("Starting Kernel PCA fit...")
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)

        self.kpca = KernelPCA(
            n_components=self.k,
            kernel=self.kernel,
            gamma=self.gamma,
            copy_X=False,
        )

        logging.info(f"Fitting KernelPCA with kernel='{self.kernel}'...")
        self.kpca.fit(features_scaled)
        self.pca_params = {
            "scaler": self.scaler,
            "kpca": self.kpca,
            "k": self.k,
            "eps": self.eps,
        }

        logging.info("Kernel PCA fit complete.")
        return self.pca_params


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"PCAModel will use device: {device}")


class PCAModel:
    """
    Memory-efficient PCA using two-pass or single-pass streaming statistics.
    Based on https://github.com/dnhkng/PCAonGPU
    """

    def __init__(self, k=None, ev=None, whiten=False, eps=1e-6, algorithm="two_pass"):
        if algorithm not in ("two_pass", "single_pass"):
            raise ValueError("algorithm must be 'two_pass' or 'single_pass'")
        self.k = k
        self.ev_ratio = ev
        self.whiten = whiten
        self.eps = eps
        self.mu_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.eigvals_ = None
        self.pca_params = {}
        self.device = device
        self.dtype = torch.float64
        self.algorithm = algorithm

    def _compute_mean(self, feature_generator, feature_dim, total_tokens, num_batches):
        """Pass 1: Compute the mean of all features."""
        logging.info("Starting PCA Pass 1/2 (Mean)...")
        self.mu_ = torch.zeros(feature_dim, dtype=self.dtype, device=self.device)
        for batch in tqdm(
            feature_generator(), total=num_batches, desc="PCA Pass 1/2 (Mean)"
        ):
            batch_gpu = torch.from_numpy(batch).to(self.device, dtype=self.dtype)
            self.mu_ += torch.sum(batch_gpu, axis=0)
        self.mu_ /= total_tokens

    def _compute_covariance(
        self, feature_generator, feature_dim, total_tokens, num_batches
    ):
        """Pass 2: Compute the covariance matrix."""
        logging.info("Starting PCA Pass 2/2 (Covariance)...")
        cov_matrix = torch.zeros(
            (feature_dim, feature_dim), dtype=self.dtype, device=self.device
        )
        for batch in tqdm(
            feature_generator(), total=num_batches, desc="PCA Pass 2/2 (Cov)"
        ):
            batch_gpu = torch.from_numpy(batch).to(self.device, dtype=self.dtype)
            batch_centered = batch_gpu - self.mu_
            cov_matrix += torch.matmul(batch_centered.T, batch_centered)
        cov_matrix /= total_tokens - 1
        return cov_matrix

    def _compute_mean_and_covariance(
        self, feature_generator, feature_dim, total_tokens, num_batches
    ):
        """Merge centered batch scatters without replaying the feature stream.

        The between-batch mean correction is the Chan--Golub--LeVeque
        pairwise update, avoiding cancellation from E[xx.T] - E[x]E[x].T.
        Count observed tokens: total_tokens can be a pre-mask estimate.
        """
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        mean = torch.zeros(feature_dim, dtype=self.dtype, device=self.device)
        scatter = torch.zeros(
            (feature_dim, feature_dim), dtype=self.dtype, device=self.device
        )
        count = 0
        for batch in tqdm(
            feature_generator(), total=num_batches, desc="PCA Single Pass"
        ):
            if batch.ndim != 2 or batch.shape[1] != feature_dim:
                raise ValueError("Feature batches must have shape (N, feature_dim)")
            batch_count = batch.shape[0]
            if batch_count == 0:
                continue
            values = torch.from_numpy(batch).to(self.device, dtype=self.dtype)
            if not torch.isfinite(values).all():
                raise ValueError("Feature batches must contain only finite values")
            batch_mean = values.mean(dim=0)
            centered = values - batch_mean
            batch_scatter = centered.T @ centered
            if count == 0:
                mean = batch_mean
                scatter = batch_scatter
                count = batch_count
                continue
            combined_count = count + batch_count
            delta = batch_mean - mean
            scatter += batch_scatter
            scatter += torch.outer(delta, delta) * (
                count * batch_count / combined_count
            )
            mean += delta * (batch_count / combined_count)
            count = combined_count

        if count < 2:
            raise ValueError("Single-pass PCA requires at least two observed tokens")
        self.mu_ = mean
        logging.info(
            "Single-pass PCA used %d observed tokens (input estimate: %d).",
            count,
            total_tokens,
        )
        return scatter / (count - 1)

    def _compute_eigendecomposition(self, cov_matrix):
        """Perform eigendecomposition on the covariance matrix."""
        logging.info("Performing eigendecomposition on GPU...")
        evals, evecs = torch.linalg.eigh(cov_matrix)
        sorted_indices = torch.argsort(evals, descending=True)
        self.explained_variance_ = evals[sorted_indices]
        return evecs[:, sorted_indices]

    def _select_k_components(self, evecs):
        """Select the number of components (k) based on 'ev_ratio' or 'k'."""
        if self.ev_ratio is not None and self.k is None:
            cumulative_variance = torch.cumsum(
                self.explained_variance_, dim=0
            ) / torch.sum(self.explained_variance_)
            self.k = (
                torch.searchsorted(
                    cumulative_variance,
                    torch.tensor([self.ev_ratio], dtype=self.dtype, device=self.device),
                ).item()
                + 1
            )
            logging.info(
                f"PCA: selected k={self.k} components to explain {self.ev_ratio * 100:.2f}% of variance."
            )

        if self.k is None:
            self.k = evecs.shape[1]
        else:
            self.k = min(self.k, evecs.shape[1])

        self.components_ = evecs[:, : self.k]
        self.eigvals_ = self.explained_variance_[: self.k]

    def _build_pca_params(self):
        """Copies GPU tensors to a CPU numpy dictionary for pipeline use."""
        self.pca_params = {
            "mu": self.mu_.cpu().numpy().astype(np.float64),
            "components": self.components_.cpu().numpy().astype(np.float64),
            "eigvals": self.eigvals_.cpu().numpy().astype(np.float64),
            "sqrt_eig": np.sqrt(
                self.eigvals_.cpu().numpy().astype(np.float64) + self.eps
            ),
            "k": self.k,
            "whiten": self.whiten,
            "eps": self.eps,
            "cov_Z_inv": np.diag(
                1.0 / (self.eigvals_.cpu().numpy().astype(np.float64) + self.eps)
            ),
        }
        return self.pca_params

    def fit(
        self, feature_generator, feature_dim: int, total_tokens: int, num_batches: int
    ):
        """
        Fit PCA with the selected streaming algorithm (two-pass by default).
        """
        logging.info(f"Starting PCA fit on {self.device}...")

        if self.algorithm == "single_pass":
            cov_matrix = self._compute_mean_and_covariance(
                feature_generator, feature_dim, total_tokens, num_batches
            )
        else:
            self._compute_mean(feature_generator, feature_dim, total_tokens, num_batches)
            cov_matrix = self._compute_covariance(
                feature_generator, feature_dim, total_tokens, num_batches
            )

        evecs = self._compute_eigendecomposition(cov_matrix)

        self._select_k_components(evecs)

        return self._build_pca_params()
