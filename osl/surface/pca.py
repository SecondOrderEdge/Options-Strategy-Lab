"""PCA of an accumulated implied-volatility surface history.

Following Cont & da Fonseca (2002), PCA is run on the **daily changes** of the
IV surface. The first three eigenmodes correspond to level, slope, and
curvature and explain the bulk of the variance (≈98% on index options).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray = npt.NDArray[np.float64]

FACTOR_NAMES = ("level", "slope", "curvature")


@dataclass(frozen=True)
class SurfacePCA:
    grid: tuple[str, ...]  # surface point labels (columns of the panel)
    eigenvalues: FloatArray
    eigenvectors: FloatArray  # shape (n_points, n_factors); columns are factors
    explained_variance_ratio: FloatArray
    factor_loadings: FloatArray  # shape (n_obs, n_factors); time series of scores
    factor_names: tuple[str, ...]


def surface_pca(iv_panel: pd.DataFrame, *, n_factors: int = 3, on: str = "changes") -> SurfacePCA:
    """PCA of an IV panel (rows = dates, columns = surface points).

    ``on="changes"`` (default) analyzes day-over-day differences; ``"levels"``
    analyzes demeaned levels.
    """
    if on == "changes":
        data = iv_panel.diff().dropna()
    elif on == "levels":
        data = iv_panel - iv_panel.mean()
    else:
        raise ValueError(f"unknown PCA basis: {on!r}")

    x = data.to_numpy(dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    cov = np.cov(x, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    total = float(eigvals.sum())
    ratio = eigvals / total if total > 0 else np.zeros_like(eigvals)

    k = min(n_factors, eigvecs.shape[1])
    vecs = eigvecs[:, :k]
    loadings = x @ vecs
    names = FACTOR_NAMES[:k] if k <= len(FACTOR_NAMES) else tuple(f"pc{i + 1}" for i in range(k))

    return SurfacePCA(
        grid=tuple(str(c) for c in iv_panel.columns),
        eigenvalues=np.asarray(eigvals, dtype=np.float64),
        eigenvectors=np.asarray(vecs, dtype=np.float64),
        explained_variance_ratio=np.asarray(ratio, dtype=np.float64),
        factor_loadings=np.asarray(loadings, dtype=np.float64),
        factor_names=names,
    )
