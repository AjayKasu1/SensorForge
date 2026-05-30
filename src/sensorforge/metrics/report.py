"""Render a per-run calibration report: markdown + embedded PNG plots.

Phase 4 calls this once per calibration run to produce the human-facing
artifact. Plots use the matplotlib Figure API directly (no pyplot global state
or backend switching), so it is safe to call from a headless run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger
from matplotlib.figure import Figure
from numpy.typing import NDArray

from sensorforge.isp.params import ISPParams
from sensorforge.metrics.emva1288 import PhotonTransfer


def _as_display(img: NDArray) -> NDArray:
    out = img.astype(np.float64)
    if img.dtype != np.uint8:
        out = np.clip(out, 0, 1)
    else:
        out /= 255.0
    return out


def _save_comparison(path: Path, sim: NDArray, real: NDArray) -> None:
    sim_d, real_d = _as_display(sim), _as_display(real)
    diff = np.abs(sim_d - real_d).mean(axis=-1) if sim_d.ndim == 3 else np.abs(sim_d - real_d)

    fig = Figure(figsize=(13, 4))
    axes = fig.subplots(1, 3)
    for ax, img, title in [(axes[0], sim_d, "sim"), (axes[1], real_d, "real")]:
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    im = axes[2].imshow(diff, cmap="magma")
    axes[2].set_title("abs diff")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    fig.savefig(path, dpi=100, bbox_inches="tight")


def _save_photon_transfer(path: Path, pt: PhotonTransfer) -> None:
    fig = Figure(figsize=(6, 4))
    ax = fig.subplots()
    ax.scatter(pt.mean_signal_dn, pt.temporal_variance_dn2, label="measured")
    x = np.linspace(0, pt.mean_signal_dn.max(), 50)
    ax.plot(x, pt.gain_dn_per_e * x + pt.dark_variance_dn2, "r-", label=f"K={pt.gain_dn_per_e:.3f}")
    ax.set_xlabel("mean signal (DN)")
    ax.set_ylabel("temporal variance (DN^2)")
    ax.set_title("Photon transfer")
    ax.legend()
    fig.savefig(path, dpi=100, bbox_inches="tight")


def _metrics_table(metrics: dict[str, float]) -> str:
    rows = "\n".join(f"| {k} | {v:.4g} |" for k, v in metrics.items())
    return f"| metric | value |\n|---|---|\n{rows}"


def write_report(
    out_dir: str | Path,
    sim: NDArray,
    real: NDArray,
    metrics: dict[str, float],
    params: ISPParams,
    photon_transfer: PhotonTransfer | None = None,
) -> Path:
    """Write ``report.md`` (plus PNGs) into ``out_dir`` and return its path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_comparison(out_dir / "comparison.png", sim, real)

    sections = [
        "# Calibration report",
        "## Metrics",
        _metrics_table(metrics),
        "## Sim vs real",
        "![comparison](comparison.png)",
    ]
    if photon_transfer is not None:
        _save_photon_transfer(out_dir / "photon_transfer.png", photon_transfer)
        sections += [
            "## Photon transfer",
            f"System gain K = {photon_transfer.gain_dn_per_e:.4g} DN/e-",
            "![photon transfer](photon_transfer.png)",
        ]
    sections += [
        "## Parameters",
        f"```json\n{params.model_dump_json(indent=2)}\n```",
    ]

    report = out_dir / "report.md"
    report.write_text("\n\n".join(sections) + "\n")
    logger.info("wrote calibration report to {}", report)
    return report
