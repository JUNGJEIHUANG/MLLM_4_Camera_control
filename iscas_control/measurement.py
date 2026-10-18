"""GLS local background and delta-method centroid/photometry uncertainty.

Noise inputs are temporal variances from repeated REAL frames. Spatial texture
in a ring is never substituted for temporal read/shot noise. These first-order
uncertainties require empirical validation; they are not a CRLB guarantee.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    flux: float
    flux_sigma: float
    centroid_sigma: float
    snr: float
    clipped: bool
    valid: bool
    background: float
    spatial_ring_mad: float

    def to_dict(self):
        return asdict(self)


class PatchEstimator:
    def __init__(self, seeds, radius=10, signal_radius=6, polarity=-1):
        if polarity not in (-1, 1):
            raise ValueError("polarity -1=dark deficit, +1=bright excess")
        self.seeds = tuple(tuple(map(int, p)) for p in seeds)
        self.radius = int(radius)
        self.polarity = polarity
        yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        self.xx, self.yy = xx.astype(float), yy.astype(float)
        rr = xx ** 2 + yy ** 2
        self.signal = rr <= signal_radius ** 2
        self.ring = (rr >= 8 ** 2) & (rr <= radius ** 2)
        self.design = np.stack([np.ones_like(xx), xx, yy], axis=-1).astype(float)

    def patches(self, image):
        a = np.asarray(image)
        if a.ndim != 2:
            raise ValueError("Image must be 2D")
        r = self.radius
        out = []
        for x, y in self.seeds:
            if x - r < 0 or y - r < 0 or x + r >= a.shape[1] or y + r >= a.shape[0]:
                raise ValueError("Point patch outside image")
            out.append(a[y-r:y+r+1, x-r:x+r+1].astype(float))
        return np.stack(out)

    def point(self, patch, temporal_variance, point_id):
        p = np.asarray(patch, float)
        v = np.asarray(temporal_variance, float)
        if p.shape != self.xx.shape or v.shape != p.shape:
            raise ValueError("Patch/temporal-variance shape mismatch")
        if not np.isfinite(p).all() or not np.isfinite(v).all() or (v <= 0).any():
            raise ValueError("Finite data and positive temporal variance required")
        xr = self.design[self.ring]
        vr = v[self.ring]
        normal = xr.T @ (xr / vr[:, None])
        cov_beta = np.linalg.inv(normal)
        beta = cov_beta @ (xr.T @ (p[self.ring] / vr))
        bg = self.design @ beta
        signed = self.polarity * (p - bg)
        # Photometry is SIGNED; no positivity clipping / rectification bias.
        flux = float(signed[self.signal].sum())
        plane_sum = self.design[self.signal].sum(axis=0)
        flux_var = float(v[self.signal].sum() + plane_sum @ cov_beta @ plane_sum)
        snr = flux / np.sqrt(flux_var)
        active = self.signal & (signed > 0)
        weights = np.where(active, signed, 0.)
        mass = float(weights.sum())
        if mass <= 1e-12:
            dx = dy = 0.
            centroid_var = 1e24  # finite sentinel; point remains invalid
        else:
            dx = float((weights * self.xx).sum() / mass)
            dy = float((weights * self.yy).sum() / mass)
            gx = (self.xx[active] - dx) / mass
            gy = (self.yy[active] - dy) / mass
            sx = gx @ self.design[active]
            sy = gy @ self.design[active]
            centroid_var = float((gx**2 + gy**2) @ v[active]
                                 + sx @ cov_beta @ sx + sy @ cov_beta @ sy)
        x, y = self.seeds[point_id]
        clipped = bool((p >= 4095).any() or (p < 0).any())
        valid = bool(snr >= 5 and dx**2 + dy**2 <= 16 and not clipped)
        residual = p[self.ring] - bg[self.ring]
        return Point(x + dx, y + dy, flux, float(np.sqrt(flux_var)),
                     float(np.sqrt(centroid_var)), float(snr), clipped, valid,
                     float(beta[0]), float(1.4826 * np.median(abs(residual - np.median(residual)))))

    def measure(self, patches, variance):
        if len(patches) != len(self.seeds) or len(variance) != len(self.seeds):
            raise ValueError("Need one patch and noise map per seed")
        return [self.point(p, v, i) for i, (p, v) in enumerate(zip(patches, variance))]


def temporal_variance(patches):
    """Shrink noisy per-pixel variance toward a per-patch temporal median.

    patches shape = (repeats, targets, height, width). No spatial-MAD estimate.
    Global illumination drift is NOT removed; it contributes conservatively.
    """
    x = np.asarray(patches, float)
    if x.ndim != 4 or len(x) < 3:
        raise ValueError("At least three repeated patch stacks required")
    v = np.var(x, axis=0, ddof=1)
    med = np.median(v, axis=(-2, -1), keepdims=True)
    return np.maximum(.5 * v + .5 * med, 1 / 12)
