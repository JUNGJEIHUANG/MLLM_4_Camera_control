"""One-time DEVELOPMENT ROI alignment, never per-policy/test-result reselection."""
import numpy as np
from scipy.ndimage import gaussian_filter


def align_development_seeds(image, seeds, search_radius=10, coherence_px=3.):
    a = np.asarray(image, float)
    aligned, offsets = [], []
    for x, y in seeds:
        if x-20 < 0 or y-20 < 0 or x+20 >= a.shape[1] or y+20 >= a.shape[0]:
            raise ValueError("Alignment patch out of bounds")
        q = a[y-20:y+21, x-20:x+21]
        contrast = gaussian_filter(q, 8) - gaussian_filter(q, 1)
        z = contrast[20-search_radius:21+search_radius, 20-search_radius:21+search_radius]
        if np.ptp(z) < 1e-8:
            raise ValueError("No alignment structure")
        iy, ix = np.unravel_index(np.argmax(z), z.shape)
        dx, dy = int(ix)-search_radius, int(iy)-search_radius
        if abs(dx) >= search_radius or abs(dy) >= search_radius:
            raise ValueError("Alignment optimum at search boundary")
        offsets.append((dx, dy)); aligned.append((x+dx, y+dy))
    shifts = np.asarray(offsets)
    if np.max(np.linalg.norm(shifts-np.median(shifts, axis=0), axis=1)) > coherence_px:
        raise ValueError("Noncoherent point displacements; manual target check required")
    if len(set(aligned)) != len(aligned):
        raise ValueError("Alignment merged point identities")
    return aligned, {"offsets_xy": offsets, "search_radius": search_radius,
                     "coherence_limit_px": coherence_px, "development_only": True}
