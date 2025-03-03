import numpy as np
from skimage.morphology import disk

from .calculate_orientation_hist import calculate_orientation_hist

def orientation(x, y,
                gradientImg: np.ndarray,
                gradientAng: np.ndarray,
                patch_size: int,
                n: int,
                ORI_PEAK_RATIO: float):
    """
    Python version of orientation.m

    We replicate the logic of building a circular neighborhood mask Sa,
    computing orientation histogram, and picking peaks that exceed 
    ORI_PEAK_RATIO * maximum.
    """

    # In MATLAB, se=strel('disk', patch_size/2,0); 
    # we do similarly with scikit-image:
    r = patch_size // 2
    se = disk(r)  # 2D boolean array with a disk shape
    Sa = se.astype(np.float64)

    # This function calls calculate_orientation_hist => we replicate that
    hist_, max_value = calculate_orientation_hist(x, y, r, gradientImg, gradientAng, n, Sa)

    mag_thr = max_value * ORI_PEAK_RATIO
    ANG = []
    for k in range(n):
        k1 = (k-1) % n
        k2 = (k+1) % n
        if (hist_[k] > hist_[k1]) and (hist_[k] > hist_[k2]) and (hist_[k] > mag_thr):
            # interpolate
            denom = (hist_[k1] + hist_[k2] - 2*hist_[k])
            if abs(denom) < 1e-12:
                bin_offset = 0.0
            else:
                bin_offset = 0.5 * (hist_[k1] - hist_[k2]) / denom
            bin_val = (k - 1) + bin_offset
            # wrap
            if bin_val < 0:
                bin_val += n
            elif bin_val >= n:
                bin_val -= n
            angle = 360.0 * bin_val / n
            ANG.append(angle)

    return ANG
