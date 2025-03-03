import numpy as np
import cv2
from .orientation import orientation  # We'll define orientation in a separate file

def kptsOrientation(key, im, is_ori: bool, patch_size: int):
    """
    Python version of kptsOrientation.m

    'key' is 2xN or 3xN from MATLAB. We'll assume key is 2xN or 3xN 
    (like you see at the end, they do kpts = kptsOrientation(key,m1,...)).

    Returns an array shape (3, K) where each col is [x, y, angle].
    """

    if len(im.shape) == 3:
        # We assume 'im' is the M_norm (which is single channel),
        # but if not single channel, convert:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    else:
        gray = im

    # Compute gradient
    if is_ori:
        # MATLAB code used a Sobel-like kernel
        h1 = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float64)
        gx = cv2.filter2D(gray, cv2.CV_64F, h1)
        gy = cv2.filter2D(gray, cv2.CV_64F, h1.T)
        gradientImg = np.sqrt(gx**2 + gy**2)
        # angles in [0,360)
        temp_angle = np.degrees(np.arctan2(gy, gx))
        temp_angle[temp_angle < 0] += 360.0
        gradientAng = temp_angle
    else:
        gradientImg = None
        gradientAng = None

    n = 24
    ORI_PEAK_RATIO = 0.8

    # Output array => up to 6 orientations per key in original code,
    # but they used 6 for something else. Let's keep the logic from the .m
    # We'll accumulate them in a list and convert at the end:
    output = []

    # key shape is (2, N) or (3, N). We'll treat the first 2 rows as x,y 
    # and ignore the 3rd if present, because we re-compute orientation anyway.
    N = key.shape[1]
    half_patch = patch_size // 2

    for i in range(N):
        x = round(key[0, i])
        y = round(key[1, i])

        # Check boundary
        if (y - half_patch < 0) or (y + half_patch >= gray.shape[0]) or \
           (x - half_patch < 0) or (x + half_patch >= gray.shape[1]):
            continue

        if is_ori:
            # orientation(...) returns a list of angles
            angles = orientation(x, y, gradientImg, gradientAng, patch_size, n, ORI_PEAK_RATIO)
            if len(angles) == 0:
                # if none found, we store angle=0
                output.append([x, y, 0.0])
            else:
                for ang in angles:
                    output.append([x, y, ang])
        else:
            output.append([x, y, 0.0])

    # shape => (3, K)
    kpts = np.array(output).T  # shape is (K, 3).T => (3,K)
    return kpts
