import numpy as np
import cv2

def extract_patches(img: np.ndarray, x: float, y: float, s: int, t_deg: float):
    """
    Python version of extract_patches.m

    Extracts a patch of size (2*s + 1) x (2*s + 1) around (x,y) from img, 
    rotated by t_deg degrees. Bilinear interpolation is done via OpenCV.
    """
    # Convert to float64 if needed
    img = img.astype(np.float64)
    h, w = img.shape[:2]
    # If grayscale, treat as single channel
    if len(img.shape) == 2:
        m = 1
    else:
        m = img.shape[2]

    # Clip x,y
    x = max(1, min(w, round(x)))
    y = max(1, min(h, round(y)))
    s = int(round(s))

    # Build the rotation around (x,y)
    # OpenCV wants center=(cx,cy), angle in degrees, scale=1
    M = cv2.getRotationMatrix2D((x, y), t_deg, 1.0)

    patch_size = 2 * s + 1

    # We'll warp just a bounding box around the patch,
    # so let's define a region-of-interest bounding box
    # around (x,y) that fully contains the patch:
    # i.e. from (x-s, y-s) to (x+s, y+s).

    # Because we do partial transformations, let's create 
    # a bigger region and warp just that portion. 
    # Alternatively, we can do the direct approach from the original code.

    # We'll create a new 'canvas' region
    x0 = int(x - s)
    y0 = int(y - s)
    x1 = x0 + patch_size
    y1 = y0 + patch_size

    # Clip to image boundaries
    x0_cl = max(0, x0);  y0_cl = max(0, y0)
    x1_cl = min(w, x1);  y1_cl = min(h, y1)
    # Crop out that region
    roi = img[y0_cl:y1_cl, x0_cl:x1_cl] if m == 1 else img[y0_cl:y1_cl, x0_cl:x1_cl, :]

    # Warp that smaller chunk in place
    # However, for simplicity, we can warp the entire image:
    # Then we crop out the patch around (x,y).
    # This might be simpler to illustrate, though it's less efficient:
    warped = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)

    # Now extract the patch from warped
    px0 = x - s
    py0 = y - s
    px1 = px0 + patch_size
    py1 = py0 + patch_size

    # integer boundaries
    px0_i = int(np.floor(px0)); py0_i = int(np.floor(py0))
    px1_i = px0_i + patch_size
    py1_i = py0_i + patch_size

    # clamp
    px0_i = max(0, px0_i); py0_i = max(0, py0_i)
    px1_i = min(w, px1_i); py1_i = min(h, py1_i)

    # build patch
    if m == 1:
        out_patch = np.zeros((patch_size, patch_size), dtype=np.float64)
        sub = warped[py0_i:py1_i, px0_i:px1_i]
        out_patch[0:sub.shape[0], 0:sub.shape[1]] = sub
    else:
        out_patch = np.zeros((patch_size, patch_size, m), dtype=np.float64)
        sub = warped[py0_i:py1_i, px0_i:px1_i, :]
        out_patch[0:sub.shape[0], 0:sub.shape[1], :] = sub

    return out_patch
