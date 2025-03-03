import numpy as np

def mosaic_map(img1: np.ndarray, img2: np.ndarray, d: int):
    """
    Python version of mosaic_map.m
    
    Applies a checkerboard-style blanking to two images using block size d,
    and returns (image1, image2, image1 + image2).
    """
    image1 = img1.copy()
    image2 = img2.copy()

    m1, n1, _ = image1.shape
    m11 = int(np.ceil(m1 / d))
    n11 = int(np.ceil(n1 / d))

    # Checkerboard pattern in image1
    # For loops in Python are typically slower; you could vectorize,
    # but here we show a direct translation:
    for i in range(0, m11, 2):
        for j in range(1, n11, 2):
            row_start = i * d
            row_end   = min(row_start + d, m1)
            col_start = j * d
            col_end   = min(col_start + d, n1)
            image1[row_start:row_end, col_start:col_end, :] = 0

    for i in range(1, m11, 2):
        for j in range(0, n11, 2):
            row_start = i * d
            row_end   = min(row_start + d, m1)
            col_start = j * d
            col_end   = min(col_start + d, n1)
            image1[row_start:row_end, col_start:col_end, :] = 0

    # We keep only the original shape
    image1 = image1[:m1, :n1, :]

    # Now do image2 similarly
    m2, n2, _ = image2.shape
    m22 = int(np.ceil(m2 / d))
    n22 = int(np.ceil(n2 / d))

    for i in range(0, m22, 2):
        for j in range(0, n22, 2):
            row_start = i * d
            row_end   = min(row_start + d, m2)
            col_start = j * d
            col_end   = min(col_start + d, n2)
            image2[row_start:row_end, col_start:col_end, :] = 0

    for i in range(1, m22, 2):
        for j in range(1, n22, 2):
            row_start = i * d
            row_end   = min(row_start + d, m2)
            col_start = j * d
            col_end   = min(col_start + d, n2)
            image2[row_start:row_end, col_start:col_end, :] = 0

    image2 = image2[:m2, :n2, :]

    img3 = image1 + image2
    return image1, image2, img3
