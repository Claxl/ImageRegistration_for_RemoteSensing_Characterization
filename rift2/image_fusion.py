import numpy as np
import cv2
from .mosaic_map import mosaic_map

def image_fusion(image_1: np.ndarray,
                 image_2: np.ndarray,
                 solution: np.ndarray):
    """
    Python version of image_fusion.m
    
    We produce a mosaic/fused result of image_1 and image_2, given a 3x3
    homography 'solution'.
    Because the original code used 'maketform' and 'imtransform', 
    we'll do the same with OpenCV's warpPerspective or warpAffine.
    """

    # If we do a projective transform:
    # Note the code uses solution_1 * solution => that basically composes 
    # a translation offset + the given transform.
    # They set solution_1 = [[1,0,N1],[0,1,M1],[0,0,1]], 
    # so that the second image is placed offset in the big mosaic.

    M1, N1 = image_1.shape[:2]
    M2, N2 = image_2.shape[:2]

    # Convert to grayscale if needed
    # The code does some checks about # of channels:
    if (image_1.ndim == 3) and (image_2.ndim == 3):
        pass
    elif (image_1.ndim == 2) and (image_2.ndim == 3):
        image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    elif (image_1.ndim == 3) and (image_2.ndim == 2):
        image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    # else both 1-channel

    # Build solution_1
    solution_1 = np.array([
        [1, 0, N1],
        [0, 1, M1],
        [0, 0, 1]
    ], dtype=np.float64)

    # Compose:
    # final_transform = solution_1 * solution
    final_transform = solution_1 @ solution

    # We'll warp both images into a large canvas: 3*N1 wide, 3*M1 tall, 
    # as the original code does
    big_h = 3*M1
    big_w = 3*N1

    # We'll warp them with warpPerspective
    out1 = cv2.warpPerspective(image_1, solution_1, (big_w, big_h))
    out2 = cv2.warpPerspective(image_2, final_transform, (big_w, big_h))

    # fuse
    fusion_image = np.zeros_like(out1, dtype=np.float64)
    # We do the same approach: if a pixel is nonzero in out1 and out2, average
    # if only in out1 => out1, only in out2 => out2
    mask1 = (out1 > 0)
    mask2 = (out2 > 0)
    both = mask1 & mask2
    only1 = mask1 & (~mask2)
    only2 = mask2 & (~mask1)

    # Python indexing
    fusion_image[both] = (out1[both].astype(np.float64) + out2[both].astype(np.float64)) / 2
    fusion_image[only1] = out1[only1]
    fusion_image[only2] = out2[only2]

    fusion_image = np.clip(fusion_image, 0, 255).astype(np.uint8)
    
    # Create a binary mask of non-black pixels
    non_black_mask = (fusion_image > 0)
    if fusion_image.ndim == 3:
        # For color images, consider a pixel non-black if any channel is > 0
        non_black_mask = np.any(non_black_mask, axis=2)
    
    # Find the bounding box of non-black pixels
    rows = np.any(non_black_mask, axis=1)
    cols = np.any(non_black_mask, axis=0)
    
    if np.any(rows) and np.any(cols):  # Ensure there are non-black pixels
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        # Add a small border (optional)
        border = 10
        y_min = max(0, y_min - border)
        y_max = min(fusion_image.shape[0] - 1, y_max + border)
        x_min = max(0, x_min - border)
        x_max = min(fusion_image.shape[1] - 1, x_max + border)
        
        # Crop the fusion image to remove black borders
        fusion_image = fusion_image[y_min:y_max+1, x_min:x_max+1]
    
    # Also create the mosaic image (keep the original version for this)
    _,_,mosaic_image = mosaic_map(out1, out2, 64)
    
    # Apply the same cropping to the mosaic image
    if np.any(rows) and np.any(cols):
        mosaic_image = mosaic_image[y_min:y_max+1, x_min:x_max+1]
    
    # Return the fused and mosaic images (now cropped)
    return fusion_image, mosaic_image