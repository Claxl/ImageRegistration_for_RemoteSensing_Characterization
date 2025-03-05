import numpy as np
import cv2

def simplified_transform_estimation(kp1, kp2, transform_type='similarity', ransac_threshold=3.0):
    """
    Estimate transformation between two sets of matching points using OpenCV.
    
    Args:
        kp1 (np.ndarray): Nx2 array of matched keypoint coordinates in image1.
        kp2 (np.ndarray): Nx2 array of matched keypoint coordinates in image2.
        transform_type (str): Type of transformation to estimate: 
                             'similarity', 'affine', or 'perspective'.
        ransac_threshold (float): Maximum allowed reprojection error in RANSAC.
        
    Returns:
        H (np.ndarray): 3x3 transformation matrix.
        mask (np.ndarray): Boolean mask indicating inliers.
        inliers_kp1 (np.ndarray): Coordinates of inliers in image1.
        inliers_kp2 (np.ndarray): Coordinates of inliers in image2.
    """
    if transform_type == 'similarity':
        # similarity transform is a subset of affine
        method = cv2.RANSAC
        H, mask = cv2.estimateAffinePartial2D(
            kp2, kp1,  # Note the order: source points, destination points
            method=method,
            ransacReprojThreshold=ransac_threshold
        )
        # Convert to 3x3 form
        H_full = np.eye(3)
        if H is not None:
            H_full[0:2, :] = H
        
    elif transform_type == 'affine':
        method = cv2.RANSAC
        H, mask = cv2.estimateAffine2D(
            kp2, kp1,  # Note the order: source points, destination points
            method=method,
            ransacReprojThreshold=ransac_threshold
        )
        # Convert to 3x3 form
        H_full = np.eye(3)
        if H is not None:
            H_full[0:2, :] = H
            
    elif transform_type == 'perspective':
        method = cv2.RANSAC
        H_full, mask = cv2.findHomography(
            kp2, kp1,  # Note the order: source points, destination points
            method=method,
            ransacReprojThreshold=ransac_threshold
        )
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")
    
    # If transformation estimation failed, return identity
    if H_full is None:
        H_full = np.eye(3)
        mask = np.zeros(len(kp1), dtype=np.uint8)
    
    # Extract inlier points
    if mask is not None:
        inlier_idxs = np.where(mask.ravel() == 1)[0]
        inliers_kp1 = kp1[inlier_idxs]
        inliers_kp2 = kp2[inlier_idxs]
    else:
        inliers_kp1 = np.array([])
        inliers_kp2 = np.array([])
    
    # Calculate RMSE for inliers
    rmse = 0.0
    if len(inliers_kp1) > 0:
        if transform_type == 'perspective':
            # For homography, we need to apply perspective division
            ones = np.ones((len(inliers_kp2), 1))
            pts_homogeneous = np.hstack((inliers_kp2, ones))
            transformed = np.dot(H_full, pts_homogeneous.T).T
            transformed = transformed[:, :2] / transformed[:, 2:]
            errors = transformed - inliers_kp1
        else:
            # For affine transforms
            transformed = cv2.transform(inliers_kp2.reshape(-1, 1, 2), H_full[:2]).reshape(-1, 2)
            errors = transformed - inliers_kp1
        
        rmse = np.sqrt(np.mean(np.sum(errors**2, axis=1)))
    
    return H_full, mask, inliers_kp1, inliers_kp2, rmse


def simplified_image_fusion(image1, image2, H):
    """
    Perform simplified image fusion by warping image2 to image1's space.
    
    Args:
        image1 (np.ndarray): Reference image.
        image2 (np.ndarray): Image to be transformed.
        H (np.ndarray): 3x3 transformation matrix from image2 to image1.
        
    Returns:
        fused (np.ndarray): Fused image.
        mosaic (np.ndarray): Mosaic image with alternating checkerboard pattern.
    """
    # Ensure both images are 3-channel
    if len(image1.shape) == 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    if len(image2.shape) == 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    
    # Get image dimensions
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    # Create a canvas large enough to hold both images
    # We'll use a 3x size of the reference image as default
    output_size = (3*w1, 3*h1)
    
    # Create the offset transformation to center the result
    offset = np.array([[1, 0, w1], [0, 1, h1], [0, 0, 1]], dtype=np.float64)
    
    # Combine transformations
    final_transform = offset @ H
    
    # Warp image2 onto the canvas
    warped1 = cv2.warpPerspective(image1, offset, output_size)
    warped2 = cv2.warpPerspective(image2, final_transform, output_size)
    
    # Create fusion and mosaic images
    fusion = np.zeros_like(warped1, dtype=np.float64)
    
    # Blend where both images exist
    mask1 = (warped1 > 0)
    mask2 = (warped2 > 0)
    both = mask1 & mask2
    only1 = mask1 & (~mask2)
    only2 = mask2 & (~mask1)
    
    fusion[both] = (warped1[both].astype(np.float64) + warped2[both].astype(np.float64)) / 2
    fusion[only1] = warped1[only1]
    fusion[only2] = warped2[only2]
    
    fusion = np.clip(fusion, 0, 255).astype(np.uint8)
    
    # Create a simple checkerboard mosaic
    mosaic = np.zeros_like(fusion)
    block_size = 64
    
    # Create checkerboard masks
    y_blocks = fusion.shape[0] // block_size + 1
    x_blocks = fusion.shape[1] // block_size + 1
    
    for y in range(y_blocks):
        for x in range(x_blocks):
            y1 = y * block_size
            y2 = min((y + 1) * block_size, fusion.shape[0])
            x1 = x * block_size
            x2 = min((x + 1) * block_size, fusion.shape[1])
            
            if (y + x) % 2 == 0:
                mosaic[y1:y2, x1:x2] = warped1[y1:y2, x1:x2]
            else:
                mosaic[y1:y2, x1:x2] = warped2[y1:y2, x1:x2]
    
    # Crop the images to remove unnecessary black borders
    # Find non-zero pixels
    non_zero = np.where(fusion > 0)
    if len(non_zero[0]) > 0 and len(non_zero[1]) > 0:
        y_min, y_max = np.min(non_zero[0]), np.max(non_zero[0])
        x_min, x_max = np.min(non_zero[1]), np.max(non_zero[1])
        
        # Add a small border
        border = 10
        y_min = max(0, y_min - border)
        y_max = min(fusion.shape[0] - 1, y_max + border)
        x_min = max(0, x_min - border)
        x_max = min(fusion.shape[1] - 1, x_max + border)
        
        # Crop both images
        fusion = fusion[y_min:y_max+1, x_min:x_max+1]
        mosaic = mosaic[y_min:y_max+1, x_min:x_max+1]
    
    return fusion, mosaic