import numpy as np
from scipy import ndimage
from skimage.morphology import disk

def calculate_orientation_hist(x, y, radius, gradient, angle, n, Sa):
    """
    Calculate orientation histogram for a keypoint.
    
    Args:
        x, y: Coordinates of the keypoint
        radius: Radius of the region around the keypoint
        gradient: Gradient magnitude image
        angle: Gradient angle image (in degrees)
        n: Number of histogram bins
        Sa: Circular mask
    
    Returns:
        hist: Orientation histogram
        max_value: Maximum value in the histogram
    """
    sigma = radius / 3
    
    # Define region boundaries
    radius_x_left = int(x - radius)
    radius_x_right = int(x + radius)
    radius_y_up = int(y - radius)
    radius_y_down = int(y + radius)
    
    # Extract sub-regions - ensure consistent dimensions
    sub_gradient = gradient[radius_y_up:radius_y_down+1, radius_x_left:radius_x_right+1]
    sub_angle = angle[radius_y_up:radius_y_down+1, radius_x_left:radius_x_right+1]
    
    # Get dimensions of extracted regions
    h, w = sub_gradient.shape
    
    # Create coordinate grids that match the extracted region dimensions
    X = np.arange(-radius, -radius + w)
    Y = np.arange(-radius, -radius + h)
    XX, YY = np.meshgrid(X, Y)
    
    # Create Gaussian weighting
    gaussian_weight = np.exp(-(XX**2 + YY**2) / (2 * sigma**2))
    
    # Weight the gradient magnitudes
    W1 = sub_gradient * gaussian_weight
    W = Sa.astype(float) * W1
    
    # Calculate histogram bins
    bin_indices = np.round(sub_angle * n / 360).astype(int)
    bin_indices[bin_indices >= n] = bin_indices[bin_indices >= n] - n
    bin_indices[bin_indices < 0] = bin_indices[bin_indices < 0] + n
    
    # Compute histogram
    temp_hist = np.zeros(n)
    for i in range(n):
        mask = (bin_indices == i)
        if np.any(mask):
            temp_hist[i] = np.sum(W[mask])
    
    # Smooth histogram
    hist = np.zeros(n)
    
    # First bin
    hist[0] = (temp_hist[n-2] + temp_hist[2]) / 16 + \
              4 * (temp_hist[n-1] + temp_hist[1]) / 16 + \
              temp_hist[0] * 6 / 16
    
    # Second bin
    hist[1] = (temp_hist[n-1] + temp_hist[3]) / 16 + \
              4 * (temp_hist[0] + temp_hist[2]) / 16 + \
              temp_hist[1] * 6 / 16
    
    # Middle bins using vectorized operations
    hist[2:n-2] = (temp_hist[:n-4] + temp_hist[4:]) / 16 + \
                  4 * (temp_hist[1:n-3] + temp_hist[3:n-1]) / 16 + \
                  temp_hist[2:n-2] * 6 / 16
    
    # Second-to-last bin
    hist[n-2] = (temp_hist[n-4] + temp_hist[0]) / 16 + \
                4 * (temp_hist[n-3] + temp_hist[n-1]) / 16 + \
                temp_hist[n-2] * 6 / 16
    
    # Last bin
    hist[n-1] = (temp_hist[n-3] + temp_hist[1]) / 16 + \
                4 * (temp_hist[n-2] + temp_hist[0]) / 16 + \
                temp_hist[n-1] * 6 / 16
    
    max_value = np.max(hist)
    return hist, max_value

def orientation(x, y, gradient_img, gradient_ang, patch_size, n, ORI_PEAK_RATIO):
    """
    Compute orientation angle(s) for a keypoint.
    
    Args:
        x, y: Coordinates of the keypoint
        gradient_img: Gradient magnitude image
        gradient_ang: Gradient angle image (in degrees)
        patch_size: Size of the patch around the keypoint
        n: Number of histogram bins
        ORI_PEAK_RATIO: Peak ratio threshold
    
    Returns:
        ANG: List of orientation angles (in degrees)
    """
    # Create circular mask with the same radius as the patch
    radius = patch_size // 2
    se = disk(radius)
    
    # Calculate orientation histogram
    hist, max_value = calculate_orientation_hist(x, y, patch_size // 2, gradient_img, gradient_ang, n, se)
    
    # Find peaks in the histogram
    mag_thr = max_value * ORI_PEAK_RATIO
    ANG = []
    
    for k in range(n):
        # Handle circular indexing
        k1 = n - 1 if k == 0 else k - 1
        k2 = 0 if k == n - 1 else k + 1
        
        # Check if bin k is a peak
        if hist[k] > hist[k1] and hist[k] > hist[k2] and hist[k] > mag_thr:
            # Refine peak position with parabola fitting
            bin_index = k + 0.5 * (hist[k1] - hist[k2]) / (hist[k1] + hist[k2] - 2 * hist[k])
            
            # Handle circular indexing for refined position
            if bin_index < 0:
                bin_index = n + bin_index
            elif bin_index >= n:
                bin_index = bin_index - n
            
            # Convert bin index to angle (in degrees)
            angle = (360 / n) * bin_index
            ANG.append(angle)
    
    return ANG

def kpts_orientation(key, im, is_ori, patch_size):
    """
    Compute keypoint orientations.
    
    Args:
        key: Input keypoints (2 x N array, where N is the number of keypoints)
        im: Input image
        is_ori: Flag to compute orientation (1) or not (0)
        patch_size: Size of the patch around each keypoint
    
    Returns:
        kpts: Output keypoints with orientation (3 x M array, where M is the number of keypoints)
    """
    if is_ori == 1:
        n = 24
        ORI_PEAK_RATIO = 0.8
        
        # Create Sobel filters
        h = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        
        # Compute gradients
        gradient_x = ndimage.convolve(im, h, mode='reflect')
        gradient_y = ndimage.convolve(im, h.T, mode='reflect')
        
        # Compute gradient magnitude and angle
        gradient_img = np.sqrt(gradient_x**2 + gradient_y**2)
        temp_angle = np.arctan2(gradient_y, gradient_x) * 180 / np.pi
        temp_angle[temp_angle < 0] += 360
        gradient_ang = temp_angle
    
    feat_index = 0
    num_keypoints = key.shape[1]
    kpts = np.zeros((3, num_keypoints * 6))
    
    for k in range(num_keypoints):
        x = int(round(key[0, k]))
        y = int(round(key[1, k]))
        r = int(round(patch_size))
        
        x1 = max(1, x - r // 2)
        y1 = max(1, y - r // 2)
        x2 = min(x + r // 2, im.shape[1])
        y2 = min(y + r // 2, im.shape[0])
        
        if y2 - y1 != r or x2 - x1 != r:
            continue
        
        if is_ori == 1:
            angles = orientation(x, y, gradient_img, gradient_ang, r, n, ORI_PEAK_RATIO)
            for angle in angles:
                kpts[:, feat_index] = [x, y, angle]
                feat_index += 1
        else:
            kpts[:, feat_index] = [x, y, 0]
            feat_index += 1
    
    # Remove unused entries
    kpts = kpts[:, :feat_index]
    return kpts