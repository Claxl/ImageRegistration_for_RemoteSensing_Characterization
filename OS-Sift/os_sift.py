#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OS-SIFT: Orientation-Scale SIFT for SAR-Optical Image Registration

This implementation is a Python port of the original MATLAB code,
designed for registering SAR and optical remote sensing images.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import morphology
import os
import time


def build_scale_opt(image, sigma, m_max, ratio, d):
    """
    Build scale space for optical images using a Harris-like function.
    
    Parameters:
    -----------
    image : ndarray
        Input optical image (grayscale)
    sigma : float
        Base scale parameter
    m_max : int
        Number of scale levels
    ratio : float
        Scale ratio between consecutive levels
    d : float
        Harris function parameter
        
    Returns:
    --------
    tuple
        (harris_function, gradient, angle) - scale-space structures
    """
    m, n = image.shape
    harris_function = np.zeros((m, n, m_max))
    gradient_magnitude = np.zeros((m, n, m_max))
    angle = np.zeros((m, n, m_max))
    
    # Create mask to ignore boundary pixels
    mask = (image == 0.001)
    struct_elem = morphology.footprint_rectangle(5)
    mask = morphology.binary_dilation(mask, struct_elem)
    mask = 1 - mask
    
    for i in range(m_max):
        # Calculate current scale
        scale = sigma * (ratio ** i)
        radius = round(scale)
        
        # Create kernel coordinates
        j_vals = np.arange(-radius, radius + 1)
        k_vals = np.arange(-radius, radius + 1)
        x_array, y_array = np.meshgrid(j_vals, k_vals)
        
        # Create filters
        w = np.exp(-((x_array**2) + (y_array**2)) / (2 * scale))
        w2 = np.zeros((2 * radius + 1, 2 * radius + 1))
        w1 = np.zeros((2 * radius + 1, 2 * radius + 1))
        
        w2[radius+1:2*radius+1, :] = w[radius+1:2*radius+1, :]
        w2[0:radius, :] = -w[0:radius, :]
        w1[:, radius+1:2*radius+1] = w[:, radius+1:2*radius+1]
        w1[:, 0:radius] = -w[:, 0:radius]
        
        # Calculate gradients
        gx = cv2.filter2D(image, -1, w1)
        gx = gx * mask
        gy = cv2.filter2D(image, -1, w2)
        gy = gy * mask
        
        # Handle non-finite values
        gx = np.abs(np.nan_to_num(gx, nan=0.0, posinf=0.0, neginf=0.0))
        gy = np.abs(np.nan_to_num(gy, nan=0.0, posinf=0.0, neginf=0.0))
        
        # Calculate gradient magnitude and normalize
        temp_gradient = np.sqrt(gx**2 + gy**2)
        max_grad = np.max(temp_gradient)
        if max_grad > 0:
            temp_gradient = temp_gradient / max_grad
        gradient_magnitude[:, :, i] = temp_gradient
        
        # Calculate gradient angle
        temp_angle = np.arctan2(gy, gx)
        temp_angle = np.nan_to_num(temp_angle)
        temp_angle = temp_angle / np.pi * 180
        temp_angle[temp_angle < 0] += 180
        angle[:, :, i] = temp_angle
        
        # Calculate Harris components
        csh_11 = scale**2 * gx**2
        csh_12 = scale**2 * gx * gy
        csh_22 = scale**2 * gy**2
        
        # Apply Gaussian window
        gaussian_sigma = np.sqrt(2) * scale
        width = round(3 * gaussian_sigma)
        width_windows = 2 * width + 1
        
        # Create circular Gaussian window
        w_gaussian = cv2.getGaussianKernel(width_windows, gaussian_sigma)
        w_gaussian = w_gaussian @ w_gaussian.T
        
        # Apply circular mask
        y_indices, x_indices = np.mgrid[-width:width+1, -width:width+1]
        circle_mask = (x_indices**2 + y_indices**2) <= width**2
        w_gaussian = w_gaussian * circle_mask
        
        # Apply window to Harris components
        csh_11 = cv2.filter2D(csh_11, -1, w_gaussian)
        csh_12 = cv2.filter2D(csh_12, -1, w_gaussian)
        csh_21 = csh_12  # Symmetric matrix
        csh_22 = cv2.filter2D(csh_22, -1, w_gaussian)
        
        # Calculate Harris function
        harris_function[:, :, i] = (csh_11 * csh_22 - csh_21 * csh_12 - d * (csh_11 + csh_22)**2)
        
    return harris_function, gradient_magnitude, angle


def build_scale_sar(image, sigma, m_max, ratio, d):
    """
    Build scale space for SAR images using ratio-based gradients.
    
    Parameters:
    -----------
    image : ndarray
        Input SAR image (grayscale)
    sigma : float
        Base scale parameter
    m_max : int
        Number of scale levels
    ratio : float
        Scale ratio between consecutive levels
    d : float
        Harris function parameter
        
    Returns:
    --------
    tuple
        (harris_function, gradient, angle) - scale-space structures
    """
    m, n = image.shape
    harris_function = np.zeros((m, n, m_max))
    gradient = np.zeros((m, n, m_max))
    angle = np.zeros((m, n, m_max))
    
    # Create mask to ignore boundary pixels
    mask = (image == 0.001)
    struct_elem = morphology.footprint_rectangle(5, 5)

    mask = morphology.binary_dilation(mask, struct_elem)
    mask = 1 - mask
    
    for i in range(m_max):
        # Calculate current scale
        scale = sigma * (ratio ** i)
        radius = round(2 * scale)
        
        # Create kernel coordinates
        j_vals = np.arange(-radius, radius + 1)
        k_vals = np.arange(-radius, radius + 1)
        x_array, y_array = np.meshgrid(j_vals, k_vals)
        
        # Create SAR-specific filter
        w = np.exp(-(np.abs(x_array) + np.abs(y_array)) / scale)
        
        # Create quadrant filters
        w34 = np.zeros((2 * radius + 1, 2 * radius + 1))
        w12 = np.zeros((2 * radius + 1, 2 * radius + 1))
        w14 = np.zeros((2 * radius + 1, 2 * radius + 1))
        w23 = np.zeros((2 * radius + 1, 2 * radius + 1))
        
        w34[radius+1:2*radius+1, :] = w[radius+1:2*radius+1, :]
        w12[0:radius, :] = w[0:radius, :]
        w14[:, radius+1:2*radius+1] = w[:, radius+1:2*radius+1]
        w23[:, 0:radius] = w[:, 0:radius]
        
        # Filter image with quadrant filters
        m34 = cv2.filter2D(image, -1, w34)
        m12 = cv2.filter2D(image, -1, w12)
        m14 = cv2.filter2D(image, -1, w14)
        m23 = cv2.filter2D(image, -1, w23)
        
        # Compute gradient as log ratio of means (robust for SAR)
        gx = np.log(m14 / np.maximum(m23, 1e-10)) * mask
        gy = np.log(m34 / np.maximum(m12, 1e-10)) * mask
        
        # Handle non-finite values
        gx = np.abs(np.nan_to_num(gx, nan=0.0, posinf=0.0, neginf=0.0))
        gy = np.abs(np.nan_to_num(gy, nan=0.0, posinf=0.0, neginf=0.0))
        
        # Calculate gradient magnitude and normalize
        temp_gradient = np.sqrt(gx**2 + gy**2)
        max_grad = np.max(temp_gradient)
        if max_grad > 0:
            temp_gradient = temp_gradient / max_grad
        gradient[:, :, i] = temp_gradient
        
        # Calculate gradient angle
        temp_angle = np.arctan2(gy, gx)
        temp_angle = np.nan_to_num(temp_angle)
        temp_angle = temp_angle / np.pi * 180
        temp_angle[temp_angle < 0] += 180
        angle[:, :, i] = temp_angle
        
        # Calculate Harris components
        csh_11 = scale**2 * gx**2
        csh_12 = scale**2 * gx * gy
        csh_22 = scale**2 * gy**2
        
        # Apply Gaussian window
        gaussian_sigma = np.sqrt(2) * scale
        width = round(3 * gaussian_sigma)
        width_windows = 2 * width + 1
        
        # Create circular Gaussian window
        w_gaussian = cv2.getGaussianKernel(width_windows, gaussian_sigma)
        w_gaussian = w_gaussian @ w_gaussian.T
        
        # Apply circular mask
        y_indices, x_indices = np.mgrid[-width:width+1, -width:width+1]
        circle_mask = (x_indices**2 + y_indices**2) <= width**2
        w_gaussian = w_gaussian * circle_mask
        
        # Apply window to Harris components
        csh_11 = cv2.filter2D(csh_11, -1, w_gaussian)
        csh_12 = cv2.filter2D(csh_12, -1, w_gaussian)
        csh_21 = csh_12  # Symmetric matrix
        csh_22 = cv2.filter2D(csh_22, -1, w_gaussian)
        
        # Calculate Harris function
        harris_function[:, :, i] = (csh_11 * csh_22 - csh_21 * csh_12 - d * (csh_11 + csh_22)**2)
        
    return harris_function, gradient, angle


def calculate_orientation_hist_sar(x, y, scale, gradient, angle, n_bins):
    """
    Calculate orientation histogram for a keypoint in SAR image.
    
    Parameters:
    -----------
    x, y : int
        Keypoint coordinates
    scale : float
        Scale of the keypoint
    gradient : ndarray
        Gradient magnitude image
    angle : ndarray
        Gradient orientation image
    n_bins : int
        Number of histogram bins (typically 18 for 180 degrees)
        
    Returns:
    --------
    tuple
        (histogram, max_value) - orientation histogram and its maximum value
    """
    m, n_cols = gradient.shape
    
    # Calculate the region of interest
    radius = round(min(6 * scale, min(m/2, n_cols/2)))
    
    # Define region boundaries
    radius_x_left = max(int(x - radius), 0)
    radius_x_right = min(int(x + radius), n_cols - 1)
    radius_y_up = max(int(y - radius), 0)
    radius_y_down = min(int(y + radius), m - 1)
    
    # Extract subregions
    sub_gradient = gradient[radius_y_up:radius_y_down+1, radius_x_left:radius_x_right+1]
    sub_angle = angle[radius_y_up:radius_y_down+1, radius_x_left:radius_x_right+1]
    
    # Define center coordinates in the subregion
    center_x = x - radius_x_left
    center_y = y - radius_y_up
    
    # Create coordinate grids relative to the keypoint
    x_coords = np.arange(-(x - radius_x_left), (radius_x_right - x) + 1)
    y_coords = np.arange(-(y - radius_y_up), (radius_y_down - y) + 1)
    xx, yy = np.meshgrid(x_coords, y_coords)
    
    # Calculate histogram bins for each pixel
    bin_indices = np.round(sub_angle * n_bins / 180).astype(int)
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Initialize histogram
    temp_hist = np.zeros(n_bins)
    
    # Create a circular mask
    mask = xx**2 + yy**2 <= radius**2
    
    # Fill histogram with weighted orientation values
    rows, cols = sub_angle.shape
    for i in range(rows):
        for j in range(cols):
            if mask[i, j]:
                bin_idx = bin_indices[i, j]
                temp_hist[bin_idx] += sub_gradient[i, j]
    
    # Smooth histogram using a [1,4,6,4,1] filter
    hist = np.zeros(n_bins)
    for i in range(n_bins):
        prev2 = (i - 2) % n_bins
        prev1 = (i - 1) % n_bins
        next1 = (i + 1) % n_bins
        next2 = (i + 2) % n_bins
        
        hist[i] = (temp_hist[prev2] + temp_hist[next2]) / 16 + \
                 4 * (temp_hist[prev1] + temp_hist[next1]) / 16 + \
                 6 * temp_hist[i] / 16
    
    max_value = np.max(hist)
    return hist, max_value


def find_scale_extreme(harris_function, threshold, sigma, ratio, gradient, angle):
    """
    Find scale-space extrema points in the Harris function.
    
    Parameters:
    -----------
    harris_function : ndarray
        Harris function response
    threshold : float
        Response threshold for keypoint detection
    sigma : float
        Base scale parameter
    ratio : float
        Scale ratio between consecutive levels
    gradient : ndarray
        Gradient magnitude images at different scales
    angle : ndarray
        Gradient orientation images at different scales
        
    Returns:
    --------
    ndarray
        Array of keypoints with [x, y, scale, layer, orientation, response]
    """
    m, n, num_layers = harris_function.shape
    
    # Parameters
    border_width = 2
    hist_bins = 18  # For 180 degrees
    ori_peak_ratio = 0.8  # For selecting orientation peaks
    
    key_points = []  # List to store keypoints
    
    # For each scale
    for layer in range(num_layers):
        current_harris = harris_function[:, :, layer]
        current_gradient = gradient[:, :, layer]
        current_angle = angle[:, :, layer]
        current_scale = sigma * (ratio ** layer)
        
        # Search for local extrema
        for y in range(border_width, m - border_width):
            for x in range(border_width, n - border_width):
                val = current_harris[y, x]
                
                # Check if value is above threshold and is local maximum
                if val > threshold and all([
                    val > current_harris[y-1:y+2, x-1:x+2].flatten()[i]
                    for i in [0,1,2,3,5,6,7,8]  # All neighbors except center
                ]):
                    # Calculate orientation histogram
                    hist, max_value = calculate_orientation_hist_sar(
                        x, y, current_scale, current_gradient, current_angle, hist_bins
                    )
                    
                    # Find orientation peaks
                    mag_thr = max_value * ori_peak_ratio
                    
                    # For each bin, check if it's a peak
                    for bin_idx in range(hist_bins):
                        prev_idx = (bin_idx - 1) % hist_bins
                        next_idx = (bin_idx + 1) % hist_bins
                        
                        if (hist[bin_idx] > hist[prev_idx] and 
                            hist[bin_idx] > hist[next_idx] and 
                            hist[bin_idx] > mag_thr):
                            
                            # Refine peak by interpolation
                            bin_value = bin_idx + 0.5 * (hist[prev_idx] - hist[next_idx]) / \
                                       (hist[prev_idx] + hist[next_idx] - 2 * hist[bin_idx])
                            
                            # Handle wraparound
                            if bin_value < 0:
                                bin_value += hist_bins
                            elif bin_value >= hist_bins:
                                bin_value -= hist_bins
                            
                            # Calculate orientation in degrees
                            orientation = (180.0 / hist_bins) * bin_value
                            
                            # Add keypoint [x, y, scale, layer, orientation, response]
                            key_points.append([
                                x, y, current_scale, layer, orientation, hist[bin_idx]
                            ])
    
    # Convert to numpy array
    if not key_points:
        return np.array([]).reshape(0, 6)
    
    return np.array(key_points)


def calc_log_polar_descriptor(gradient, angle, x, y, scale, main_angle, d, n):
    """
    Calculate log-polar descriptor for a keypoint.
    
    Parameters:
    -----------
    gradient : ndarray
        Gradient magnitude image
    angle : ndarray
        Gradient orientation image
    x, y : float
        Keypoint coordinates
    scale : float
        Keypoint scale
    main_angle : float
        Keypoint dominant orientation
    d : int
        Number of spatial bins in descriptor
    n : int
        Number of orientation bins in descriptor
        
    Returns:
    --------
    ndarray
        Feature descriptor vector
    """
    # Precompute rotation parameters
    cos_t = np.cos(-main_angle * np.pi / 180)
    sin_t = np.sin(-main_angle * np.pi / 180)
    
    m, n_cols = gradient.shape
    
    # Set descriptor radius based on scale
    radius = round(min(12 * scale, min(m, n_cols) / 3))
    
    # Define region boundaries
    radius_x_left = max(int(x - radius), 0)
    radius_x_right = min(int(x + radius), n_cols - 1)
    radius_y_up = max(int(y - radius), 0)
    radius_y_down = min(int(y + radius), m - 1)
    
    # Calculate center coordinates in the extracted region
    center_x = x - radius_x_left
    center_y = y - radius_y_up
    
    # Extract subregions
    sub_gradient = gradient[radius_y_up:radius_y_down+1, radius_x_left:radius_x_right+1]
    sub_angle = angle[radius_y_up:radius_y_down+1, radius_x_left:radius_x_right+1]
    
    # Adjust orientation relative to the keypoint
    sub_angle = np.round((sub_angle - main_angle) * n / 180)
    sub_angle = np.mod(sub_angle, n)
    sub_angle[sub_angle == 0] = n
    
    # Create coordinate grids
    x_coords = np.arange(-(x - radius_x_left), (radius_x_right - x) + 1)
    y_coords = np.arange(-(y - radius_y_up), (radius_y_down - y) + 1)
    xx, yy = np.meshgrid(x_coords, y_coords)
    
    # Rotate coordinates
    c_rot = xx * cos_t - yy * sin_t
    r_rot = xx * sin_t + yy * cos_t
    
    # Convert to log-polar coordinates
    log_angle = np.arctan2(r_rot, c_rot) * 180 / np.pi
    log_angle[log_angle < 0] += 360
    log_amplitude = np.log2(np.sqrt(c_rot**2 + r_rot**2))
    
    # Quantize log-polar coordinates
    log_angle = np.round(log_angle * d / 360)
    log_angle = np.mod(log_angle, d)
    
    # Define amplitude bins
    r1 = np.log2(radius * 0.73 * 0.25)
    r2 = np.log2(radius * 0.73)
    
    log_amplitude_bins = np.ones_like(log_amplitude)
    log_amplitude_bins[(log_amplitude > r1) & (log_amplitude <= r2)] = 2
    log_amplitude_bins[log_amplitude > r2] = 3
    
    # Initialize histogram
    temp_hist = np.zeros((2*d+1) * n)
    
    # Fill histogram
    rows, cols = log_angle.shape
    for i in range(rows):
        for j in range(cols):
            # Only consider points within the circle
            if (i - center_y)**2 + (j - center_x)**2 <= radius**2:
                angle_bin = int(log_angle[i, j])
                amplitude_bin = int(log_amplitude_bins[i, j])
                bin_vertical = int(sub_angle[i, j])
                mag = sub_gradient[i, j]
                
                if amplitude_bin == 1:
                    # Central bin
                    temp_hist[bin_vertical - 1] += mag
                else:
                    # Outer bins
                    idx = ((amplitude_bin - 2) * d + angle_bin) * n + bin_vertical
                    temp_hist[n + idx - 1] += mag
    
    # Normalize the descriptor
    norm = np.sqrt(np.sum(temp_hist**2))
    if norm > 0:
        temp_hist = temp_hist / norm
    
    # Threshold and renormalize
    temp_hist[temp_hist > 0.2] = 0.2
    norm = np.sqrt(np.sum(temp_hist**2))
    if norm > 0:
        temp_hist = temp_hist / norm
    
    return temp_hist


def calc_descriptors(gradient, angle, key_point_array, is_multi_region=False):
    """
    Calculate descriptors for an array of keypoints.
    
    Parameters:
    -----------
    gradient : ndarray
        3D array of gradient magnitude images at different scales
    angle : ndarray
        3D array of gradient orientation images at different scales
    key_point_array : ndarray
        Array of keypoints [x, y, scale, layer, orientation, response]
    is_multi_region : bool
        Whether to compute descriptors at multiple scales
        
    Returns:
    --------
    tuple
        (descriptors, keypoints) - feature descriptors and corresponding keypoints
    """
    circle_bin = 8
    hist_bins = 8
    
    num_keypoints = key_point_array.shape[0]
    d = circle_bin
    n = hist_bins
    
    # Define descriptor size based on multi-region flag
    if not is_multi_region:
        descriptors = np.zeros((num_keypoints, (2*d+1) * n))
    else:
        descriptors = np.zeros((num_keypoints, (2*d+1) * n * 3))
    
    for i in range(num_keypoints):
        x = key_point_array[i, 0]
        y = key_point_array[i, 1]
        scale = key_point_array[i, 2]
        layer = int(key_point_array[i, 3])
        main_angle = key_point_array[i, 4]
        
        # Get gradient and angle at the keypoint's scale
        current_gradient = gradient[:, :, layer]
        current_gradient = current_gradient / np.max(current_gradient)
        current_angle = angle[:, :, layer]
        
        # Calculate descriptor at the main scale
        descriptors[i, 0:(2*d+1)*n] = calc_log_polar_descriptor(
            current_gradient, current_angle, x, y, scale, main_angle, d, n
        )
        
        # If multi-region is requested, calculate descriptors at additional scales
        if is_multi_region:
            # Larger region (4/3 scale)
            descriptors[i, (2*d+1)*n:(2*d+1)*n*2] = calc_log_polar_descriptor(
                current_gradient, current_angle, x, y, scale*4/3, main_angle, d, n
            )
            
            # Smaller region (2/3 scale)
            descriptors[i, (2*d+1)*n*2:(2*d+1)*n*3] = calc_log_polar_descriptor(
                current_gradient, current_angle, x, y, scale*2/3, main_angle, d, n
            )
    
    return descriptors, key_point_array


def filter_keypoints_by_strength(keypoints, max_points=5000):
    """
    Filter keypoints to keep only the strongest ones.
    
    Parameters:
    -----------
    keypoints : ndarray
        Array of keypoints [x, y, scale, layer, orientation, response]
    max_points : int
        Maximum number of keypoints to keep
        
    Returns:
    --------
    ndarray
        Filtered keypoints
    """
    if keypoints.shape[0] <= max_points:
        return keypoints
    
    # Sort by response (column 5)
    idx = np.argsort(keypoints[:, 5])[::-1]
    return keypoints[idx[:max_points]]


def match_keypoints(des1, loc1, des2, loc2, ratio_thresh=0.9, is_multi_region=False):
    """
    Match keypoints between two images using nearest neighbor ratio test.
    
    Parameters:
    -----------
    des1, des2 : ndarray
        Descriptors from the two images
    loc1, loc2 : ndarray
        Keypoints from the two images
    ratio_thresh : float
        Lowe's ratio test threshold
    is_multi_region : bool
        Whether descriptors were computed with multiple regions
        
    Returns:
    --------
    tuple
        (matches1, matches2) - Matched keypoints from both images
    """
    if not is_multi_region:
        # Use standard descriptors
        matches = []
        
        # For each descriptor in the first image
        for i in range(des1.shape[0]):
            # Compute dot products with all descriptors in the second image
            dot_products = des1[i] @ des2.T
            
            # Convert to angles (smaller angle = better match)
            angles = np.arccos(np.clip(dot_products, -1.0, 1.0))
            
            # Find the two best matches
            idx = np.argsort(angles)
            
            # Apply ratio test
            if idx.size > 1 and angles[idx[0]] < ratio_thresh * angles[idx[1]]:
                matches.append((i, idx[0]))
    else:
        # When using multi-region descriptors, split them and use the average score
        des1_regions = np.split(des1, 3, axis=1)
        des2_regions = np.split(des2, 3, axis=1)
        
        matches = []
        
        # For each descriptor in the first image
        for i in range(des1.shape[0]):
            # Compute dot products for each region
            dp1 = des1_regions[0][i] @ des2_regions[0].T
            dp2 = des1_regions[1][i] @ des2_regions[1].T
            dp3 = des1_regions[2][i] @ des2_regions[2].T
            
            # Take the average of the three regions
            dot_products = (dp1 + dp2 + dp3) / 3
            
            # Convert to angles
            angles = np.arccos(np.clip(dot_products, -1.0, 1.0))
            
            # Find the two best matches
            idx = np.argsort(angles)
            
            # Apply ratio test
            if idx.size > 1 and angles[idx[0]] < ratio_thresh * angles[idx[1]]:
                matches.append((i, idx[0]))
    
    if not matches:
        return np.array([]), np.array([])
    
    # Extract matched keypoints
    matches = np.array(matches)
    idx1, idx2 = matches[:, 0], matches[:, 1]
    
    matches1 = np.hstack((loc1[idx1, :], idx2.reshape(-1, 1)))
    matches2 = np.hstack((loc2[idx2, :], idx2.reshape(-1, 1)))
    
    # Remove duplicate matches based on location in the second image
    unique_matches = np.concatenate((matches1[:, :2], matches2[:, :2]), axis=1)
    _, unique_idx = np.unique(unique_matches, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    
    matches1 = matches1[unique_idx]
    matches2 = matches2[unique_idx]
    
    print(f"NNDR found {len(matches1)} matches.")
    
    return matches1, matches2


def fsc_match(pts1, pts2, method='affine', error_thresh=3):
    """
    Filter matches using Fast Sample Consensus (FSC) algorithm.
    
    Parameters:
    -----------
    pts1, pts2 : ndarray
        Matched keypoints from both images
    method : str
        Transformation model ('affine', 'homography', etc.)
    error_thresh : float
        Error threshold for inlier classification
        
    Returns:
    --------
    tuple
        (transformation_matrix, inlier_pts1, inlier_pts2)
    """
    # Extract coordinates
    src_pts = pts1[:, :2].astype(np.float32)
    dst_pts = pts2[:, :2].astype(np.float32)
    
    # Use OpenCV's findHomography with RANSAC
    if method == 'homography':
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, error_thresh)
    else:  # Default to affine
        # Need at least 3 points for affine
        if len(src_pts) < 3:
            return np.eye(3), np.array([]), np.array([])
            
        # Estimate affine transform with RANSAC
        H, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, 
                                              ransacReprojThreshold=error_thresh)
        
        # Convert to 3x3 homography format
        if H is not None:
            H_full = np.eye(3)
            H_full[:2, :] = H
            H = H_full
    
    if H is None:
        return np.eye(3), np.array([]), np.array([])
    
    # Get inliers
    inliers = mask.ravel() == 1
    inlier_pts1 = pts1[inliers]
    inlier_pts2 = pts2[inliers]
    
    # Calculate RMSE for inliers
    if np.sum(inliers) > 0:
        if method == 'homography':
            # For homography, apply perspective transform
            src_homogeneous = np.hstack((src_pts[inliers], np.ones((np.sum(inliers), 1))))
            transformed = src_homogeneous @ H.T
            transformed = transformed[:, :2] / transformed[:, 2:3]
            errors = np.sqrt(np.sum((transformed - dst_pts[inliers])**2, axis=1))
            rmse = np.mean(errors)
        else:
            # For affine, direct transform
            transformed = cv2.transform(src_pts[inliers].reshape(-1, 1, 2), H[:2]).reshape(-1, 2)
            errors = np.sqrt(np.sum((transformed - dst_pts[inliers])**2, axis=1))
            rmse = np.mean(errors)
    else:
        rmse = float('inf')
    
    print(f"After FSC found {np.sum(inliers)} matches. RMSE: {rmse:.2f}")
    
    return H, inlier_pts1, inlier_pts2


def csc2_match(img1, img2, des1, loc1, des2, loc2, dist_ratio=0.95, error_thresh=3):
    """
    Implementation of CSC2 (Consensus-based Keypoint Matching) algorithm.
    
    Parameters:
    -----------
    img1, img2 : ndarray
        Input images
    des1, des2 : ndarray
        Descriptors from both images
    loc1, loc2 : ndarray
        Keypoints from both images
    dist_ratio : float
        Distance ratio for initial matching
    error_thresh : float
        Error threshold for FSC
        
    Returns:
    --------
    tuple
        (H, inlier_pts1, inlier_pts2) - Transformation matrix and inlier points
    """
    # Initial matching using descriptor distance ratio
    matches = []
    des2t = des2.T
    
    for i in range(des1.shape[0]):
        # Compute dot products
        dot_products = des1[i] @ des2t
        
        # Convert to angles (smaller = better match)
        angles = np.arccos(np.clip(dot_products, -1.0, 1.0))
        
        # Sort matches
        idx = np.argsort(angles)
        
        # Apply ratio test
        if angles[idx[0]] < dist_ratio * angles[idx[1]]:
            matches.append((i, idx[0]))
    
    if not matches:
        return np.eye(3), np.array([]), np.array([])
    
    # Convert to numpy arrays
    matches = np.array(matches)
    idx1, idx2 = matches[:, 0], matches[:, 1]
    
    # Extract matched keypoints
    cor1 = loc1[idx1, :2]
    cor2 = loc2[idx2, :2]
    
    print(f"Initial matching found {len(cor1)} matches")
    
    # First round of FSC with higher threshold
    H1, _, _ = fsc_match(cor1, cor2, 'affine', error_thresh=10)
    
    # Refine matches using the estimated transformation
    cor1_new = []
    cor2_new = []
    
    # For each point in the first image
    for i in range(loc1.shape[0]):
        # Get point coordinates (homogeneous)
        pt = np.array([loc1[i, 0], loc1[i, 1], 1])
        
        # Transform the point using the initial homography
        pt_trans = H1 @ pt
        pt_trans = pt_trans[:2]
        
        # Find closest matches in the second image
        diffs = loc2[:, :2] - pt_trans
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        
        # Get the top 3 closest matches
        top_indices = np.argsort(distances)[:3]
        
        # Find the best match using descriptors
        if len(top_indices) > 0:
            # Compute descriptor similarity
            dot_products = des1[i] @ des2[top_indices].T
            angles = np.arccos(np.clip(dot_products, -1.0, 1.0))
            best_idx = top_indices[np.argmin(angles)]
            
            # Add to refined matches
            cor1_new.append(loc1[i, :2])
            cor2_new.append(loc2[best_idx, :2])
    
    if not cor1_new:
        return H1, cor1, cor2
    
    # Convert to numpy arrays
    cor1_new = np.array(cor1_new)
    cor2_new = np.array(cor2_new)
    
    # Second round of FSC with lower threshold
    H2, inlier_pts1, inlier_pts2 = fsc_match(cor1_new, cor2_new, 'affine', error_thresh=error_thresh)
    
    print(f"After CSC2 found {len(inlier_pts1)} matches")
    
    # Visualize matches if available
    if len(inlier_pts1) > 0:
        visualize_matches(img1, img2, inlier_pts1[:, :2], inlier_pts2[:, :2])
    
    return H2, inlier_pts1, inlier_pts2


def image_fusion(img1, img2, H):
    """
    Fuse two images using the provided transformation matrix.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Create a larger canvas
    canvas_width = w1 * 3
    canvas_height = h1 * 3
    
    # Create a transformation to center the reference image
    offset = np.array([
        [1, 0, w1],
        [0, 1, h1],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Combine transformations
    full_transform = offset @ H
    
    # Convert images to 8-bit if needed
    if img1.dtype != np.uint8 and np.issubdtype(img1.dtype, np.floating):
        img1 = (img1 * 255).astype(np.uint8)
    if img2.dtype != np.uint8 and np.issubdtype(img2.dtype, np.floating):
        img2 = (img2 * 255).astype(np.uint8)
    
    # Warp both images to the canvas
    if len(img1.shape) == 2:
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    else:
        img1_rgb = img1.copy()
        img2_rgb = img2.copy()
    
    
    warped1 = cv2.warpPerspective(img1_rgb, offset, (canvas_width, canvas_height))
    warped2 = cv2.warpPerspective(img2_rgb, full_transform, (canvas_width, canvas_height))
    
    # Create blended image
    fused = np.zeros_like(warped1, dtype=np.float32)
    
    # Create masks for blending
    mask1 = np.any(warped1 > 0, axis=2)
    mask2 = np.any(warped2 > 0, axis=2)
    
    # Areas where both images exist
    both = np.logical_and(mask1, mask2)
    only1 = np.logical_and(mask1, ~mask2)
    only2 = np.logical_and(mask2, ~mask1)
    
    # Alpha blending
    fused[both] = warped1[both] * 0.5 + warped2[both] * 0.5
    fused[only1] = warped1[only1]
    fused[only2] = warped2[only2]
    
    fused = np.clip(fused, 0, 255).astype(np.uint8)
    
    # Create checkerboard visualization
    checkerboard = np.zeros_like(fused)
    block_size = 64
    
    # Create checkerboard pattern
    y_blocks = fused.shape[0] // block_size + 1
    x_blocks = fused.shape[1] // block_size + 1
    
    for y in range(y_blocks):
        for x in range(x_blocks):
            y1 = y * block_size
            y2 = min((y + 1) * block_size, fused.shape[0])
            x1 = x * block_size
            x2 = min((x + 1) * block_size, fused.shape[1])
            
            if (y + x) % 2 == 0:
                checkerboard[y1:y2, x1:x2] = warped1[y1:y2, x1:x2]
            else:
                checkerboard[y1:y2, x1:x2] = warped2[y1:y2, x1:x2]
    
    # Crop to remove empty borders
    non_zero = np.where(np.any(fused > 0, axis=2))
    if len(non_zero[0]) > 0:
        y_min, y_max = np.min(non_zero[0]), np.max(non_zero[0])
        x_min, x_max = np.min(non_zero[1]), np.max(non_zero[1])
        
        # Add border
        border = 10
        y_min = max(0, y_min - border)
        y_max = min(fused.shape[0] - 1, y_max + border)
        x_min = max(0, x_min - border)
        x_max = min(fused.shape[1] - 1, x_max + border)
        
        fused = fused[y_min:y_max+1, x_min:x_max+1]
        checkerboard = checkerboard[y_min:y_max+1, x_min:x_max+1]
    
    # Visualize
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(fused)
    plt.title('Fused Image')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(checkerboard)
    plt.title('Checkerboard Visualization')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('fusion_result.png')
    plt.show()
    
    return fused, checkerboard

def visualize_matches(img1, img2, pts1, pts2):
    """
    Visualize matched keypoints between two images.
    
    Parameters:
    -----------
    img1, img2 : ndarray
        Input images
    pts1, pts2 : ndarray
        Matched keypoints
    """
    # Convert images to RGB if needed
    if len(img1.shape) == 2:
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    else:
        img1_rgb = img1.copy()
    
    if len(img2.shape) == 2:
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    else:
        img2_rgb = img2.copy()
    
    # Convert to 8-bit unsigned integer format if needed
    if img1_rgb.dtype != np.uint8:
        img1_rgb = (img1_rgb * 255).astype(np.uint8)
    if img2_rgb.dtype != np.uint8:
        img2_rgb = (img2_rgb * 255).astype(np.uint8)
    
    # Create match visualization
    matches_viz = cv2.drawMatches(
        img2_rgb, 
        [cv2.KeyPoint(float(x), float(y), 1) for x, y in pts2],
        img1_rgb, 
        [cv2.KeyPoint(float(x), float(y), 1) for x, y in pts1],
        [cv2.DMatch(i, i, 0) for i in range(len(pts1))],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    plt.figure(figsize=(12, 8))
    plt.imshow(matches_viz)
    plt.title(f'Matched Keypoints ({len(pts1)} matches)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('matches.png')
    plt.show()

def main():
    """
    Main function for OS-SIFT (Orientation-Scale SIFT) algorithm.
    """
    # Read images
    opt_img_path = '../DATASET/RemoteSensing/SAR_Optical/SO1b.png'
    sar_img_path = '../DATASET/RemoteSensing/SAR_Optical/SO1a.png'
    
    try:
        image_1 = cv2.imread(opt_img_path, cv2.IMREAD_GRAYSCALE)
        image_2 = cv2.imread(sar_img_path, cv2.IMREAD_GRAYSCALE)
        
        if image_1 is None or image_2 is None:
            print("Error: Could not read one or both images.")
            return
        
        # Normalize images to [0, 1]
        image_1 = cv2.normalize(image_1.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
        image_2 = cv2.normalize(image_2.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
        
        # Add small constant to avoid division by zero
        image_11 = image_1 + 0.001
        image_22 = image_2 + 0.001
    except Exception as e:
        print(f"Error loading images: {e}")
        return
    
    # Parameters
    sigma = 2.0          # Base scale parameter
    ratio = 2**(1/3)     # Scale ratio between consecutive levels
    m_max = 8            # Number of scale levels
    d = 0.04            # Harris function parameter
    d_sh_1 = 0.00001     # Harris response threshold for optical image
    d_sh_2 = 0.00001     # Harris response threshold for SAR image
    is_multi_region = False  # Whether to use multi-region descriptors
    
    print("Creating scale space representations...")
    
    # Create scale space representations
    sar_harris_1, gradient_1, angle_1 = build_scale_opt(image_11, sigma, m_max, ratio, d)
    sar_harris_2, gradient_2, angle_2 = build_scale_sar(image_22, sigma, m_max, ratio, d)
    
    print("Detecting keypoints...")
    
    # Detect keypoints
    keypoints_1 = find_scale_extreme(sar_harris_1, d_sh_1, sigma, ratio, gradient_1, angle_1)
    keypoints_2 = find_scale_extreme(sar_harris_2, d_sh_2, sigma, ratio, gradient_2, angle_2)
    
    # Filter to strongest keypoints
    keypoints_1 = filter_keypoints_by_strength(keypoints_1, 5000)
    keypoints_2 = filter_keypoints_by_strength(keypoints_2, 5000)
    
    print(f"Found {len(keypoints_1)} keypoints in optical image")
    print(f"Found {len(keypoints_2)} keypoints in SAR image")
    
    # Compute descriptors
    print("Computing descriptors...")
    descriptors_1, keypoints_1 = calc_descriptors(gradient_1, angle_1, keypoints_1, is_multi_region)
    descriptors_2, keypoints_2 = calc_descriptors(gradient_2, angle_2, keypoints_2, is_multi_region)
    
    # Match keypoints using CSC2 algorithm
    print("Matching keypoints...")
    H, matches_1, matches_2 = csc2_match(image_1, image_2, descriptors_2, keypoints_2, 
                                       descriptors_1, keypoints_1)
    
    # Fuse images using the transformation
    print("Fusing images...")
    image_fusion(image_1, image_2, H)
    
    print("Processing complete!")


if __name__ == "__main__":
    main()