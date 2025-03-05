#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Metrics for evaluating image registration performance.
"""

import numpy as np
import cv2

def compute_rmse_points(points1, points2):
    """
    Computes the Root Mean Square Error between two sets of points.
    
    Args:
        points1 (np.ndarray): First set of points
        points2 (np.ndarray): Second set of points
        
    Returns:
        float: RMSE between the point sets
    """
    if len(points1) != len(points2):
        raise ValueError(f"Point sets must have the same length. Got {len(points1)} and {len(points2)}")
    
    points1 = np.array(points1)
    points2 = np.array(points2)
    
    # Ensure points are 2D (x, y)
    if points1.shape[1] > 2:
        points1 = points1[:, :2]
    if points2.shape[1] > 2:
        points2 = points2[:, :2]
    
    # Compute squared Euclidean distances
    squared_diffs = np.sum((points1 - points2) ** 2, axis=1)
    
    # Compute RMSE
    rmse = np.sqrt(np.mean(squared_diffs))
    
    return rmse

def compute_transformation_error(H_pred, H_gt):
    """
    Computes the error between predicted and ground truth homography matrices.
    
    Several metrics are computed:
    1. RMSE - Root Mean Square Error between matrix elements
    2. Frobenius norm of the difference matrix
    3. Transform error on a set of sample points
    
    Args:
        H_pred (np.ndarray): Predicted homography matrix
        H_gt (np.ndarray): Ground truth homography matrix
        
    Returns:
        dict: Dictionary with error metrics
    """
    if H_pred is None or H_gt is None:
        return {
            'rmse': None,
            'frobenius': None,
            'transform_error': None
        }
    
    # Ensure both matrices are normalized for fair comparison
    H_pred_norm = H_pred / np.linalg.norm(H_pred)
    H_gt_norm = H_gt / np.linalg.norm(H_gt)
    
    # 1. RMSE between matrix elements
    rmse = np.sqrt(np.mean((H_pred_norm - H_gt_norm) ** 2))
    
    # 2. Frobenius norm of difference
    frobenius = np.linalg.norm(H_pred_norm - H_gt_norm, 'fro')
    
    # 3. Transform error on sample points
    # Create a grid of sample points
    h, w = 10, 10
    sample_points = []
    for y in range(h):
        for x in range(w):
            sample_points.append([x * 50, y * 50, 1])  # Using a 500x500 grid with 50px spacing
    
    sample_points = np.array(sample_points)
    
    # Transform points with both matrices
    points_pred = np.dot(H_pred, sample_points.T).T
    points_gt = np.dot(H_gt, sample_points.T).T
    
    # Normalize homogeneous coordinates
    points_pred = points_pred[:, :2] / points_pred[:, 2:]
    points_gt = points_gt[:, :2] / points_gt[:, 2:]
    
    # Compute average Euclidean distance
    distances = np.sqrt(np.sum((points_pred - points_gt) ** 2, axis=1))
    transform_error = np.mean(distances)
    
    return {
        'rmse': rmse,
        'frobenius': frobenius,
        'transform_error': transform_error
    }

def min_point_distance(point, points_list):
    """
    Finds the minimum distance from a point to any point in a list.
    
    Args:
        point (tuple or list): The point to calculate distance from
        points_list (list or np.ndarray): List of points to calculate distance to
        
    Returns:
        float: Minimum distance to any point in the list
    """
    min_dist = float('inf')
    for p in points_list:
        dist = np.sqrt((point[0] - p[0])**2 + (point[1] - p[1])**2)
        min_dist = min(min_dist, dist)
    return min_dist

def count_repeatable_points(kp1, kp2, landmarks_fix, landmarks_mov, threshold=5):
    """
    Counts the number of repeatable points (keypoints that match with ground truth landmarks).
    
    Args:
        kp1: Keypoints from first image (either cv2.KeyPoint objects or numpy array of points)
        kp2: Keypoints from second image (either cv2.KeyPoint objects or numpy array of points)
        landmarks_fix: Ground truth landmarks for optical image
        landmarks_mov: Ground truth landmarks for SAR image
        threshold (float): Maximum distance to consider a keypoint as matching a landmark
        
    Returns:
        int: Number of repeatable points
    """
    # Handle both cv2.KeyPoint objects and numpy arrays
    if isinstance(kp1, list) and len(kp1) > 0 and isinstance(kp1[0], cv2.KeyPoint):
        kp1_points = np.array([kp.pt for kp in kp1])
    else:
        kp1_points = np.array(kp1)
    
    if isinstance(kp2, list) and len(kp2) > 0 and isinstance(kp2[0], cv2.KeyPoint):
        kp2_points = np.array([kp.pt for kp in kp2])
    else:
        kp2_points = np.array(kp2)
    
    # Count keypoints close to landmarks
    count_fix = 0
    for lm in landmarks_fix:
        if min_point_distance(lm, kp1_points) < threshold:
            count_fix += 1
    
    count_mov = 0
    for lm in landmarks_mov:
        if min_point_distance(lm, kp2_points) < threshold:
            count_mov += 1
    
    # Return the minimum of the two counts
    return min(count_fix, count_mov)

def count_ground_truth_matches(matches, kp1, kp2, landmarks_fix, landmarks_mov, threshold=5):
    """
    Counts the number of matches that correspond to ground truth landmark pairs.
    
    Args:
        matches (list): List of DMatch objects from OpenCV matcher
        kp1 (list): Keypoints from first image (cv2.KeyPoint objects)
        kp2 (list): Keypoints from second image (cv2.KeyPoint objects)
        landmarks_fix (np.ndarray): Ground truth landmarks for optical image
        landmarks_mov (np.ndarray): Ground truth landmarks for SAR image
        threshold (float): Maximum distance to consider a match as corresponding to a ground truth pair
        
    Returns:
        int: Number of matches corresponding to ground truth landmark pairs
    """
    count = 0
    for m in matches:
        pt1 = kp1[m.queryIdx].pt
        pt2 = kp2[m.trainIdx].pt
        
        # Check if this match is close to any ground truth landmark pair
        for lm_fix, lm_mov in zip(landmarks_fix, landmarks_mov):
            dist1 = np.sqrt((pt1[0] - lm_mov[0])**2 + (pt1[1] - lm_mov[1])**2)  # SAR to landmarks_mov
            dist2 = np.sqrt((pt2[0] - lm_fix[0])**2 + (pt2[1] - lm_fix[1])**2)  # Optical to landmarks_fix
            
            if dist1 < threshold and dist2 < threshold:
                count += 1
                break
    
    return count

def count_ground_truth_matches_array(pts1, pts2, landmarks_fix, landmarks_mov, threshold=5):
    """
    Counts the number of matches that correspond to ground truth landmark pairs
    when using keypoints in array format (for RIFT).
    
    Args:
        pts1 (np.ndarray): Array of keypoints from SAR image (shape: Nx2)
        pts2 (np.ndarray): Array of keypoints from optical image (shape: Nx2)
        landmarks_fix (np.ndarray): Ground truth landmarks for optical image
        landmarks_mov (np.ndarray): Ground truth landmarks for SAR image
        threshold (float): Distance threshold to consider a match as corresponding to a ground truth pair
        
    Returns:
        int: Number of matches corresponding to ground truth landmark pairs
    """
    count = 0
    
    # For each pair of matched keypoints
    for i in range(len(pts1)):
        pt1 = pts1[i]
        pt2 = pts2[i]
        
        # Check if this match is close to any ground truth landmark pair
        for lm_fix, lm_mov in zip(landmarks_fix, landmarks_mov):
            dist1 = np.sqrt((pt1[0] - lm_mov[0])**2 + (pt1[1] - lm_mov[1])**2)  # SAR to landmarks_mov
            dist2 = np.sqrt((pt2[0] - lm_fix[0])**2 + (pt2[1] - lm_fix[1])**2)  # Optical to landmarks_fix
            
            if dist1 < threshold and dist2 < threshold:
                count += 1
                break
    
    return count

def compute_entropy(img):
    """
    Computes the entropy of an image.
    
    Args:
        img (np.ndarray): Input image
        
    Returns:
        float: Entropy value of the image
    """
    # Ensure the image is normalized to 0-255
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Compute histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist / float(np.sum(hist))
    
    # Remove zero probabilities and compute entropy
    non_zero = hist[hist > 0]
    entropy = -np.sum(non_zero * np.log2(non_zero))
    
    return entropy

def compute_mutual_information(img1, img2):
    """
    Computes the mutual information between two images.
    
    Args:
        img1 (np.ndarray): First image
        img2 (np.ndarray): Second image
        
    Returns:
        float: Mutual information between the images
    """
    # Ensure the images are normalized to 0-255
    if img1.dtype != np.uint8:
        img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img2.dtype != np.uint8:
        img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Ensure images have the same dimensions
    h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
    img1 = img1[:h, :w]
    img2 = img2[:h, :w]
    
    # Compute histograms and entropies
    H1 = compute_entropy(img1)
    H2 = compute_entropy(img2)
    
    # Compute joint histogram
    hist_2d, _, _ = np.histogram2d(img1.flatten(), img2.flatten(), bins=256, range=[[0, 256], [0, 256]])
    
    # Normalize the joint histogram
    pxy = hist_2d / float(np.sum(hist_2d))
    pxy_non_zero = pxy[pxy > 0]
    
    # Compute joint entropy
    H12 = -np.sum(pxy_non_zero * np.log2(pxy_non_zero))
    
    # Compute mutual information
    MI = H1 + H2 - H12
    
    return MI