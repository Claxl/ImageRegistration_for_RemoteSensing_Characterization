#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Metrics for evaluating image registration performance.
"""

import numpy as np
import cv2

def compute_rmse_matrices(H_pred, H_gt):
    """
    Computes the Root Mean Square Error between predicted and ground truth transformation matrices.
    
    Args:
        H_pred (np.ndarray): Predicted homography matrix
        H_gt (np.ndarray): Ground truth homography matrix
        
    Returns:
        float: RMSE between the matrices
    """
    if H_pred is None or H_gt is None:
        return None
    
    # Normalize matrices for fair comparison
    H_pred_norm = H_pred / np.linalg.norm(H_pred)
    H_gt_norm = H_gt / np.linalg.norm(H_gt)
    
    # Compute RMSE between matrix elements
    rmse = np.sqrt(np.mean((H_pred_norm - H_gt_norm) ** 2))
    
    return rmse

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