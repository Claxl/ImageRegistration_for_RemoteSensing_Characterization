#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Metrics for evaluating image registration performance.

This module provides functions to compute various metrics to evaluate 
the quality of image registration, including:
- RMSE between transformation matrices
- RMSE between point sets
- Minimum point distance
- Image entropy
- Mutual information between images
"""

import numpy as np
import cv2
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_rmse_matrices(H_pred, H_gt):
    """
    Compute the absolute RMSE between two transformation matrices.
    
    Args:
        H_pred (np.ndarray): Predicted transformation matrix
        H_gt (np.ndarray): Ground truth transformation matrix
        
    Returns:
        float or None: RMSE between matrices, or None if inputs are invalid
    """
    if H_pred is None or H_gt is None:
        logger.warning("Cannot compute RMSE: one or both matrices are None")
        return None
    
    try:
        # Verify matrix shapes
        if H_pred.shape != H_gt.shape:
            logger.warning(f"Matrix shape mismatch: {H_pred.shape} vs {H_gt.shape}")
            return None
        
        # Compute RMSE directly between matrix elements
        rmse = np.sqrt(np.mean((H_pred - H_gt) ** 2))
        return rmse
        
    except Exception as e:
        logger.error(f"Error computing matrix RMSE: {e}")
        return None


def compute_rmse_points(points1, points2):
    """
    Compute the Root Mean Square Error between two sets of points.
    
    Args:
        points1 (np.ndarray): First set of points
        points2 (np.ndarray): Second set of points
        
    Returns:
        float: RMSE between the point sets
        
    Raises:
        ValueError: If point sets have different lengths
    """
    # Input validation
    if not isinstance(points1, np.ndarray):
        points1 = np.array(points1)
    if not isinstance(points2, np.ndarray):
        points2 = np.array(points2)
    
    if len(points1) != len(points2):
        raise ValueError(f"Point sets must have the same length. Got {len(points1)} and {len(points2)}")
    
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
    Find the minimum distance from a point to any point in a list.
    
    Args:
        point (tuple or np.ndarray): The point to calculate distance from (x, y)
        points_list (list or np.ndarray): List of points to calculate distance to
        
    Returns:
        float: Minimum distance to any point in the list
    """
    # Convert inputs to numpy arrays for vectorized operations
    if not isinstance(point, np.ndarray):
        point = np.array(point)
    
    if isinstance(points_list, list):
        points_list = np.array(points_list)
    
    # Handle empty points list
    if len(points_list) == 0:
        return float('inf')
    
    # Calculate distances using vectorized operations
    if point.ndim == 1 and points_list.ndim == 2:
        # Compute distances from point to all points in list
        diffs = points_list - point
        squared_dists = np.sum(diffs**2, axis=1)
        min_dist = np.sqrt(np.min(squared_dists))
        return min_dist
    else:
        # Fallback to loop implementation for irregular inputs
        min_dist = float('inf')
        for p in points_list:
            dist = np.sqrt((point[0] - p[0])**2 + (point[1] - p[1])**2)
            min_dist = min(min_dist, dist)
        return min_dist


def compute_entropy(img):
    """
    Compute the entropy of an image.
    
    Entropy measures the amount of information or "randomness" in the image.
    Higher entropy indicates more information content.
    
    Args:
        img (np.ndarray): Input image
        
    Returns:
        float: Entropy value of the image
    """
    # Input validation
    if img is None or img.size == 0:
        logger.warning("Cannot compute entropy on empty image")
        return 0.0
    
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
    Compute the mutual information between two images.
    
    Mutual information measures how much information one image provides about another.
    Higher values indicate better image alignment.
    
    Args:
        img1 (np.ndarray): First image
        img2 (np.ndarray): Second image
        
    Returns:
        float: Mutual information between the images
    """
    # Input validation
    if img1 is None or img2 is None or img1.size == 0 or img2.size == 0:
        logger.warning("Cannot compute mutual information on empty images")
        return 0.0
    
    # Ensure the images are normalized to 0-255
    if img1.dtype != np.uint8:
        img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img2.dtype != np.uint8:
        img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Ensure images have the same dimensions
    h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
    img1 = img1[:h, :w]
    img2 = img2[:h, :w]
    
    # Compute entropies
    H1 = compute_entropy(img1)
    H2 = compute_entropy(img2)
    
    # Compute joint histogram
    hist_2d, _, _ = np.histogram2d(
        img1.flatten(), img2.flatten(), 
        bins=256, 
        range=[[0, 256], [0, 256]]
    )
    
    # Normalize the joint histogram
    pxy = hist_2d / float(np.sum(hist_2d))
    pxy_non_zero = pxy[pxy > 0]
    
    # Compute joint entropy
    H12 = -np.sum(pxy_non_zero * np.log2(pxy_non_zero))
    
    # Compute mutual information
    MI = H1 + H2 - H12
    
    return MI