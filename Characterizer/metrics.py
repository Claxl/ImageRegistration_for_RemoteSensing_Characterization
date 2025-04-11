#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Metrics for evaluating image registration performance.

This module provides functions to compute various metrics to evaluate 
the quality of image registration, including:
- RMSE between transformation matrices
- RMSE between point sets
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
        return 1e4
    if isinstance(H_pred, list):
        H_pred = np.array(H_pred)
    try:
        # Verify matrix shapes
        if H_pred.shape != H_gt.shape:
            logger.warning(f"Matrix shape mismatch: {H_pred.shape} vs {H_gt.shape}")
            return None
        
        # Compute RMSE directly between matrix elements
        rmse = np.sqrt(np.mean((H_pred - H_gt) ** 2))
        if np.isnan(rmse):
            return 1e4
        return rmse
        
    except Exception as e:
        logger.error(f"Error computing matrix RMSE: {e}")
        return 1e4


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
