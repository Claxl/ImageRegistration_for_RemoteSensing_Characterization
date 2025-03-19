#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Core image registration functions.

This module provides the core functionality for registering image pairs
using different feature detection methods, with support for ground truth
evaluation when available.
"""

import cv2
import numpy as np
import time
import logging
from .metrics import compute_rmse_matrices, compute_rmse_points
from .detectors import process_rift, process_lghd, process_sarsift
from .utils import make_match_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_image_pair_with_gt(sar_img_path, opt_img_path, detector, matcher, 
                              landmarks_mov, landmarks_fix, transform_gt=None, 
                              ratio_thresh=0.7, method=""):
    """
    Process an image pair with ground truth data for evaluation.
    
    Args:
        sar_img_path (str): Path to SAR (moving) image
        opt_img_path (str): Path to optical (fixed) image
        detector: Feature detector object
        matcher: Feature matcher object
        landmarks_mov (np.ndarray): Ground truth landmarks for SAR image
        landmarks_fix (np.ndarray): Ground truth landmarks for optical image
        transform_gt (np.ndarray, optional): Ground truth transformation matrix
        ratio_thresh (float): Threshold for Lowe's ratio test
        method (str): Name of the method (needed for RIFT, LGHD, SAR-SIFT)
        
    Returns:
        dict: Registration results and evaluation metrics
    """
    start_reg_time = time.time()
    
    # Load images
    try:
        sar_img = cv2.imread(sar_img_path, cv2.IMREAD_GRAYSCALE)
        opt_img = cv2.imread(opt_img_path, cv2.IMREAD_GRAYSCALE)
        
        if sar_img is None or opt_img is None:
            logger.error(f"Error loading images: {sar_img_path}, {opt_img_path}")
            raise IOError(f"Failed to load images: {sar_img_path}, {opt_img_path}")
            
    except Exception as e:
        logger.error(f"Error loading images: {str(e)}")
        raise
    
    # Process based on method type
    method = method.upper()
    
    if method == "RIFT" and detector is None and matcher is None:
        return _process_with_rift(sar_img, opt_img, transform_gt, start_reg_time,landmarks_mov, landmarks_fix,)
        
    elif method == "LGHD" and detector is None and matcher is None:
        return _process_with_lghd(sar_img_path, opt_img_path, transform_gt, start_reg_time,landmarks_mov, landmarks_fix,)
        
    elif hasattr(detector, 'sar_sift'):
        return _process_with_sarsift(sar_img, opt_img, transform_gt, start_reg_time,landmarks_mov, landmarks_fix,)
    
    else:
        return _process_with_opencv(
            sar_img, opt_img, detector, matcher, 
            transform_gt, ratio_thresh, start_reg_time,landmarks_mov, landmarks_fix,
        )


def _process_with_rift(sar_img, opt_img, transform_gt, start_time,landmarks_mov, landmarks_fix,):
    """Process images using RIFT algorithm."""
    logger.info("Processing with RIFT algorithm")
    results = process_rift(sar_img, opt_img)
    
    # Calculate additional metrics
    num_keypoints_sar = len(results['keypoints_sar'])
    num_keypoints_opt = len(results['keypoints_opt'])
    num_matches = results['NM']
    num_inliers = results['NCM']
    
    # Calculate matrix RMSE if ground truth is available
    matrix_rmse = None
    if transform_gt is not None and results['transformation_matrix'] is not None:
        matrix_rmse = compute_rmse_matrices(results['transformation_matrix'], transform_gt)
    
    return {
        'num_keypoints_sar': num_keypoints_sar,
        'num_keypoints_opt': num_keypoints_opt,
        'num_matches': num_matches,
        'num_inliers': num_inliers,
        'matrix_rmse': matrix_rmse,
        'execution_time': results['reg_time'],
        'transformation_matrix': results['transformation_matrix'],
        'registered_img': results['registered_img'],
        'matches_img': results['matches_img'],
        'mosaic_img': results.get('mosaic_img')
    }


def _process_with_lghd(sar_img_path, opt_img_path, transform_gt, start_time,landmarks_mov, landmarks_fix,):
    """Process images using LGHD algorithm."""
    logger.info("Processing with LGHD algorithm")
    results = process_lghd(sar_img_path, opt_img_path)
    
    # Calculate matrix RMSE if ground truth is available
    matrix_rmse = None
    if transform_gt is not None and results['transformation_matrix'] is not None:
        matrix_rmse = compute_rmse_matrices(results['transformation_matrix'], transform_gt)
    if landmarks_mov is not None and landmarks_fix is not None:
            # Transform the SAR (moving) landmarks using estimated homography
            landmarks_mov_homogeneous = np.hstack((landmarks_mov, np.ones((landmarks_mov.shape[0], 1))))
            transformed_landmarks = np.dot(landmarks_mov_homogeneous, results['transformation_matrix'].T)
            # Apply perspective division
            transformed_landmarks = transformed_landmarks[:, :2] / transformed_landmarks[:, 2:3]
                
            # Calculate RMSE between transformed landmarks and ground truth landmarks
            points_rmse = compute_rmse_points(transformed_landmarks, landmarks_fix)
            print(f"Point-based RMSE: {points_rmse:.2f} pixels")

    return {
        'num_keypoints_sar': len(results['keypoints_sar']),
        'num_keypoints_opt': len(results['keypoints_opt']),
        'num_matches': results['NM'],
        'num_inliers': results['NCM'],
        'matrix_rmse': matrix_rmse,
        'execution_time': results['reg_time'],
        'transformation_matrix': results['transformation_matrix'],
        'points_rmse': points_rmse,  # Add this line to include points RMSE in results
        'registered_img': results['registered_img'],
        'matches_img': results['matches_img'],
        'mosaic_img': results.get('mosaic_img')
    }


def _process_with_sarsift(sar_img, opt_img, transform_gt, start_time,landmarks_mov, landmarks_fix,):
    """Process images using SAR-SIFT algorithm."""
    logger.info("Processing with SAR-SIFT algorithm")
    results = process_sarsift(sar_img, opt_img)
    
    # Calculate matrix RMSE if ground truth is available
    matrix_rmse = None
    if transform_gt is not None and results['transformation_matrix'] is not None:
        matrix_rmse = compute_rmse_matrices(results['transformation_matrix'], transform_gt)
    
    return {
        'num_keypoints_sar': len(results['keypoints_sar']),
        'num_keypoints_opt': len(results['keypoints_opt']),
        'num_matches': results['NM'],
        'num_inliers': results['NCM'],
        'matrix_rmse': matrix_rmse,
        'execution_time': results['reg_time'],
        'transformation_matrix': results['transformation_matrix'],
        'registered_img': results['registered_img'],
        'matches_img': results['matches_img'],
        'mosaic_img': results.get('mosaic_img')
    }


def _process_with_opencv(sar_img, opt_img, detector, matcher, transform_gt, ratio_thresh, start_time,landmarks_mov, landmarks_fix,):
    """Process images using standard OpenCV detectors and matchers."""
    logger.info("Processing with OpenCV-based algorithm")
    
    try:
        # Extract keypoints and descriptors
        kp_sar, desc_sar = detector.detectAndCompute(sar_img, None)
        kp_opt, desc_opt = detector.detectAndCompute(opt_img, None)
        
        num_keypoints_sar = len(kp_sar)
        num_keypoints_opt = len(kp_opt)
        
        if desc_sar is None or desc_opt is None:
            raise ValueError("No descriptors found in one or both images.")
            
        # Match features and apply ratio test
        good_matches = _match_features(matcher, desc_sar, desc_opt, ratio_thresh)
        num_matches = len(good_matches)
        
        # Initialize output variables
        transformation_matrix = None
        num_inliers = 0
        matrix_rmse = None
        
        # Compute homography if enough good matches
        if num_matches >= 4:
            transformation_matrix, mask, num_inliers = _compute_homography(
                kp_sar, kp_opt, good_matches
            )
            
            # Calculate matrix RMSE if ground truth is available
            if transform_gt is not None and transformation_matrix is not None:
                matrix_rmse = compute_rmse_matrices(transformation_matrix, transform_gt)
        
        execution_time = time.time() - start_time
        
        return {
            'num_keypoints_sar': num_keypoints_sar,
            'num_keypoints_opt': num_keypoints_opt,
            'num_matches': num_matches,
            'num_inliers': num_inliers,
            'matrix_rmse': matrix_rmse,
            'execution_time': execution_time,
            'transformation_matrix': transformation_matrix,
            'registered_img': None,
            'matches_img': None,
            'mosaic_img': None
        }
        
    except Exception as e:
        logger.error(f"Error in OpenCV processing: {str(e)}")
        raise


def _match_features(matcher, desc1, desc2, ratio_thresh=0.7):
    """
    Match features and apply Lowe's ratio test.
    
    Args:
        matcher: OpenCV feature matcher
        desc1, desc2: Feature descriptors
        ratio_thresh: Threshold for Lowe's ratio test
        
    Returns:
        list: Good matches that pass the ratio test
    """
    matches = matcher.knnMatch(desc1, desc2, k=2)
    good_matches = []
    
    for m_n in matches:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
            
    return good_matches


def _compute_homography(kp1, kp2, good_matches, ransac_threshold=5.0):
    """
    Compute homography matrix from matched keypoints using RANSAC.
    
    Args:
        kp1, kp2: Keypoints from two images
        good_matches: List of good feature matches
        ransac_threshold: RANSAC threshold for homography estimation
        
    Returns:
        tuple: (transformation_matrix, mask, num_inliers)
    """
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
    
    if M is None:
        return None, None, 0
        
    mask = mask.ravel()
    num_inliers = int(np.sum(mask))
    
    return M, mask, num_inliers


def process_image_pair(sar_img_path, opt_img_path, detector, matcher, ratio_thresh=0.7, method=""):
    """
    Process an image pair without ground truth data.
    
    Args:
        sar_img_path (str): Path to SAR image
        opt_img_path (str): Path to optical image
        detector: Feature detector object
        matcher: Feature matcher object
        ratio_thresh (float): Threshold for Lowe's ratio test
        method (str): Name of the method
        
    Returns:
        tuple: Processing results including keypoints, matches, transformation, etc.
    """
    start_reg_time = time.time()
    
    try:
        # Load images
        sar_img = cv2.imread(sar_img_path, cv2.IMREAD_GRAYSCALE)
        opt_img = cv2.imread(opt_img_path, cv2.IMREAD_GRAYSCALE)
        
        if sar_img is None:
            raise IOError(f"Error loading SAR image: {sar_img_path}")
        if opt_img is None:
            raise IOError(f"Error loading optical image: {opt_img_path}")
            
        method = method.upper()
        
        # Process using method-specific pipeline
        if method == "RIFT" and detector is None and matcher is None:
            return _process_pair_rift(sar_img, opt_img, start_reg_time)
            
        elif method == "LGHD" and detector is None and matcher is None:
            return _process_pair_lghd(sar_img, opt_img, start_reg_time)
            
        elif hasattr(detector, 'sar_sift'):
            return _process_pair_sarsift(sar_img, opt_img, start_reg_time)
            
        else:
            return _process_pair_opencv(
                sar_img, opt_img, detector, matcher, 
                ratio_thresh, start_reg_time
            )
            
    except Exception as e:
        logger.error(f"Error processing image pair: {str(e)}")
        raise


def _process_pair_rift(sar_img, opt_img, start_time):
    """Process an image pair using RIFT algorithm."""
    logger.info("Using RIFT algorithm for processing...")
    
    results = process_rift(sar_img, opt_img)
    
    # Extract required metrics
    num_keypoints_sar = len(results['keypoints_sar'])
    num_keypoints_opt = len(results['keypoints_opt'])
    num_matches = results['NM'] 
    num_inliers = results['NCM']
    execution_time = results['reg_time']
    
    return (
        num_keypoints_sar, num_keypoints_opt, num_matches, num_inliers, 
        results['transformation_matrix'], execution_time, 
        results['registered_img'], results['matches_img']
    )


def _process_pair_lghd(sar_img, opt_img, start_time):
    """Process an image pair using LGHD algorithm."""
    logger.info("Using LGHD algorithm for processing...")
    
    results = process_lghd(sar_img, opt_img)
    
    # Extract required metrics
    num_keypoints_sar = len(results['keypoints_sar'])
    num_keypoints_opt = len(results['keypoints_opt'])
    num_matches = results['NM'] 
    num_inliers = results['NCM']
    execution_time = results['reg_time']
    
    return (
        num_keypoints_sar, num_keypoints_opt, num_matches, num_inliers, 
        results['transformation_matrix'], execution_time, 
        results['registered_img'], results['matches_img']
    )


def _process_pair_sarsift(sar_img, opt_img, start_time):
    """Process an image pair using SAR-SIFT algorithm."""
    logger.info("Using SAR-SIFT algorithm for processing...")
    
    results = process_sarsift(sar_img, opt_img)
    
    # Extract required metrics
    num_keypoints_sar = len(results['keypoints_sar'])
    num_keypoints_opt = len(results['keypoints_opt'])
    num_matches = results['NM'] 
    num_inliers = results['NCM']
    execution_time = results['reg_time']
    
    return (
        num_keypoints_sar, num_keypoints_opt, num_matches, num_inliers, 
        results['rmse'], execution_time, 
        results['registered_img'], results['matches_img']
    )


def _process_pair_opencv(sar_img, opt_img, detector, matcher, ratio_thresh, start_time):
    """Process an image pair using OpenCV detector and matcher."""
    logger.info("Using OpenCV-based algorithm for processing...")
    
    try:
        # Extract keypoints and descriptors
        kp_sar, desc_sar = detector.detectAndCompute(sar_img, None)
        kp_opt, desc_opt = detector.detectAndCompute(opt_img, None)
        
    except cv2.error as e:
        raise cv2.error(f"Error in feature extraction: {e}")
        
    num_keypoints_sar = len(kp_sar)
    num_keypoints_opt = len(kp_opt)
    
    if desc_sar is None or desc_opt is None:
        raise ValueError("No descriptors found in one or both images.")
    
    # Match features
    good_matches = _match_features(matcher, desc_sar, desc_opt, ratio_thresh)
    num_matches = len(good_matches)
    
    # Initialize output variables
    registered_img = None
    matches_img = None
    transformation_matrix = None
    num_inliers = 0
    
    # Compute homography if enough matches
    if num_matches >= 4:
        transformation_matrix, mask, num_inliers = _compute_homography_with_visualization(
            sar_img, opt_img, kp_sar, kp_opt, good_matches
        )
        
        # Create visualizations if homography was found
        if transformation_matrix is not None:
            registered_img = _create_registered_image(sar_img, opt_img, transformation_matrix)
            matches_img = _create_matches_image(sar_img, opt_img, kp_sar, kp_opt, good_matches, mask)
    
    execution_time = time.time() - start_time
    
    return (
        num_keypoints_sar, num_keypoints_opt, num_matches, num_inliers,
        transformation_matrix, execution_time, registered_img, matches_img
    )


def _compute_homography_with_visualization(img1, img2, kp1, kp2, good_matches, ransac_threshold=2.0):
    """
    Compute homography and prepare for visualization.
    
    Args:
        img1, img2: Input images
        kp1, kp2: Keypoints from two images
        good_matches: List of good feature matches
        ransac_threshold: RANSAC threshold
        
    Returns:
        tuple: (transformation_matrix, mask, num_inliers)
    """
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
    
    if M is None:
        return None, None, 0
        
    mask = mask.ravel()
    num_inliers = int(np.sum(mask))
    
    return M, mask, num_inliers


def _create_registered_image(src_img, dst_img, transformation_matrix):
    """Create registered image by applying transformation."""
    height, width = dst_img.shape
    return cv2.warpPerspective(src_img, transformation_matrix, (width, height))


def _create_matches_image(img1, img2, kp1, kp2, matches, mask):
    """Create visualization of matches between images."""
    return cv2.drawMatches(
        img1, kp1, img2, kp2, matches, None, 
        matchesMask=mask.tolist(), flags=2
    )