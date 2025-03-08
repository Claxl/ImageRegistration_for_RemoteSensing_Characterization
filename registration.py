#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Core image registration functions.
"""

import cv2
import numpy as np
import time
from metrics import compute_rmse_matrices, compute_rmse_points
from detectors import process_rift, process_lghd,process_sarsift
from utils import make_match_image
def process_image_pair_with_gt(sar_img_path, opt_img_path, detector, matcher, landmarks_mov, landmarks_fix, transform_gt=None, ratio_thresh=0.7, method=""):
    """
    Processes a pair of images with ground truth landmarks and optional transformation matrix.
    
    Args:
        sar_img (np.ndarray): SAR (moving) image
        opt_img (np.ndarray): Optical (fixed) image
        detector: Feature detector object
        matcher: Feature matcher object
        landmarks_mov (np.ndarray): Ground truth landmarks for SAR image
        landmarks_fix (np.ndarray): Ground truth landmarks for optical image
        transform_gt (np.ndarray, optional): Ground truth transformation matrix
        ratio_thresh (float): Threshold for Lowe's ratio test
        method (str): Name of the method (needed for LGHD and RIFT)
        
    Returns:
        dict: Dictionary containing registration results and evaluation metrics
    """
    start_reg_time = time.time()
            # Load images
    sar_img = cv2.imread(sar_img_path, cv2.IMREAD_GRAYSCALE)
    opt_img = cv2.imread(opt_img_path, cv2.IMREAD_GRAYSCALE)
        
    if sar_img is None or opt_img is None:
            print(f"Error loading images: {sar_img_path}, {opt_img_path}")
    # Check if using RIFT method (which has its own pipeline)
    if detector is None and matcher is None and method.upper() == "RIFT":
        # Process using RIFT
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
        
        # Return results with the metrics you specifically want
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
            'mosaic_img': results['mosaic_img'] if 'mosaic_img' in results else None
        }
    
    # Check if using LGHD method (which has its own pipeline)
    elif detector is None and matcher is None and method.upper() == "LGHD":
        # Process using LGHD
        results = process_lghd(sar_img_path, opt_img_path)
        
        # Calculate matrix RMSE if ground truth is available
        matrix_rmse = None
        if transform_gt is not None and results['transformation_matrix'] is not None:
            matrix_rmse = compute_rmse_matrices(results['transformation_matrix'], transform_gt)
        
        # Return results with the metrics you specifically want
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
            'mosaic_img': results['mosaic_img'] if 'mosaic_img' in results else None
        }
    # Check if we're using SAR-SIFT
    if hasattr(detector, 'sar_sift'):
        # Process using SAR-SIFT
        results = process_sarsift(sar_img, opt_img)
        
        # Calculate additional metrics
        num_keypoints_sar = len(results['keypoints_sar'])
        num_keypoints_opt = len(results['keypoints_opt'])
        num_matches = results['NM']
        num_inliers = results['NCM']
        
        # Calculate matrix RMSE if ground truth is available
        matrix_rmse = None
        if transform_gt is not None and results['transformation_matrix'] is not None:
            matrix_rmse = compute_rmse_matrices(results['transformation_matrix'], transform_gt)
        
        # Return results with the metrics you specifically want
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
            'mosaic_img': results['mosaic_img']
        }
    
    # Standard OpenCV-based processing
    # Extract keypoints and descriptors
    kp_sar, desc_sar = detector.detectAndCompute(sar_img, None)
    kp_opt, desc_opt = detector.detectAndCompute(opt_img, None)
    
    num_keypoints_sar = len(kp_sar)
    num_keypoints_opt = len(kp_opt)
    
    if desc_sar is None or desc_opt is None:
        raise ValueError("No descriptors found in one or both images.")
    
    # Perform matching with k=2 and apply Lowe's ratio test
    matches = matcher.knnMatch(desc_sar, desc_opt, k=2)
    good_matches = []
    for m_n in matches:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    
    num_matches = len(good_matches)
    
    # Initialize output variables
    registered_img = None
    matches_img = None
    transformation_matrix = None
    num_inliers = 0
    matrix_rmse = None
    
    # Compute homography if enough good matches are found
    if num_matches >= 4:
        src_pts = np.float32([kp_sar[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_opt[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is not None:
            mask = mask.ravel()
            num_inliers = int(np.sum(mask))  # number of correct (inlier) matches
            
            height, width = opt_img.shape
            transformation_matrix = M
            
            # Calculate matrix RMSE if ground truth is available
            if transform_gt is not None:
                matrix_rmse = compute_rmse_matrices(M, transform_gt)
    
    execution_time = time.time() - start_reg_time
    
    # Return results with the metrics you specifically want
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

def process_image_pair(sar_img_path, opt_img_path, detector, matcher, ratio_thresh=0.7, method=""):
    """
    Processes a pair of images using the specified feature detection and matching method.
    
    Args:
        sar_img_path (str): Path to SAR image file
        opt_img_path (str): Path to optical image file
        detector: Feature detector object
        matcher: Feature matcher object
        ratio_thresh (float): Threshold for Lowe's ratio test
        method (str): Name of the method (needed for LGHD and RIFT)
        
    Returns:
        tuple: Contains (num_keypoints_sar, num_keypoints_opt, num_matches, num_inliers, 
               transformation_matrix, execution_time, registered_img, matches_img)
    """
    start_reg_time = time.time()
    
    # Load images in grayscale
    sar_img = cv2.imread(sar_img_path, cv2.IMREAD_GRAYSCALE)
    opt_img = cv2.imread(opt_img_path, cv2.IMREAD_GRAYSCALE)
    
    if sar_img is None:
        raise IOError(f"Error loading SAR image: {sar_img_path}")
    if opt_img is None:
        raise IOError(f"Error loading optical image: {opt_img_path}")
    
    # Check if using RIFT method (which has its own pipeline)
    if detector is None and matcher is None and method.upper() == "RIFT":
        # Process using RIFT
        print("Using RIFT algorithm for processing...")
        
        results = process_rift(sar_img, opt_img)
        
        # Extract required metrics
        num_keypoints_sar = len(results['keypoints_sar'])
        num_keypoints_opt = len(results['keypoints_opt'])
        num_matches = results['NM'] 
        num_inliers = results['NCM']
        execution_time = results['reg_time']
        
        return (num_keypoints_sar, num_keypoints_opt, num_matches, num_inliers, 
                results['transformation_matrix'], execution_time, results['registered_img'], results['matches_img'])
    
    # Check if using LGHD method (which has its own pipeline)
    elif detector is None and matcher is None and method.upper() == "LGHD":
        # Process using LGHD
        print("Using LGHD algorithm for processing...")
        
        results = process_lghd(sar_img, opt_img)
        
        # Extract required metrics
        num_keypoints_sar = len(results['keypoints_sar'])
        num_keypoints_opt = len(results['keypoints_opt'])
        num_matches = results['NM'] 
        num_inliers = results['NCM']
        execution_time = results['reg_time']
        
        return (num_keypoints_sar, num_keypoints_opt, num_matches, num_inliers, 
                results['transformation_matrix'], execution_time, results['registered_img'], results['matches_img'])
    if hasattr(detector, 'sar_sift'):
        # Process using SAR-SIFT
        print("Using SAR-SIFT algorithm for processing...")
        
        results = process_sarsift(sar_img, opt_img)
        
        # Extract required metrics
        num_keypoints_sar = len(results['keypoints_sar'])
        num_keypoints_opt = len(results['keypoints_opt'])
        num_matches = results['NM'] 
        num_inliers = results['NCM']
        execution_time = results['reg_time']
        
        return (num_keypoints_sar, num_keypoints_opt, num_matches, num_inliers, 
                results['rmse'], execution_time, results['registered_img'], results['matches_img'])
    # Standard OpenCV-based processing
    # Extract keypoints and descriptors
    print("Using OpenCV-based algorithm for processing...")
    try:
        kp_sar, desc_sar = detector.detectAndCompute(sar_img, None)
        kp_opt, desc_opt = detector.detectAndCompute(opt_img, None)
    except cv2.error as e:
        raise cv2.error(f"Error in feature extraction: {e}")
    num_keypoints_sar = len(kp_sar)
    num_keypoints_opt = len(kp_opt)
    
    if desc_sar is None or desc_opt is None:
        raise ValueError("No descriptors found in one or both images.")
    
    # Perform matching with k=2 and apply Lowe's ratio test
    matches = matcher.knnMatch(desc_sar, desc_opt, k=2)
    good_matches = []
    for m_n in matches:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    
    num_matches = len(good_matches)
    
    # Initialize output images
    registered_img = None
    matches_img = None
    transformation_matrix = None
    num_inliers = 0
    
    # Compute homography if enough good matches are found
    if num_matches >= 4:
        src_pts = np.float32([kp_sar[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_opt[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0)
        
        if M is not None:
            mask = mask.ravel()
            num_inliers = int(np.sum(mask))  # number of correct (inlier) matches
            height, width = opt_img.shape
            registered_img = cv2.warpPerspective(sar_img, M, (width, height))
            matches_img = cv2.drawMatches(sar_img, kp_sar, opt_img, kp_opt, good_matches, None, 
                                          matchesMask=mask.tolist(), flags=2)
            transformation_matrix = M
    
    execution_time = time.time() - start_reg_time
    
    return (num_keypoints_sar, num_keypoints_opt, num_matches, num_inliers, 
            transformation_matrix, execution_time, registered_img, matches_img)