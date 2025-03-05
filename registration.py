#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Core image registration functions.
"""

import cv2
import numpy as np
import time
from metrics import compute_entropy, compute_mutual_information, compute_rmse_points
from detectors import process_rift
from utils import make_match_image

def process_image_pair(sar_img_path, opt_img_path, detector, matcher, ratio_thresh=0.7):
    """
    Processes a pair of images using the specified feature detection and matching method.
    
    This function implements the complete image registration pipeline:
    1. Load the images
    2. Extract keypoints and descriptors
    3. Match features
    4. Compute homography transformation
    5. Generate registered image
    
    Args:
        sar_img_path (str): Path to SAR image file
        opt_img_path (str): Path to optical image file
        detector: Feature detector object
        matcher: Feature matcher object
        ratio_thresh (float): Threshold for Lowe's ratio test
        
    Returns:
        tuple: Contains (NM, NCM, ratio, reg_time, registered_img, matches_img)
            - NM: Number of matches
            - NCM: Number of correct matches (inliers)
            - ratio: Ratio of NM to NCM
            - reg_time: Registration time in seconds
            - registered_img: Warped (registered) SAR image
            - matches_img: Visualization of matches
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
    if detector is None and matcher is None:
        # Process using RIFT
        print("Using RIFT algorithm for processing...")
        
        results = process_rift(sar_img, opt_img)
        
        # Return values in the same format as other methods
        return (results['NM'], 
                results['NCM'], 
                results['ratio'], 
                results['reg_time'], 
                results['registered_img'], 
                results['matches_img'])
    
    # Standard OpenCV-based processing
    # Extract keypoints and descriptors
    kp_sar, desc_sar = detector.detectAndCompute(sar_img, None)
    kp_opt, desc_opt = detector.detectAndCompute(opt_img, None)
    
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
    
    NM = len(good_matches)
    
    # Initialize output images
    registered_img = None
    matches_img = None
    
    # Compute homography if enough good matches are found
    if NM >= 4:
        src_pts = np.float32([kp_sar[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_opt[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0)
        
        if M is not None:
            mask = mask.ravel()
            NCM = int(np.sum(mask))  # number of correct (inlier) matches
            height, width = opt_img.shape
            registered_img = cv2.warpPerspective(sar_img, M, (width, height))
            matches_img = cv2.drawMatches(sar_img, kp_sar, opt_img, kp_opt, good_matches, None, 
                                          matchesMask=mask.tolist(), flags=2)
        else:
            NCM = 0
    else:
        NCM = 0

    ratio = NM / NCM if NCM != 0 else 0
    reg_time = time.time() - start_reg_time
    
    return NM, NCM, ratio, reg_time, registered_img, matches_img

def process_image_pair_with_gt(sar_img, opt_img, detector, matcher, landmarks_mov, landmarks_fix, ratio_thresh=0.7):
    """
    Processes a pair of images with ground truth landmarks.
    
    Args:
        sar_img (np.ndarray): SAR (moving) image
        opt_img (np.ndarray): Optical (fixed) image
        detector: Feature detector object
        matcher: Feature matcher object
        landmarks_mov (np.ndarray): Ground truth landmarks for SAR image
        landmarks_fix (np.ndarray): Ground truth landmarks for optical image
        ratio_thresh (float): Threshold for Lowe's ratio test
        
    Returns:
        dict: Dictionary containing registration results and evaluation metrics
    """
    from metrics import count_repeatable_points, count_ground_truth_matches, count_ground_truth_matches_array
    
    start_reg_time = time.time()
    
    # Check if using RIFT method (which has its own pipeline)
    if detector is None and matcher is None:
        # Process using RIFT
        results = process_rift(sar_img, opt_img)
        
        # Calculate additional metrics
        total_points = len(results['keypoints_sar']) + len(results['keypoints_opt'])
        repeatable_points = count_repeatable_points(
            results['keypoints_sar'], results['keypoints_opt'], 
            landmarks_fix, landmarks_mov
        )
        gt_matches = count_ground_truth_matches_array(
            results['keypoints_sar'], results['keypoints_opt'],
            landmarks_fix, landmarks_mov
        )
        
        # Add these metrics to the results
        results['total_points'] = total_points
        results['repeatable_points'] = repeatable_points
        results['gt_matches'] = gt_matches
        results['entropy_opt'] = compute_entropy(opt_img)
        
        if results['registered_img'] is not None:
            # Convert to grayscale for entropy/MI calculation if needed
            reg_img_gray = results['registered_img'].copy()
            if len(reg_img_gray.shape) > 2:
                reg_img_gray = cv2.cvtColor(reg_img_gray, cv2.COLOR_BGR2GRAY)
                
            results['entropy_reg'] = compute_entropy(reg_img_gray)
            results['mutual_information'] = compute_mutual_information(opt_img, reg_img_gray)
            
            # Transform ground truth landmarks using the homography
            if results['transformation_matrix'] is not None:
                h_landmarks_mov = np.ones((len(landmarks_mov), 3))
                h_landmarks_mov[:, :2] = landmarks_mov
                
                transformed_points = []
                for pt in h_landmarks_mov:
                    tp = np.dot(results['transformation_matrix'], pt)
                    transformed_points.append([tp[0]/tp[2], tp[1]/tp[2]])
                
                transformed_points = np.array(transformed_points)
                
                # Compute RMSE between transformed landmarks and ground truth
                rmse_landmarks = compute_rmse_points(transformed_points, landmarks_fix)
                results['rmse_landmarks'] = rmse_landmarks
        
        return results
    
    # Standard OpenCV-based processing
    # Extract keypoints and descriptors
    kp_sar, desc_sar = detector.detectAndCompute(sar_img, None)
    kp_opt, desc_opt = detector.detectAndCompute(opt_img, None)
    
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
    
    # Calculate metrics
    total_points = len(kp_sar) + len(kp_opt)
    repeatable_points = count_repeatable_points(kp_sar, kp_opt, landmarks_fix, landmarks_mov)
    gt_matches = count_ground_truth_matches(good_matches, kp_sar, kp_opt, landmarks_fix, landmarks_mov)
    
    NM = len(good_matches)
    
    # Initialize output variables
    registered_img = None
    matches_img = None
    rmse = None
    entropy_reg = None
    entropy_opt = compute_entropy(opt_img)
    mi = None
    transformation_matrix = None
    
    # Compute homography if enough good matches are found
    if NM >= 4:
        src_pts = np.float32([kp_sar[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_opt[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is not None:
            mask = mask.ravel()
            NCM = int(np.sum(mask))  # number of correct (inlier) matches
            
            height, width = opt_img.shape
            registered_img = cv2.warpPerspective(sar_img, M, (width, height))
            matches_img = cv2.drawMatches(sar_img, kp_sar, opt_img, kp_opt, good_matches, None, 
                                          matchesMask=mask.tolist(), flags=2)
            
            # Transform ground truth landmarks using the homography
            h_landmarks_mov = np.ones((len(landmarks_mov), 3))
            h_landmarks_mov[:, :2] = landmarks_mov
            
            transformed_points = []
            for pt in h_landmarks_mov:
                tp = np.dot(M, pt)
                transformed_points.append([tp[0]/tp[2], tp[1]/tp[2]])
            
            transformed_points = np.array(transformed_points)
            
            # Compute RMSE between transformed landmarks and ground truth
            rmse = compute_rmse_points(transformed_points, landmarks_fix)
            
            # Compute entropy and MI
            entropy_reg = compute_entropy(registered_img)
            mi = compute_mutual_information(opt_img, registered_img)
            
            transformation_matrix = M
        else:
            NCM = 0
    else:
        NCM = 0
    
    ratio = NM / NCM if NCM != 0 else 0
    reg_time = time.time() - start_reg_time
    
    return {
        'NM': NM,
        'NCM': NCM,
        'ratio': ratio,
        'reg_time': reg_time,
        'total_points': total_points,
        'repeatable_points': repeatable_points,
        'gt_matches': gt_matches,
        'rmse': rmse,
        'entropy_opt': entropy_opt,
        'entropy_reg': entropy_reg,
        'mutual_information': mi,
        'transformation_matrix': transformation_matrix,
        'registered_img': registered_img,
        'matches_img': matches_img,
        'keypoints_sar': kp_sar,
        'keypoints_opt': kp_opt,
        'good_matches': good_matches
    }