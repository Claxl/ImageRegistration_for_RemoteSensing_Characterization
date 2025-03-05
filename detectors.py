#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature detectors and matchers for image registration.
"""

import cv2
import numpy as np

# Check if RIFT modules are available
try:
    from rift2.FeatureDetection import FeatureDetection
    from rift2.kptsOrientation import kptsOrientation
    from rift2.FeatureDescribe import FeatureDescribe
   # from rift2.FSC import FSC
   # from rift2.image_fusion import image_fusion
    RIFT_AVAILABLE = True
    print("RIFT2 module is available and will be used if selected.")
except ImportError:
    RIFT_AVAILABLE = False
    print("RIFT2 modules not available. RIFT method will be skipped if requested.")

def create_detector_and_matcher(method):
    """
    Given a method name ('SIFT', 'SURF', 'ORB', 'AKAZE', or 'RIFT'), returns a tuple (detector, matcher)
    properly configured for that method.
    
    Args:
        method (str): The feature detection and matching method to use
        
    Returns:
        tuple: A tuple containing (detector, matcher) objects configured for the specified method
        
    Raises:
        ValueError: If the method is not recognized or not available
        AttributeError: If SURF is requested but not available
    """
    if method.upper() == "SIFT":
        detector = cv2.SIFT_create()
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    elif method.upper() == "SURF":
        try:
            detector = cv2.xfeatures2d.SURF_create()
        except AttributeError:
            raise AttributeError("SURF is not available. Please install opencv-contrib-python.")
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    elif method.upper() == "ORB":
        detector = cv2.ORB_create()
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    elif method.upper() == "AKAZE":
        detector = cv2.AKAZE_create()
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    elif method.upper() == "RIFT":
        if not RIFT_AVAILABLE:
            raise ValueError("RIFT method is not available. Please install the RIFT2 package.")
        # For RIFT, we'll use None for both detector and matcher since RIFT has its own pipeline
        # The actual RIFT processing will be handled separately
        detector = None
        matcher = None
    else:
        raise ValueError(f"Method {method} not recognized.")
    return detector, matcher

def process_rift(sar_img, opt_img):
    """
    Process a pair of images using the RIFT2 algorithm with OpenCV-based transformation.
    
    This function handles the complete RIFT2 pipeline:
    1. Feature detection on both images
    2. Orientation assignment to keypoints
    3. Feature description
    4. Feature matching
    5. Removing duplicate matches
    6. Estimating transformation using OpenCV's findHomography instead of FSC
    7. Registering the images
    
    Args:
        sar_img (np.ndarray): SAR image (grayscale or BGR)
        opt_img (np.ndarray): Optical image (grayscale or BGR)
        
    Returns:
        dict: Dictionary containing processing results, including:
            - NM: Number of matches
            - NCM: Number of correct matches (inliers)
            - ratio: Ratio of NM to NCM
            - transformation_matrix: Estimated homography matrix
            - registered_img: The warped (registered) SAR image
            - mosaic_img: Mosaic of the registered images
            - matches_img: Visualization of the matches
            - rmse: Root Mean Square Error of the registration
            - execution_time: Time taken for processing
    """
    if not RIFT_AVAILABLE:
        raise ImportError("RIFT2 modules are not available. Please install the RIFT2 package.")
    
    import time
    import numpy as np
    import cv2
    from utils import make_match_image
    
    start_time = time.time()
    
    # Convert to BGR if grayscale (RIFT2 expects BGR)
    if sar_img.ndim == 2:
        sar_img_bgr = cv2.cvtColor(sar_img, cv2.COLOR_GRAY2BGR)
    else:
        sar_img_bgr = sar_img.copy()
        
    if opt_img.ndim == 2:
        opt_img_bgr = cv2.cvtColor(opt_img, cv2.COLOR_GRAY2BGR)
    else:
        opt_img_bgr = opt_img.copy()
    
    # Feature detection
    key1, m1, eo1 = FeatureDetection(sar_img_bgr, 4, 6, 5000)
    key2, m2, eo2 = FeatureDetection(opt_img_bgr, 4, 6, 5000)
    
    # Orientation assignment
    kpts1 = kptsOrientation(key1, m1, True, 96)
    kpts2 = kptsOrientation(key2, m2, True, 96)
    
    # Feature description
    des1 = FeatureDescribe(sar_img_bgr, eo1, kpts1, 96, 6, 6)
    des2 = FeatureDescribe(opt_img_bgr, eo2, kpts2, 96, 6, 6)
    des1 = des1.T  # Shape: (numKeypoints1, descriptorDimension)
    des2 = des2.T  # Shape: (numKeypoints2, descriptorDimension)
    
    # Feature matching
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1.astype(np.float32),
                          des2.astype(np.float32),
                          k=2)
    
    # Apply ratio test
    good_matches = []
    ratio_thresh = 1.0  # Take all matches (filtering done in RANSAC)
    for m in matches:
        if len(m) == 2:
            if m[0].distance < ratio_thresh * m[1].distance:
                good_matches.append(m[0])
        elif len(m) == 1:
            good_matches.append(m[0])
    
    # Extract matched keypoints
    matched_pts1 = []
    matched_pts2 = []
    for m in good_matches:
        matched_pts1.append(kpts1[:2, m.queryIdx])  # (x, y)
        matched_pts2.append(kpts2[:2, m.trainIdx])
    
    matched_pts1 = np.array(matched_pts1)
    matched_pts2 = np.array(matched_pts2)
    
    # Remove duplicate matches
    matched_pts2_unique, idxs = np.unique(matched_pts2, axis=0, return_index=True)
    matched_pts1_unique = matched_pts1[idxs]
    
    # Check if we have enough matches
    if len(matched_pts1_unique) < 4:
        print("Not enough unique matches for homography estimation.")
        return {
            'NM': 0,
            'NCM': 0,
            'ratio': 0,
            'reg_time': time.time() - start_time,
            'rmse': float('inf'),
            'transformation_matrix': np.eye(3),
            'registered_img': sar_img_bgr.copy(),
            'mosaic_img': None,
            'matches_img': None,
            'keypoints_sar': None,
            'keypoints_opt': None,
            'good_matches': None
        }
    
    # MODIFIED: Use OpenCV's findHomography instead of FSC
    ransac_threshold = 3.0
    H, mask = cv2.findHomography(
        matched_pts2_unique,  # Source points (opt_img)
        matched_pts1_unique,  # Destination points (sar_img)
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold
    )
    
    # If homography estimation failed, return identity
    if H is None:
        print("Homography estimation failed!")
        H = np.eye(3)
        mask = np.zeros(len(matched_pts1_unique), dtype=np.uint8)
    
    # Extract inlier points
    inlier_idxs = np.where(mask.ravel() == 1)[0]
    consensus_pts1 = matched_pts1_unique[inlier_idxs]
    consensus_pts2 = matched_pts2_unique[inlier_idxs]
    
    # Calculate RMSE for inliers
    rmse = 0.0
    if len(consensus_pts1) > 0:
        # For perspective transform, we need to apply perspective division
        ones = np.ones((len(consensus_pts2), 1))
        pts_homogeneous = np.hstack((consensus_pts2, ones))
        transformed = np.dot(H, pts_homogeneous.T).T
        transformed = transformed[:, :2] / transformed[:, 2:]
        errors = transformed - consensus_pts1
        rmse = np.sqrt(np.mean(np.sum(errors**2, axis=1)))
    
    # MODIFIED: Simplified image fusion with perspective transform
    def simplified_image_fusion(image1, image2, H):
        """
        Perform simplified image fusion using perspective transform.
        """
        # Get image dimensions
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]
        
        # Create a canvas large enough to hold both images
        output_size = (3*w1, 3*h1)
        
        # Create the offset transformation to center the result
        offset = np.array([[1, 0, w1], [0, 1, h1], [0, 0, 1]], dtype=np.float64)
        
        # Combine transformations
        final_transform = offset @ H
        
        # Warp images onto the canvas
        warped1 = cv2.warpPerspective(image1, offset, output_size)
        warped2 = cv2.warpPerspective(image2, final_transform, output_size)
        
        # Create fusion image
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
    
    # Generate registered and mosaic images using simplified approach
    registered_img, mosaic_img = simplified_image_fusion(sar_img_bgr, opt_img_bgr, H)
    
    # Create match visualization
    matches_img = make_match_image(sar_img_bgr, opt_img_bgr, consensus_pts1, consensus_pts2)
    
    # Calculate statistics
    NM = matched_pts1_unique.shape[0]  # Number of matches after removing duplicates
    NCM = consensus_pts1.shape[0]      # Number of consensus (inlier) matches
    ratio = NCM / NM if NM != 0 else 0
    execution_time = time.time() - start_time
    
    # Return results as a dictionary
    return {
        'NM': NM,
        'NCM': NCM,
        'ratio': ratio,
        'reg_time': execution_time,
        'rmse': rmse,
        'transformation_matrix': H,
        'registered_img': registered_img,
        'mosaic_img': mosaic_img,
        'matches_img': matches_img,
        'keypoints_sar': matched_pts1_unique,
        'keypoints_opt': matched_pts2_unique,
        'good_matches': consensus_pts1
    }