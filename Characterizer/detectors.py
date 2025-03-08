#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature detectors and matchers for image registration.
"""

import cv2
import numpy as np
import time

# Check if RIFT modules are available
try:
    from rift2.FeatureDetection import FeatureDetection
    from rift2.kptsOrientation import kptsOrientation
    from rift2.FeatureDescribe import FeatureDescribe
    RIFT_AVAILABLE = True
    print("RIFT2 module is available and will be used if selected.")
except ImportError as e:
    RIFT_AVAILABLE = False
    print("RIFT2 modules not available. RIFT method will be skipped if requested. Exception: {e}")
try:
    from LGHD.feature_descriptor import FeatureDescriptor
    from LGHD.LGHDRegistration import LGHDRegistration
    LGHD_AVAILABLE = True
    print("LGHD module is available and will be used if selected.")
except ImportError as e:
    LGHD_AVAILABLE = False
    print(f"LGHD modules not available. LGHD method will be skipped if requested. Exception: {e}")

# Check if SAR-SIFT is available
try:
    from SAR_Sift.sar_sift import SarSift, Keypoint as SARKeypoint
    SARSIFT_AVAILABLE = True
    print("SAR-SIFT module is available and will be used if selected.")
except ImportError:
    SARSIFT_AVAILABLE = False
    print("SAR-SIFT module not available. SAR-SIFT method will be skipped if requested.")


class SARSIFTDetector:
    """
    Wrapper class for SAR-SIFT to match OpenCV detector interface.
    This allows the SAR-SIFT detector to be used with the existing framework.
    """
    def __init__(self, nFeatures=5000, Mmax=8, sigma=2.0, ratio=2.0**(1.0/3.0), threshold=0.8/5, d=0.04):
        """
        Initialize the SAR-SIFT detector with default parameters.
        
        Args:
            nFeatures (int): Maximum number of features to detect
            Mmax (int): Number of scale layers
            sigma (float): Initial scale
            ratio (float): Scale factor between layers
            threshold (float): Harris function response threshold
            d (float): Parameter for sar_harris function
        """
        self.sar_sift = SarSift(nFeatures, Mmax, sigma, ratio, threshold, d)
        
    def _convert_to_cv_keypoints(self, sar_keypoints):
        """Convert SAR-SIFT keypoints to OpenCV keypoints."""
        cv_keypoints = []
        for kp in sar_keypoints:
            cv_kp = cv2.KeyPoint(
                x=kp.pt[0], 
                y=kp.pt[1],
                size=kp.size,
                angle=kp.angle,
                response=kp.response,
                octave=kp.octave
            )
            cv_keypoints.append(cv_kp)
        return cv_keypoints


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
        cv2.setUseOptimized(False)  # Evita problemi di accesso a memoria su FPGA
        cv2.setNumThreads(1)  # Disabilita il multi-threading su FPGA
        try:
            detector = cv2.xfeatures2d.SURF_create()
        except AttributeError:
            raise AttributeError("SURF is not available. Please install opencv-contrib-python.")
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    elif method.upper() == "ORB":
        cv2.setUseOptimized(False)
        cv2.setNumThreads(1)  # Evita il multi-threading su FPGA
        try:
            print(f"Creating {method}")
            detector = cv2.ORB_create()
            print(f"Created {method}")
        except AttributeError:
            raise AttributeError("ORB is not available. Please install opencv-contrib")
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    elif method.upper() == "AKAZE":
        detector = cv2.AKAZE_create()
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    elif method.upper() == "BRISK":
        detector = cv2.BRISK_create()
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    elif method.upper() == "RIFT":
        if not RIFT_AVAILABLE:
            raise ValueError("RIFT method is not available. Please install the RIFT2 package.")
        # For RIFT, we'll use None for both detector and matcher since RIFT has its own pipeline
        # The actual RIFT processing will be handled separately
        detector = None
        matcher = None
    elif method.upper() == "LGHD":
        if not LGHD_AVAILABLE:
            raise ValueError("LGHD method is not available. Please install the required packages.")
        # For LGHD, similar to RIFT, we'll use None for both detector and matcher
        # The actual LGHD processing will be handled separately
        detector = None
        matcher = None
    elif method.upper() == "SARSIFT":
        if not SARSIFT_AVAILABLE:
            raise ValueError("SAR-SIFT method is not available. Please make sure sar_sift.py is in your PYTHONPATH.")
        # For SAR-SIFT, we'll use a special detector wrapper and a cosine similarity matcher
        detector = SARSIFTDetector()
        # FLANN_INDEX_KDTREE is appropriate as SAR-SIFT uses float descriptors
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
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

def process_lghd(sar_img_path, opt_img_path):
    """
    Process a pair of images using the LGHD algorithm.
    This function adapts the original ImageRegistration class to match the
    expected interface of the image registration framework.
    
    Args:
        sar_img (np.ndarray): SAR/LWIR image
        opt_img (np.ndarray): Optical/RGB image
        patch_size (int): Size of patches for LGHD descriptor
        max_keypoints (int): Maximum number of keypoints to detect
        min_rgb_contrast (float): Minimum contrast for RGB keypoint detection
        min_lwir_contrast (float): Minimum contrast for LWIR keypoint detection
        ratio_threshold (float): Ratio test threshold for descriptor matching
        ransac_threshold (float): Threshold for RANSAC homography estimation
        
    Returns:
        dict: Dictionary containing processing results
    """

    start_time = time.time()
    opt_img = cv2.imread(opt_img_path)
    sar_img = cv2.imread(sar_img_path, cv2.IMREAD_GRAYSCALE)
    # Create LGHD image registration object
    # This uses the original ImageRegistration class without modification
    # Create image registration object
    reg = LGHDRegistration(
        patch_size=96,
        max_keypoints=5000,
        min_rgb_contrast=0.1,
        min_lwir_contrast=0.05,
        ratio_threshold=1,
        ransac_threshold=3
    )
    
    # Register the images using the original method
    H, src_pts, dst_pts, inlier_mask = reg.register(opt_img, sar_img)
    
    # Handle the case where registration fails
    if H is None or src_pts is None or dst_pts is None:
        print("LGHD registration failed.")
        return {
            'NM': 0,
            'NCM': 0,
            'ratio': 0,
            'reg_time': time.time() - start_time,
            'rmse': float('inf'),
            'transformation_matrix': np.eye(3),
            'registered_img': sar_img.copy() if len(sar_img.shape) == 3 else cv2.cvtColor(sar_img, cv2.COLOR_GRAY2BGR),
            'mosaic_img': None,
            'matches_img': None,
            'keypoints_sar': None,
            'keypoints_opt': None,
            'good_matches': None
        }
    
    # Count matches
    NM = len(src_pts)  # Total matches
    NCM = np.sum(inlier_mask) if inlier_mask is not None else 0  # Inliers
    
    # Calculate RMSE
    rmse = 0.0
    if NCM > 0:
        # Get inlier points
        inlier_mask_corretto = inlier_mask[:len(src_pts)]
        inlier_src = src_pts[inlier_mask_corretto]
        inlier_dst = dst_pts[inlier_mask_corretto]
        
        # Calculate RMSE
        ones = np.ones((len(inlier_src), 1))
        pts_homogeneous = np.hstack((inlier_src, ones))
        transformed = np.dot(H, pts_homogeneous.T).T
        transformed = transformed[:, :2] / transformed[:, 2:]
        errors = transformed - inlier_dst
        rmse = np.sqrt(np.mean(np.sum(errors**2, axis=1)))
    
    # Create registered image (warp optical to SAR space)
    h, w = sar_img.shape[:2]
    registered_img = reg.warp_image(opt_img, H, (h, w))

    
    # Calculate statistics
    reg_time = time.time() - start_time
    ratio = NCM / NM if NM > 0 else 0
    
    # Return results in the format expected by the framework
    return {
        'NM': NM,
        'NCM': NCM,
        'ratio': ratio,
        'reg_time': reg_time,
        'rmse': rmse,
        'transformation_matrix': H,
        'registered_img': registered_img,
        'mosaic_img': None,
        'matches_img': None,
        'keypoints_sar': dst_pts,
        'keypoints_opt': src_pts,
        'good_matches': src_pts[inlier_mask_corretto] if inlier_mask_corretto is not None else None
    }


def process_sarsift(sar_img, opt_img, feature_density=0.008):
    """
    Process a pair of images using the SAR-SIFT algorithm.
    
    Args:
        sar_img (np.ndarray): SAR image
        opt_img (np.ndarray): Optical image
        feature_density (float): Feature density per pixel
        
    Returns:
        dict: Dictionary containing processing results
    """
    import time
    import numpy as np
    import cv2
    from .utils import make_match_image
    from SAR_Sift.matching import match_descriptors, match, DistanceCriterion
    
    start_time = time.time()
    
    # Calculate feature count based on image size
    num_pixels_1 = sar_img.shape[0] * sar_img.shape[1]
    num_pixels_2 = opt_img.shape[0] * opt_img.shape[1]
    nFeatures_1 = int(round(num_pixels_1 * feature_density))
    nFeatures_2 = int(round(num_pixels_2 * feature_density))
    
    # Initialize SAR-SIFT detectors
    sar_sift_1 = SarSift(nFeatures_1, 8, 2, 2**(1.0/3.0), 0.8/5, 0.04)
    sar_sift_2 = SarSift(nFeatures_2, 8, 2, 2**(1.0/3.0), 0.8/5, 0.04)
    
    # Detect keypoints in SAR image
    print("Detecting features in SAR image...")
    keypoints_1, sar_harris_fun_1, amplit_1, orient_1 = sar_sift_1.detect_keys(sar_img)
    print(f"Number of features detected in SAR image: {len(keypoints_1)}")
    
    # Compute descriptors for SAR image
    print("Computing descriptors for SAR image...")
    descriptors_1 = sar_sift_1.compute_descriptors(keypoints_1, amplit_1, orient_1)
    
    # Detect keypoints in optical image
    print("Detecting features in optical image...")
    keypoints_2, sar_harris_fun_2, amplit_2, orient_2 = sar_sift_2.detect_keys(opt_img)
    print(f"Number of features detected in optical image: {len(keypoints_2)}")
    
    # Compute descriptors for optical image
    print("Computing descriptors for optical image...")
    descriptors_2 = sar_sift_2.compute_descriptors(keypoints_2, amplit_2, orient_2)
    
    # Match descriptors
    print("Matching descriptors...")
    dmatchs = match_descriptors(descriptors_1, descriptors_2, DistanceCriterion.COS)
    
    # Find transformation and eliminate outliers
    print("Finding transformation...")
    homography, right_matchs, matched_line = match(
        sar_img, opt_img, dmatchs, keypoints_1, keypoints_2, "perspective"
    )
    
    # Calculate statistics
    NM = len(dmatchs)        # Number of matches
    NCM = len(right_matchs)  # Number of consensus (inlier) matches
    ratio = NCM / NM if NM != 0 else 0
    
    # Create mosaic and fusion images
    from SAR_Sift.matching import image_fusion
#    fusion_image, mosaic_image = image_fusion(sar_img, opt_img, homography)
    
    # Calculate RMSE
    rmse = 0
    if len(right_matchs) > 0:
        # Extract matched points for RMSE calculation
        pts1 = np.array([keypoints_1[m.queryIdx].pt for m in right_matchs])
        pts2 = np.array([keypoints_2[m.trainIdx].pt for m in right_matchs])
        
        # Apply transformation
        ones = np.ones((len(pts2), 1))
        pts_homogeneous = np.hstack((pts2, ones))
        transformed = np.dot(homography, pts_homogeneous.T).T
        transformed[:, :2] = transformed[:, :2] / transformed[:, 2:]
        
        # Calculate RMSE
        errors = transformed[:, :2] - pts1
        rmse = np.sqrt(np.mean(np.sum(errors**2, axis=1)))
    
    execution_time = time.time() - start_time
    
    # Convert SAR-SIFT keypoints to OpenCV keypoints for visualization
    cv_keypoints_1 = []
    cv_keypoints_2 = []
    
    for kp in keypoints_1:
        cv_kp = cv2.KeyPoint(
            x=kp.pt[0], 
            y=kp.pt[1],
            size=kp.size,
            angle=kp.angle,
            response=kp.response,
            octave=kp.octave
        )
        cv_keypoints_1.append(cv_kp)
    
    for kp in keypoints_2:
        cv_kp = cv2.KeyPoint(
            x=kp.pt[0], 
            y=kp.pt[1],
            size=kp.size,
            angle=kp.angle,
            response=kp.response,
            octave=kp.octave
        )
        cv_keypoints_2.append(cv_kp)
    
    # Convert match objects for visualization
    cv_matches = []
    for m in right_matchs:
        cv_match = cv2.DMatch(m.queryIdx, m.trainIdx, m.distance)
        cv_matches.append(cv_match)
    
    # Create match visualization
    matches_img = cv2.drawMatches(
        sar_img, cv_keypoints_1,
        opt_img, cv_keypoints_2,
        cv_matches, None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    # Return results as a dictionary
    return {
        'NM': NM,
        'NCM': NCM,
        'ratio': ratio,
        'reg_time': execution_time,
        'rmse': rmse,
        'transformation_matrix': homography,
        'registered_img': None,
        'mosaic_img': None,
        'matches_img': matches_img,
        'keypoints_sar': np.array([kp.pt for kp in keypoints_1]),
        'keypoints_opt': np.array([kp.pt for kp in keypoints_2]),
        'good_matches': cv_matches
    }