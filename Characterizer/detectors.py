#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature detectors and matchers for image registration.

This module provides a unified interface for various feature detection
and matching algorithms used in image registration, including:
- OpenCV built-in methods (SIFT, SURF, ORB, AKAZE, BRISK)
- External methods (RIFT, LGHD, SAR-SIFT) when available
"""
import cv2
import numpy as np
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check for optional dependencies
RIFT_AVAILABLE = False
LGHD_AVAILABLE = False
SARSIFT_AVAILABLE = False
MINIMA_AVAILABLE = False

# Try to import RIFT modules
try:
    from rift2.FeatureDetection import FeatureDetection
    from rift2.kptsOrientation import kptsOrientation
    from rift2.FeatureDescribe import FeatureDescribe
    RIFT_AVAILABLE = True
    logger.info("RIFT2 module is available and will be used if selected.")
except ImportError as e:
    logger.warning(f"RIFT2 modules not available. RIFT method will be skipped if requested. Exception: {e}")

# Try to import LGHD modules
try:
    from LGHD.feature_descriptor import FeatureDescriptor
    from LGHD.LGHDRegistration import LGHDRegistration
    LGHD_AVAILABLE = True
    logger.info("LGHD module is available and will be used if selected.")
except ImportError as e:
    logger.warning(f"LGHD modules not available. LGHD method will be skipped if requested. Exception: {e}")

# Try to import SAR-SIFT module
try:
    from SAR_Sift.sar_sift import SarSift, Keypoint as SARKeypoint
    from SAR_Sift.matching import match_descriptors, match, DistanceCriterion
    SARSIFT_AVAILABLE = True
    logger.info("SAR-SIFT module is available and will be used if selected.")
except ImportError:
    logger.warning("SAR-SIFT module not available. SAR-SIFT method will be skipped if requested.")


try:
    from MINIMA import demo
    MINIMA_AVAILABLE = True
    logger.info("MINIMA module is available and will be used if selected.")
except ImportError as e:
    logger.warning(f"MINIMA module not available. MINIMA method will be skipped if requested. {e}")



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
    Creates a detector and matcher based on the specified method.
    
    Args:
        method (str): The feature detection method (SIFT, SURF, ORB, AKAZE, BRISK, RIFT, LGHD, SARSIFT)
        
    Returns:
        tuple: (detector, matcher) configured for the specified method
        
    Raises:
        ValueError: If the method is not recognized or not available
    """
    method = method.upper()
    logger.error(f"Creating detector and matcher for method: {method}")
    # Dictionary mapping methods to their initialization functions
    detector_factory = {
        "SIFT": _create_sift,
        "SURF": _create_surf,
        "ORB": _create_orb,
        "AKAZE": _create_akaze,
        "BRISK": _create_brisk,
        "RIFT": _create_rift,
        "LGHD": _create_lghd,
        "SARSIFT": _create_sarsift,
        "MINIMA": _create_MINIMA,
    }
    
    if method not in detector_factory:
        raise ValueError(f"Method {method} not recognized.")
    
    return detector_factory[method]()


def _create_MINIMA():
    """Create MINIMA detector and matcher."""
    return None, None

def _create_sift():
    """Create SIFT detector and matcher."""
    detector = cv2.SIFT_create()
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    return detector, matcher


def _create_surf():
    """Create SURF detector and matcher."""
    cv2.setUseOptimized(False)  # Avoid memory access issues on FPGA
    cv2.setNumThreads(1)  # Disable multi-threading on FPGA
    try:
        detector = cv2.xfeatures2d.SURF_create()
    except AttributeError:
        raise AttributeError("SURF is not available. Please install opencv-contrib-python.")
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    return detector, matcher


def _create_orb():
    """Create ORB detector and matcher."""
    cv2.setUseOptimized(False)
    cv2.setNumThreads(1)
    try:
        logger.info("Creating ORB detector")
        detector = cv2.ORB_create()
        logger.info("ORB detector created")
    except AttributeError:
        raise AttributeError("ORB is not available. Please install opencv-contrib")
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    return detector, matcher


def _create_akaze():
    """Create AKAZE detector and matcher."""
    detector = cv2.AKAZE_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    return detector, matcher


def _create_brisk():
    """Create BRISK detector and matcher."""
    detector = cv2.BRISK_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    return detector, matcher


def _create_rift():
    """Create RIFT detector and matcher."""
    if not RIFT_AVAILABLE:
        raise ValueError("RIFT method is not available. Please install the RIFT2 package.")
    # RIFT has its own pipeline, so return None for both detector and matcher
    return None, None


def _create_lghd():
    """Create LGHD detector and matcher."""
    if not LGHD_AVAILABLE:
        raise ValueError("LGHD method is not available. Please install the required packages.")
    # LGHD has its own pipeline, so return None for both detector and matcher
    return None, None


def _create_sarsift():
    """Create SAR-SIFT detector and matcher."""
    if not SARSIFT_AVAILABLE:
        raise ValueError("SAR-SIFT method is not available. Please make sure SAR_Sift is in your PYTHONPATH.")
    # Use special SAR-SIFT detector wrapper
    detector = SARSIFTDetector()
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    return detector, matcher


def process_rift(sar_img, opt_img):
    """
    Process a pair of images using the RIFT2 algorithm.
    
    Args:
        sar_img (np.ndarray): SAR image (grayscale or BGR)
        opt_img (np.ndarray): Optical image (grayscale or BGR)
        
    Returns:
        dict: Registration results including keypoints, matches, and transformation
        
    Raises:
        ImportError: If RIFT2 modules are not available
    """
    if not RIFT_AVAILABLE:
        raise ImportError("RIFT2 modules are not available. Please install the RIFT2 package.")
    
    start_time = time.time()
    
    # Prepare images (convert to BGR if grayscale)
    sar_img_bgr = _ensure_bgr(sar_img)
    opt_img_bgr = _ensure_bgr(opt_img)
    
    # Feature detection
    kpts_sar, descriptors_sar = _rift_detect_and_describe(sar_img_bgr)
    kpts_opt, descriptors_opt = _rift_detect_and_describe(opt_img_bgr)
    
    # Feature matching
    good_matches, matched_pts1, matched_pts2 = _rift_match_features(
        kpts_sar, kpts_opt, descriptors_sar, descriptors_opt
    )
    
    # Estimate transformation
    H, inliers, consensus_pts1, consensus_pts2 = _estimate_transformation(matched_pts1, matched_pts2)
    
    # Generate visualization images
    registered_img, mosaic_img = _create_fusion_images(sar_img_bgr, opt_img_bgr, H)
    matches_img = _create_matches_visualization_rift(sar_img_bgr, opt_img_bgr, consensus_pts1, consensus_pts2)
    
    # Calculate statistics
    NM = len(matched_pts1)
    NCM = len(consensus_pts1)
    ratio = NCM / NM if NM != 0 else 0
    rmse = _calculate_rmse(consensus_pts1, consensus_pts2, H)
    execution_time = time.time() - start_time
    
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
        'keypoints_sar': matched_pts1,
        'keypoints_opt': matched_pts2,
        'good_matches': consensus_pts1
    }


def _ensure_bgr(img):
    """Convert grayscale image to BGR if needed."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


def _rift_detect_and_describe(img):
    """Detect and describe features using RIFT algorithm."""
    # Feature detection
    key, m, eo = FeatureDetection(img, 4, 6, 5000)
    
    # Orientation assignment
    kpts = kptsOrientation(key, m, True, 96)
    
    # Feature description
    descriptors = FeatureDescribe(img, eo, kpts, 96, 6, 6)
    descriptors = descriptors.T  # Shape: (numKeypoints, descriptorDimension)
    
    return kpts, descriptors


def _rift_match_features(kpts1, kpts2, desc1, desc2):
    """Match features using RIFT descriptors."""
    # Feature matching
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(desc1.astype(np.float32),
                          desc2.astype(np.float32),
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
    
    return good_matches, matched_pts1_unique, matched_pts2_unique


def _estimate_transformation(pts1, pts2, ransac_threshold=3.0):
    """Estimate transformation matrix using RANSAC."""
    # Check if we have enough matches
    if len(pts1) < 4:
        logger.warning("Not enough unique matches for homography estimation.")
        return np.eye(3), np.zeros(0), pts1, pts2
    
    # Estimate homography using RANSAC
    H, mask = cv2.findHomography(
        pts2,  # Source points
        pts1,  # Destination points
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold
    )
    
    # If homography estimation failed, return identity
    if H is None:
        logger.error("Homography estimation failed!")
        return np.eye(3), np.zeros(0), pts1, pts2
    
    # Extract inlier points
    if len(mask) > 0:
        inlier_idxs = np.where(mask.ravel() == 1)[0]
        consensus_pts1 = pts1[inlier_idxs]
        consensus_pts2 = pts2[inlier_idxs]
        return H, mask, consensus_pts1, consensus_pts2
    else:
        return H, mask, pts1, pts2

class Args:
    def __init__(self, fig1, fig2, method):
        self.fig1 = fig1
        self.fig2 = fig2
        self.method = method
        self.ckpt = None
        self.ckpt2 = None
        self.exp_name =None
        self.thr = None

    def add_method_arguments(self, method):
        if method == "loftr":
            self.ckpt = "MINIMA/weights/minima_loftr.ckpt"
            self.thr = 0.2
        elif method == "sp_lg":
            self.ckpt = "MINIMA/weights/minima_lightglue.pth"
        elif method == "roma":
            self.ckpt2 = "large"
            self.ckpt = "MINIMA/weights/minima_roma.pth"



def process_minima(sar_img_path, opt_img_path,method):
    sar_img = cv2.imread(sar_img_path, cv2.IMREAD_GRAYSCALE)
    opt_img = cv2.imread(opt_img_path, cv2.IMREAD_GRAYSCALE)
    args = Args(fig1=opt_img_path, fig2=sar_img_path, method=method)
    args.add_method_arguments(method)
    return demo.test_relative_pose_demo(method= method, save_dir=None, save_figs=True, args=args)


def _calculate_rmse(pts1, pts2, H):
    """Calculate RMSE between transformed points."""
    if len(pts1) == 0:
        return 0.0
    
    # For perspective transform, we need to apply perspective division
    ones = np.ones((len(pts2), 1))
    pts_homogeneous = np.hstack((pts2, ones))
    transformed = np.dot(H, pts_homogeneous.T).T
    transformed = transformed[:, :2] / transformed[:, 2:]
    errors = transformed - pts1
    rmse = np.sqrt(np.mean(np.sum(errors**2, axis=1)))
    
    return rmse


def _create_fusion_images(image1, image2, H):
    """Create fusion and mosaic visualization images."""
    # Get image dimensions
    h1, w1 = image1.shape[:2]
    
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
    mosaic = _create_checkerboard_mosaic(warped1, warped2)
    
    # Crop images to remove unnecessary black borders
    fusion, mosaic = _crop_to_content(fusion, mosaic)
    
    return fusion, mosaic


def _create_checkerboard_mosaic(img1, img2, block_size=64):
    """Create a checkerboard mosaic from two images."""
    mosaic = np.zeros_like(img1)
    
    # Create checkerboard masks
    y_blocks = img1.shape[0] // block_size + 1
    x_blocks = img1.shape[1] // block_size + 1
    
    for y in range(y_blocks):
        for x in range(x_blocks):
            y1 = y * block_size
            y2 = min((y + 1) * block_size, img1.shape[0])
            x1 = x * block_size
            x2 = min((x + 1) * block_size, img1.shape[1])
            
            if (y + x) % 2 == 0:
                mosaic[y1:y2, x1:x2] = img1[y1:y2, x1:x2]
            else:
                mosaic[y1:y2, x1:x2] = img2[y1:y2, x1:x2]
    
    return mosaic


def _crop_to_content(img1, img2, border=10):
    """Crop images to non-zero content with a small border."""
    # Find non-zero pixels
    non_zero = np.where(img1 > 0)
    if len(non_zero[0]) > 0 and len(non_zero[1]) > 0:
        y_min, y_max = np.min(non_zero[0]), np.max(non_zero[0])
        x_min, x_max = np.min(non_zero[1]), np.max(non_zero[1])
        
        # Add a small border
        y_min = max(0, y_min - border)
        y_max = min(img1.shape[0] - 1, y_max + border)
        x_min = max(0, x_min - border)
        x_max = min(img1.shape[1] - 1, x_max + border)
        
        # Crop both images
        img1_cropped = img1[y_min:y_max+1, x_min:x_max+1]
        img2_cropped = img2[y_min:y_max+1, x_min:x_max+1]
        
        return img1_cropped, img2_cropped
    
    return img1, img2


def _create_matches_visualization_rift(img1, img2, pts1, pts2):
    """Create a visualization of matched points between two images."""
    from .utils import make_match_image
    return make_match_image(img1, img2, pts1, pts2)

def process_lghd(sar_img_path, opt_img_path):
    """
    Process a pair of images using the LGHD (Local Gradient Histogram Descriptor) algorithm.
    
    This function handles the complete LGHD processing pipeline:
    1. Load images
    2. Create LGHD registration object with optimized parameters
    3. Detect and match features
    4. Compute transformation matrix
    5. Generate registered image and visualizations
    
    Args:
        sar_img_path (str): Path to SAR/LWIR image file
        opt_img_path (str): Path to Optical/RGB image file
        
    Returns:
        dict: Registration results including:
            - NM: Number of matches
            - NCM: Number of correct matches (inliers)
            - ratio: Ratio of inliers to total matches
            - reg_time: Registration time
            - rmse: Root Mean Square Error
            - transformation_matrix: Estimated homography
            - registered_img: Warped optical image in SAR space
            - other visualization images and keypoints
    """
    import time
    import logging
    import numpy as np
    import cv2
    
    logger = logging.getLogger(__name__)
    start_time = time.time()
    
    try:
        # Load images
        logger.info(f"Loading images from {sar_img_path} and {opt_img_path}")
        opt_img = cv2.imread(opt_img_path)
        sar_img = cv2.imread(sar_img_path, cv2.IMREAD_GRAYSCALE)
        
        if opt_img is None or sar_img is None:
            logger.error("Failed to load one or both images")
            return _create_empty_lghd_result(start_time)
        
        # Create and configure LGHD registration object
        registration_result = _perform_lghd_registration(sar_img, opt_img)
        
        # Check if registration failed
        if registration_result is None:
            logger.warning("LGHD registration failed")
            return _create_empty_lghd_result(start_time, sar_img)
        
        # Extract results
        H, src_pts, dst_pts, inlier_mask = registration_result
        
        # Process results
        processed_result = _process_lghd_results(
            H, src_pts, dst_pts, inlier_mask, sar_img, opt_img, start_time
        )
        
        return processed_result
        
    except Exception as e:
        logger.error(f"Error in LGHD processing: {str(e)}", exc_info=True)
        return _create_empty_lghd_result(start_time)


def _create_empty_lghd_result(start_time, sar_img=None):
    """Create an empty result dictionary when LGHD registration fails."""
    import time
    import numpy as np
    import cv2
    
    reg_time = time.time() - start_time
    
    # Create a dummy registered image if SAR image is provided
    if sar_img is not None:
        if len(sar_img.shape) == 3:
            registered_img = sar_img.copy()
        else:
            registered_img = cv2.cvtColor(sar_img, cv2.COLOR_GRAY2BGR)
    else:
        registered_img = None
    
    return {
        'NM': 0,
        'NCM': 0,
        'ratio': 0,
        'reg_time': reg_time,
        'rmse': float('inf'),
        'transformation_matrix': np.eye(3),
        'registered_img': registered_img,
        'mosaic_img': None,
        'matches_img': None,
        'keypoints_sar': None,
        'keypoints_opt': None,
        'good_matches': None
    }


def _perform_lghd_registration(sar_img, opt_img):
    """
    Perform image registration using LGHD algorithm.
    
    Args:
        sar_img: SAR/LWIR image
        opt_img: Optical/RGB image
        
    Returns:
        tuple or None: (H, src_pts, dst_pts, inlier_mask) or None if registration fails
    """
    import logging
    from LGHD.LGHDRegistration import LGHDRegistration
    
    logger = logging.getLogger(__name__)
    
    try:
        # Create LGHD registration object with optimized parameters
        logger.info("Creating LGHD registration object")
        reg = LGHDRegistration(
            patch_size=96,
            max_keypoints=5000,
            min_rgb_contrast=0.1,
            min_lwir_contrast=0.05,
            ratio_threshold=1,
            ransac_threshold=3
        )
        
        # Register the images
        logger.info("Performing LGHD registration")
        H, src_pts, dst_pts, inlier_mask = reg.register(opt_img, sar_img)
        
        # Validate registration results
        if H is None or src_pts is None or dst_pts is None:
            logger.warning("LGHD registration returned invalid results")
            return None
            
        return H, src_pts, dst_pts, inlier_mask
        
    except Exception as e:
        logger.error(f"Error in LGHD registration: {str(e)}")
        return None


def _process_lghd_results(H, src_pts, dst_pts, inlier_mask, sar_img, opt_img, start_time):
    """
    Process LGHD registration results.
    
    Args:
        H: Homography matrix
        src_pts: Source points (optical image)
        dst_pts: Destination points (SAR image)
        inlier_mask: Boolean mask of inliers
        sar_img: SAR image
        opt_img: Optical image
        start_time: Processing start time
        
    Returns:
        dict: Processed registration results
    """
    import time
    import numpy as np
    import cv2
    from LGHD.LGHDRegistration import LGHDRegistration
    
    # Count matches
    NM = len(src_pts)  # Total matches
    NCM = np.sum(inlier_mask) if inlier_mask is not None else 0  # Inliers
    
    # Ensure inlier mask is properly sized
    if inlier_mask is not None and len(inlier_mask) > len(src_pts):
        inlier_mask = inlier_mask[:len(src_pts)]
    
    # Calculate RMSE for inliers
    rmse = _calculate_lghd_rmse(H, src_pts, dst_pts, inlier_mask)
    
    # Create registered image (warp optical to SAR space)
    h, w = sar_img.shape[:2]
    reg = LGHDRegistration(patch_size=96)  # Create temporary object just for warping
    registered_img = reg.warp_image(opt_img, H, (h, w))
    
    # Calculate statistics
    reg_time = time.time() - start_time
    ratio = NCM / NM if NM > 0 else 0
    
    # Return results
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
        'good_matches': src_pts[inlier_mask] if inlier_mask is not None and NM > 0 else None
    }


def _calculate_lghd_rmse(H, src_pts, dst_pts, inlier_mask):
    """
    Calculate RMSE between transformed points and ground truth.
    
    Args:
        H: Homography matrix
        src_pts: Source points
        dst_pts: Destination points
        inlier_mask: Boolean mask of inliers
        
    Returns:
        float: RMSE value or 0.0 if calculation fails
    """
    import numpy as np
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # Return 0 if no inliers
        if inlier_mask is None or np.sum(inlier_mask) == 0:
            return 0.0
            
        # Extract inlier points
        inlier_src = src_pts[inlier_mask]
        inlier_dst = dst_pts[inlier_mask]
        
        # Check if we have enough points
        if len(inlier_src) == 0:
            return 0.0
            
        # Apply transformation
        ones = np.ones((len(inlier_src), 1))
        pts_homogeneous = np.hstack((inlier_src, ones))
        transformed = np.dot(H, pts_homogeneous.T).T
        
        # Apply perspective division
        transformed_pts = transformed[:, :2] / transformed[:, 2:]
        
        # Calculate errors
        errors = transformed_pts - inlier_dst
        rmse = np.sqrt(np.mean(np.sum(errors**2, axis=1)))
        
        return rmse
        
    except Exception as e:
        logger.error(f"Error calculating RMSE: {str(e)}")
        return 0.0
    



def process_sarsift(sar_img, opt_img, feature_density=0.008):
    """
    Process a pair of images using the SAR-SIFT algorithm.
    
    This function handles the complete SAR-SIFT pipeline:
    1. Feature detection in both images based on image size
    2. Feature description
    3. Feature matching
    4. Transformation estimation
    5. Visualization creation
    
    Args:
        sar_img (np.ndarray): SAR image
        opt_img (np.ndarray): Optical image
        feature_density (float): Feature density per pixel (controls keypoint count)
        
    Returns:
        dict: Registration results including:
            - NM: Number of matches
            - NCM: Number of correct matches (inliers)
            - ratio: Ratio of inliers to total matches
            - reg_time: Registration time
            - rmse: Root Mean Square Error
            - transformation_matrix: Estimated homography
            - matches_img: Visualization of feature matches
            - keypoints_sar, keypoints_opt: Detected keypoints
            - good_matches: Inlier matches
    """
    import time
    import numpy as np
    import cv2
    import logging
    from SAR_Sift.matching import match_descriptors, match, DistanceCriterion
    
    logger = logging.getLogger(__name__)
    start_time = time.time()
    
    try:
        # Input validation
        if sar_img is None or opt_img is None:
            logger.error("Invalid input images")
            return _create_empty_sarsift_result(start_time)
        
        # Detect features in both images
        keypoints_info = _detect_sarsift_features(sar_img, opt_img, feature_density)
        if keypoints_info is None:
            return _create_empty_sarsift_result(start_time)
            
        keypoints_1, descriptors_1, keypoints_2, descriptors_2 = keypoints_info
        
        # Match features
        matches_info = _match_sarsift_features(
            sar_img, opt_img, keypoints_1, keypoints_2, descriptors_1, descriptors_2
        )
        if matches_info is None:
            return _create_empty_sarsift_result(start_time)
            
        homography, right_matchs, NM, NCM = matches_info
        
        # Calculate statistics and generate visualizations
        results = _process_sarsift_results(
            sar_img, opt_img, keypoints_1, keypoints_2, 
            right_matchs, homography, NM, NCM, start_time
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Error in SAR-SIFT processing: {str(e)}", exc_info=True)
        return _create_empty_sarsift_result(start_time)


def _create_empty_sarsift_result(start_time):
    """Create an empty result dictionary when SAR-SIFT fails."""
    import time
    import numpy as np
    
    return {
        'NM': 0,
        'NCM': 0,
        'ratio': 0,
        'reg_time': time.time() - start_time,
        'rmse': 0,
        'transformation_matrix': np.eye(3),
        'registered_img': None,
        'mosaic_img': None,
        'matches_img': None,
        'keypoints_sar': np.array([]),
        'keypoints_opt': np.array([]),
        'good_matches': []
    }


def _detect_sarsift_features(sar_img, opt_img, feature_density):
    """
    Detect and describe features in both images using SAR-SIFT.
    
    Args:
        sar_img: SAR image
        opt_img: Optical image
        feature_density: Feature density per pixel
        
    Returns:
        tuple or None: (keypoints_1, descriptors_1, keypoints_2, descriptors_2)
    """
    import logging
    from SAR_Sift.sar_sift import SarSift
    
    logger = logging.getLogger(__name__)
    
    try:
        # Calculate feature count based on image size
        num_pixels_1 = sar_img.shape[0] * sar_img.shape[1]
        num_pixels_2 = opt_img.shape[0] * opt_img.shape[1]
        nFeatures_1 = int(round(num_pixels_1 * feature_density))
        nFeatures_2 = int(round(num_pixels_2 * feature_density))
        
        # Initialize SAR-SIFT detectors
        sar_sift_1 = SarSift(nFeatures_1, 8, 2, 2**(1.0/3.0), 0.8/5, 0.04)
        sar_sift_2 = SarSift(nFeatures_2, 8, 2, 2**(1.0/3.0), 0.8/5, 0.04)
        
        # Detect keypoints in SAR image
        logger.info("Detecting features in SAR image...")
        keypoints_1, sar_harris_fun_1, amplit_1, orient_1 = sar_sift_1.detect_keys(sar_img)
        logger.info(f"Number of features detected in SAR image: {len(keypoints_1)}")
        
        # Compute descriptors for SAR image
        logger.info("Computing descriptors for SAR image...")
        descriptors_1 = sar_sift_1.compute_descriptors(keypoints_1, amplit_1, orient_1)
        
        # Detect keypoints in optical image
        logger.info("Detecting features in optical image...")
        keypoints_2, sar_harris_fun_2, amplit_2, orient_2 = sar_sift_2.detect_keys(opt_img)
        logger.info(f"Number of features detected in optical image: {len(keypoints_2)}")
        
        # Compute descriptors for optical image
        logger.info("Computing descriptors for optical image...")
        descriptors_2 = sar_sift_2.compute_descriptors(keypoints_2, amplit_2, orient_2)
        
        return keypoints_1, descriptors_1, keypoints_2, descriptors_2
        
    except Exception as e:
        logger.error(f"Error detecting SAR-SIFT features: {str(e)}")
        return None


def _match_sarsift_features(sar_img, opt_img, keypoints_1, keypoints_2, descriptors_1, descriptors_2):
    """
    Match features between the two images.
    
    Args:
        sar_img, opt_img: Input images
        keypoints_1, keypoints_2: Detected keypoints
        descriptors_1, descriptors_2: Feature descriptors
        
    Returns:
        tuple or None: (homography, right_matchs, NM, NCM)
    """
    import logging
    from SAR_Sift.matching import match_descriptors, match, DistanceCriterion
    
    logger = logging.getLogger(__name__)
    
    try:
        # Match descriptors
        logger.info("Matching descriptors...")
        dmatchs = match_descriptors(descriptors_1, descriptors_2, DistanceCriterion.COS)
        
        # Find transformation and eliminate outliers
        logger.info("Finding transformation...")
        homography, right_matchs, matched_line = match(
            sar_img, opt_img, dmatchs, keypoints_1, keypoints_2, "perspective"
        )
        
        # Calculate match statistics
        NM = len(dmatchs)        # Number of matches
        NCM = len(right_matchs)  # Number of consensus (inlier) matches
        
        logger.info(f"Total matches: {NM}, Inlier matches: {NCM}")
        
        return homography, right_matchs, NM, NCM
        
    except Exception as e:
        logger.error(f"Error matching SAR-SIFT features: {str(e)}")
        return None


def _process_sarsift_results(sar_img, opt_img, keypoints_1, keypoints_2, 
                           right_matchs, homography, NM, NCM, start_time):
    """
    Process the results of SAR-SIFT feature matching.
    
    Args:
        sar_img, opt_img: Input images
        keypoints_1, keypoints_2: Detected keypoints
        right_matchs: Inlier matches
        homography: Transformation matrix
        NM, NCM: Match statistics
        start_time: Processing start time
        
    Returns:
        dict: Processed results dictionary
    """
    import time
    import cv2
    import numpy as np
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # Calculate ratio
        ratio = NCM / NM if NM != 0 else 0
        
        # Calculate RMSE
        rmse = _calculate_sarsift_rmse(keypoints_1, keypoints_2, right_matchs, homography)
        
        # Create visualizations
        cv_keypoints_1 = _convert_to_cv_keypoints(keypoints_1)
        cv_keypoints_2 = _convert_to_cv_keypoints(keypoints_2)
        cv_matches = _convert_to_cv_matches(right_matchs)
        
        # Create match visualization
        matches_img = _create_matches_visualization(
            sar_img, opt_img, cv_keypoints_1, cv_keypoints_2, cv_matches
        )
        
        # Calculate total execution time
        execution_time = time.time() - start_time
        
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
        
    except Exception as e:
        logger.error(f"Error processing SAR-SIFT results: {str(e)}")
        return _create_empty_sarsift_result(start_time)


def _calculate_sarsift_rmse(keypoints_1, keypoints_2, right_matchs, homography):
    """
    Calculate RMSE between transformed points and ground truth.
    
    Args:
        keypoints_1, keypoints_2: Detected keypoints
        right_matchs: Inlier matches
        homography: Transformation matrix
        
    Returns:
        float: RMSE value or 0.0 if calculation fails
    """
    import numpy as np
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # Return 0 if no inliers
        if len(right_matchs) == 0:
            return 0.0
            
        # Extract matched points
        pts1 = np.array([keypoints_1[m.queryIdx].pt for m in right_matchs])
        pts2 = np.array([keypoints_2[m.trainIdx].pt for m in right_matchs])
        
        # Apply transformation
        ones = np.ones((len(pts2), 1))
        pts_homogeneous = np.hstack((pts2, ones))
        transformed = np.dot(homography, pts_homogeneous.T).T
        
        # Apply perspective division
        transformed[:, :2] = transformed[:, :2] / transformed[:, 2:]
        
        # Calculate errors
        errors = transformed[:, :2] - pts1
        rmse = np.sqrt(np.mean(np.sum(errors**2, axis=1)))
        
        return rmse
        
    except Exception as e:
        logger.error(f"Error calculating RMSE: {str(e)}")
        return 0.0


def _convert_to_cv_keypoints(keypoints):
    """
    Convert SAR-SIFT keypoints to OpenCV keypoints.
    
    Args:
        keypoints: List of SAR-SIFT keypoints
        
    Returns:
        list: List of OpenCV KeyPoint objects
    """
    import cv2
    
    cv_keypoints = []
    for kp in keypoints:
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


def _convert_to_cv_matches(matches):
    """
    Convert SAR-SIFT matches to OpenCV DMatch objects.
    
    Args:
        matches: List of SAR-SIFT matches
        
    Returns:
        list: List of OpenCV DMatch objects
    """
    import cv2
    
    cv_matches = []
    for m in matches:
        cv_match = cv2.DMatch(m.queryIdx, m.trainIdx, m.distance)
        cv_matches.append(cv_match)
    
    return cv_matches


def _create_matches_visualization(img1, img2, keypoints1, keypoints2, matches):
    """
    Create visualization of matched features between images.
    
    Args:
        img1, img2: Input images
        keypoints1, keypoints2: OpenCV keypoints
        matches: OpenCV matches
        
    Returns:
        np.ndarray: Visualization image
    """
    import cv2
    
    # Create visualization image
    matches_img = cv2.drawMatches(
        img1, keypoints1,
        img2, keypoints2,
        matches, None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    return matches_img