#!/usr/bin/env python3
"""
Test script for LGHD-based image registration
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from feature_descriptor import FeatureDescriptor

def detect_keypoints(image, max_keypoints=1000, min_contrast=0.1):
    """
    Detect keypoints in an image using FAST detector
    
    Args:
        image: Input image
        max_keypoints: Maximum number of keypoints to detect
        min_contrast: Minimum contrast threshold for FAST detector
        
    Returns:
        keypoints: Array of keypoint coordinates [x, y]
    """
    # Create FAST detector
    detector = cv2.FastFeatureDetector_create(threshold=int(min_contrast * 255))
    
    # Detect keypoints
    kps = detector.detect(image, None)
    
    # Sort keypoints by response (strength)
    kps = sorted(kps, key=lambda x: x.response, reverse=True)
    
    # Limit the number of keypoints
    kps = kps[:max_keypoints]
    
    # Extract keypoint coordinates
    keypoints = np.array([[kp.pt[0], kp.pt[1]] for kp in kps])
    
    return keypoints

def match_descriptors(desc1, desc2, ratio_threshold=0.8):
    """
    Match descriptors using ratio test
    
    Args:
        desc1: First descriptor set
        desc2: Second descriptor set
        ratio_threshold: Ratio test threshold
        
    Returns:
        matches: List of DMatch objects
    """
    # Convert descriptors to float32
    desc1 = desc1.astype(np.float32)
    desc2 = desc2.astype(np.float32)
    
    # Create BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2)
    
    # Compute two nearest neighbors for each descriptor
    matches_nn = bf.knnMatch(desc1, desc2, k=2)
    
    # Apply ratio test
    good_matches = []
    for m, n in matches_nn:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)
    
    return good_matches

def test_lghd_registration(rgb_path, lwir_path, output_path=None):
    """
    Test LGHD-based image registration
    
    Args:
        rgb_path: Path to RGB image
        lwir_path: Path to LWIR image
        output_path: Path to save result image (optional)
        
    Returns:
        H: Homography matrix
    """
    print(f"Loading images from {rgb_path} and {lwir_path}")
    
    # Load images
    im_rgb = cv2.imread(rgb_path, cv2.IMREAD_GRAYSCALE)
    im_lwir = cv2.imread(lwir_path, cv2.IMREAD_GRAYSCALE)
    
    if im_rgb is None:
        raise ValueError(f"Could not load RGB image from {rgb_path}")
    if im_lwir is None:
        raise ValueError(f"Could not load LWIR image from {lwir_path}")
    
    # Detect keypoints
    print("Detecting keypoints...")
    rgb_kps = detect_keypoints(im_rgb, max_keypoints=500)
    lwir_kps = detect_keypoints(im_lwir, max_keypoints=500, min_contrast=0.05)
    
    print(f"Found {len(rgb_kps)} keypoints in RGB and {len(lwir_kps)} in LWIR")
    
    # Compute LGHD descriptors
    print("Computing LGHD descriptors...")
    fd = FeatureDescriptor('LGHD')
    fd.set_parameters({'patch_size': 64})  # Smaller patch size for speed
    
    res_rgb = fd.compute(im_rgb, rgb_kps)
    res_lwir = fd.compute(im_lwir, lwir_kps)
    
    print(f"Computed descriptors for {len(res_rgb['kps'])} RGB points and {len(res_lwir['kps'])} LWIR points")
    
    # Match descriptors
    print("Matching descriptors...")
    matches = match_descriptors(res_rgb['des'], res_lwir['des'], ratio_threshold=0.8)
    
    print(f"Found {len(matches)} good matches")
    
    if len(matches) < 10:
        print("Not enough matches to compute homography")
        return None
    
    # Extract matched points
    src_pts = np.float32([res_rgb['kps'][m.queryIdx] for m in matches])
    dst_pts = np.float32([res_lwir['kps'][m.trainIdx] for m in matches])
    
    # Compute homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    
    inliers = mask.ravel() == 1
    inlier_matches = [matches[i] for i in range(len(matches)) if inliers[i]]
    
    print(f"Found {np.sum(inliers)} inliers")
    
    # Visualize matches
    rgb_key_pts = [cv2.KeyPoint(x[0], x[1], 1) for x in res_rgb['kps']]
    lwir_key_pts = [cv2.KeyPoint(x[0], x[1], 1) for x in res_lwir['kps']]
    
    match_img = cv2.drawMatches(im_rgb, rgb_key_pts, im_lwir, lwir_key_pts, 
                             inlier_matches, None, 
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(match_img, cmap='gray')
    plt.title(f'LGHD Matches: {len(inlier_matches)} inliers')
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    else:
        plt.show()
    
    # Warp RGB to LWIR for visualization
    if H is not None:
        h, w = im_lwir.shape
        warped_rgb = cv2.warpPerspective(im_rgb, H, (w, h))
        
        # Create checkerboard visualization
        checkerboard = np.zeros_like(im_lwir)
        block_size = 32
        for i in range(0, h, block_size*2):
            for j in range(0, w, block_size*2):
                checkerboard[i:min(i+block_size, h), j:min(j+block_size, w)] = 1
                if i+block_size < h and j+block_size < w:
                    checkerboard[i+block_size:min(i+block_size*2, h), 
                                j+block_size:min(j+block_size*2, w)] = 1
        
        # Blend images
        blend = np.zeros((h, w, 3), dtype=np.uint8)
        blend[..., 0] = np.where(checkerboard == 1, im_lwir, 0)
        blend[..., 1] = np.where(checkerboard == 1, 0, warped_rgb)
        blend[..., 2] = np.where(checkerboard == 1, 0, warped_rgb)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(blend)
        plt.title('Registration Result (Checkerboard)')
        plt.axis('off')
        
        if output_path:
            base, ext = os.path.splitext(output_path)
            plt.savefig(f"{base}_checkerboard{ext}", bbox_inches='tight')
        else:
            plt.show()
    
    return H

if __name__ == "__main__":
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test LGHD-based image registration')
    parser.add_argument('--rgb', type=str, required=True, help='Path to RGB image')
    parser.add_argument('--lwir', type=str, required=True, help='Path to LWIR image')
    parser.add_argument('--output', type=str, help='Path to save result image')
    
    args = parser.parse_args()
    
    # Run test
    H = test_lghd_registration(args.rgb, args.lwir, args.output)
    
    if H is not None:
        print(f"Homography matrix:\n{H}")
    else:
        print("Registration failed.")