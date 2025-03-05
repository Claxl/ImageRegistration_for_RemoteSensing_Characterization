#!/usr/bin/env python3
"""
Image Registration using LGHD (Log-Gabor Histogram Descriptor)

This script provides a simple interface for registering RGB and LWIR images
using the LGHD descriptor.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from .feature_descriptor import FeatureDescriptor

class LGHDRegistration:
    """
    Class for registering RGB and LWIR images using the LGHD descriptor
    """
    
    def __init__(self, patch_size=64, max_keypoints=500, min_rgb_contrast=0.1, 
                 min_lwir_contrast=0.05, ratio_threshold=0.8, ransac_threshold=3.0):
        """
        Initialize the image registration object
        
        Args:
            patch_size: Size of patches for LGHD descriptor
            max_keypoints: Maximum number of keypoints to detect
            min_rgb_contrast: Minimum contrast for RGB keypoint detection
            min_lwir_contrast: Minimum contrast for LWIR keypoint detection
            ratio_threshold: Ratio test threshold for descriptor matching
            ransac_threshold: Threshold for RANSAC homography estimation
        """
        self.patch_size = patch_size
        self.max_keypoints = max_keypoints
        self.min_rgb_contrast = min_rgb_contrast
        self.min_lwir_contrast = min_lwir_contrast
        self.ratio_threshold = ratio_threshold
        self.ransac_threshold = ransac_threshold
        
        # Initialize feature descriptor
        self.fd = FeatureDescriptor('LGHD')
        self.fd.set_parameters({'patch_size': patch_size})
    
    def detect_keypoints(self, image, min_contrast):
        """
        Detect keypoints in an image
        
        Args:
            image: Input image
            min_contrast: Minimum contrast for keypoint detection
            
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
        kps = kps[:self.max_keypoints]
        
        # Extract keypoint coordinates
        keypoints = np.array([[kp.pt[0], kp.pt[1]] for kp in kps])
        
        return keypoints
    
    def match_descriptors(self, desc1, desc2):
        """
        Match descriptors using ratio test
        
        Args:
            desc1: First descriptor set
            desc2: Second descriptor set
            
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
            if m.distance < self.ratio_threshold * n.distance:
                good_matches.append(m)
        
        return good_matches
    
    def register(self, rgb_image, lwir_image):
        """
        Register RGB image to LWIR image
        
        Args:
            rgb_image: RGB image (grayscale or color)
            lwir_image: LWIR image (grayscale)
            
        Returns:
            H: Homography matrix
            src_pts: Source points (inliers)
            dst_pts: Destination points (inliers)
            inlier_mask: Boolean mask of inliers
        """
        # Convert to grayscale if needed
        if len(rgb_image.shape) == 3:
            rgb_gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        else:
            rgb_gray = rgb_image
            
        if len(lwir_image.shape) == 3:
            lwir_gray = cv2.cvtColor(lwir_image, cv2.COLOR_BGR2GRAY)
        else:
            lwir_gray = lwir_image
        
        # Detect keypoints
        rgb_kps = self.detect_keypoints(rgb_gray, self.min_rgb_contrast)
        lwir_kps = self.detect_keypoints(lwir_gray, self.min_lwir_contrast)
        
        if len(rgb_kps) < 10 or len(lwir_kps) < 10:
            print(f"Not enough keypoints detected: RGB={len(rgb_kps)}, LWIR={len(lwir_kps)}")
            return None, None, None, None
        
        # Compute LGHD descriptors
        res_rgb = self.fd.compute(rgb_gray, rgb_kps)
        res_lwir = self.fd.compute(lwir_gray, lwir_kps)
        
        if len(res_rgb['kps']) < 10 or len(res_lwir['kps']) < 10:
            print(f"Not enough valid descriptors: RGB={len(res_rgb['kps'])}, LWIR={len(res_lwir['kps'])}")
            return None, None, None, None
        
        # Match descriptors
        matches = self.match_descriptors(res_rgb['des'], res_lwir['des'])
        
        if len(matches) < 10:
            print(f"Not enough matches: {len(matches)}")
            return None, None, None, None
        
        # Extract matched points
        src_pts = np.float32([res_rgb['kps'][m.queryIdx] for m in matches])
        dst_pts = np.float32([res_lwir['kps'][m.trainIdx] for m in matches])
        
        # Compute homography
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_threshold)
        
        if H is None:
            print("Failed to estimate homography")
            return None, None, None, None
        
        # Extract inliers
        inlier_mask = mask.ravel() == 1
        inlier_src_pts = src_pts[inlier_mask]
        inlier_dst_pts = dst_pts[inlier_mask]
        
        return H, inlier_src_pts, inlier_dst_pts, inlier_mask
    
    def visualize_matches(self, rgb_image, lwir_image, src_pts, dst_pts, inlier_mask=None, output_path=None):
        """
        Visualize matches between RGB and LWIR images
        
        Args:
            rgb_image: RGB image
            lwir_image: LWIR image
            src_pts: Source points
            dst_pts: Destination points
            inlier_mask: Boolean mask of inliers
            output_path: Path to save visualization (optional)
        """
        if inlier_mask is None:
            inlier_mask = np.ones(len(src_pts), dtype=bool)
        
        # Convert to grayscale if needed
        if len(rgb_image.shape) == 3:
            rgb_gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        else:
            rgb_gray = rgb_image
            
        if len(lwir_image.shape) == 3:
            lwir_gray = cv2.cvtColor(lwir_image, cv2.COLOR_BGR2GRAY)
        else:
            lwir_gray = lwir_image
        
        # Create keypoints for visualization
        rgb_key_pts = [cv2.KeyPoint(x[0], x[1], 1) for x in src_pts]
        lwir_key_pts = [cv2.KeyPoint(x[0], x[1], 1) for x in dst_pts]
        
        # Create matches
        matches = [cv2.DMatch(i, i, 0) for i in range(len(src_pts)) if inlier_mask[i]]
        
        # Draw matches
        match_img = cv2.drawMatches(rgb_gray, rgb_key_pts, lwir_gray, lwir_key_pts, 
                                matches, None, 
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(match_img, cmap='gray')
        plt.title(f'LGHD Matches: {np.sum(inlier_mask)} inliers')
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def warp_image(self, image, H, target_shape):
        """
        Warp image using homography
        
        Args:
            image: Image to warp
            H: Homography matrix
            target_shape: Shape of target image (height, width)
            
        Returns:
            warped_image: Warped image
        """
        return cv2.warpPerspective(image, H, (target_shape[1], target_shape[0]))
    
    def visualize_registration(self, rgb_image, lwir_image, H, output_path=None):
        """
        Visualize registration result
        
        Args:
            rgb_image: RGB image
            lwir_image: LWIR image
            H: Homography matrix
            output_path: Path to save visualization (optional)
        """
        h, w = lwir_image.shape[:2]
        
        # Warp RGB to LWIR
        warped_rgb = self.warp_image(rgb_image, H, (h, w))
        
        # Convert to grayscale if needed
        if len(warped_rgb.shape) == 3:
            warped_gray = cv2.cvtColor(warped_rgb, cv2.COLOR_BGR2GRAY)
        else:
            warped_gray = warped_rgb
            
        if len(lwir_image.shape) == 3:
            lwir_gray = cv2.cvtColor(lwir_image, cv2.COLOR_BGR2GRAY)
        else:
            lwir_gray = lwir_image
        
        # Create checkerboard pattern
        checkerboard = np.zeros((h, w), dtype=bool)
        block_size = 32
        for i in range(0, h, block_size*2):
            for j in range(0, w, block_size*2):
                checkerboard[i:min(i+block_size, h), j:min(j+block_size, w)] = True
                if i+block_size < h and j+block_size < w:
                    checkerboard[i+block_size:min(i+block_size*2, h), 
                                j+block_size:min(j+block_size*2, w)] = True
        
        # Create overlay visualization
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        overlay[..., 0] = lwir_gray
        overlay[..., 1] = warped_gray
        overlay[..., 2] = 0
        
        # Create checkerboard visualization - FIXED
        checker = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Convert both images to BGR format first
        lwir_color = cv2.cvtColor(lwir_gray, cv2.COLOR_GRAY2BGR)
        
        if len(warped_rgb.shape) == 3:
            warped_color = warped_rgb
        else:
            warped_color = cv2.cvtColor(warped_gray, cv2.COLOR_GRAY2BGR)
        
        # Use where to combine images based on the checkerboard mask
        checker = np.where(
            np.repeat(checkerboard[:, :, np.newaxis], 3, axis=2),
            lwir_color,
            warped_color
        )
        
        # Create figure
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        if len(warped_rgb.shape) == 3:
            plt.imshow(cv2.cvtColor(warped_rgb, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(warped_rgb, cmap='gray')
        plt.title('Warped RGB')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(overlay)
        plt.title('Overlay (R=LWIR, G=RGB)')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(checker, cv2.COLOR_BGR2RGB))
        plt.title('Checkerboard')
        plt.axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return warped_rgb, overlay, checker

