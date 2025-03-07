#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for OS-SIFT algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2


def visualize_keypoints(image, keypoints, title='Detected Keypoints'):
    """
    Visualize detected keypoints on an image.
    
    Parameters:
    -----------
    image : ndarray
        Input image
    keypoints : ndarray
        Array of keypoints [x, y, scale, layer, orientation, ...]
    title : str
        Title for the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Show image in grayscale
    if len(image.shape) == 3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image, cmap='gray')
    
    # Extract keypoint coordinates
    kp_x = keypoints[:, 0]
    kp_y = keypoints[:, 1]
    
    # Plot keypoints
    plt.scatter(kp_x, kp_y, c='r', s=5, marker='o')
    
    plt.title(f'{title} ({len(keypoints)} points)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.show()


def build_line(x, y, num_points):
    """
    Fit a line to a set of points using least squares.
    
    Parameters:
    -----------
    x : array_like
        X coordinates
    y : array_like
        Y coordinates
    num_points : int
        Number of points
        
    Returns:
    --------
    tuple
        (errors, slope, intercept) - Fitting errors and line parameters
    """
    x = np.array(x)
    y = np.array(y)
    
    # Calculate mean values
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # Calculate slope
    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sum((x - mean_x)**2)
    
    if denominator == 0:
        # Vertical line case
        slope = float('inf')
        intercept = mean_x
    else:
        slope = numerator / denominator
        intercept = mean_y - slope * mean_x
    
    # Calculate errors
    errors = np.zeros(num_points)
    for i in range(num_points):
        if slope == float('inf'):
            # For vertical line, error is horizontal distance
            errors[i] = abs(x[i] - intercept)
        else:
            # For other lines, error is perpendicular distance
            errors[i] = abs(y[i] - (slope * x[i] + intercept)) / np.sqrt(1 + slope**2)
            
    return errors, slope, intercept


def point_refine(image, keypoints, n, sigma):
    """
    Refine keypoint positions using local line fitting.
    
    Parameters:
    -----------
    image : ndarray
        Input image
    keypoints : ndarray
        Array of keypoints [x, y, scale, layer, orientation, ...]
    n : int
        Neighborhood size
    sigma : float
        Scale parameter
        
    Returns:
    --------
    ndarray
        Refined keypoints
    """
    num_keypoints = keypoints.shape[0]
    processed = np.zeros(num_keypoints, dtype=bool)
    refined_keypoints = keypoints.copy()
    
    for i in range(num_keypoints):
        if processed[i]:
            continue
            
        x = keypoints[i, 0]
        y = keypoints[i, 1]
        scale = keypoints[i, 3]  # Layer (not scale value)
        orientation = keypoints[i, 4]
        
        # Initialize lists for connected keypoints
        x_connected = [x]
        y_connected = [y]
        idx_connected = [i]
        
        # Find connected keypoints with similar orientation
        for j in range(num_keypoints):
            if j == i or processed[j]:
                continue
                
            x_j = keypoints[j, 0]
            y_j = keypoints[j, 1]
            scale_j = keypoints[j, 3]
            orientation_j = keypoints[j, 4]
            
            # Check if orientation is similar (within 10 degrees)
            if abs(orientation - orientation_j) < 10 or abs(orientation - orientation_j) > 170:
                # Check if keypoint is within neighborhood
                if np.sqrt((x - x_j)**2 + (y - y_j)**2) <= n * sigma:
                    x_connected.append(x_j)
                    y_connected.append(y_j)
                    idx_connected.append(j)
                    processed[j] = True
        
        # If less than 4 connected keypoints, skip refinement
        if len(x_connected) < 4:
            continue
            
        # Fit line to connected keypoints
        errors, slope, intercept = build_line(x_connected, y_connected, len(x_connected))
        
        # Refine keypoint positions
        for k, idx in enumerate(idx_connected):
            if errors[k] > 0.5:  # Only adjust if error is significant
                if len(np.where(np.array(x_connected) == x_connected[k])[0]) > 1:
                    # Multiple keypoints with same x, adjust y
                    if slope != 0:
                        refined_keypoints[idx, 1] = slope * x_connected[k] + intercept
                else:
                    # Otherwise, adjust x
                    if slope != 0:
                        refined_keypoints[idx, 0] = (y_connected[k] - intercept) / slope
            
            # Round to integer coordinates
            refined_keypoints[idx, 0] = round(refined_keypoints[idx, 0])
            refined_keypoints[idx, 1] = round(refined_keypoints[idx, 1])
    
    return refined_keypoints


def append_images(img1, img2, pts1, pts2):
    """
    Create a side-by-side visualization of two images with matched keypoints.
    
    Parameters:
    -----------
    img1, img2 : ndarray
        Input images
    pts1, pts2 : ndarray
        Matched keypoints
        
    Returns:
    --------
    ndarray
        Visualization image
    """
    # Ensure images are in color format
    if len(img1.shape) == 2:
        img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    else:
        img1_color = img1.copy()
        
    if len(img2.shape) == 2:
        img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    else:
        img2_color = img2.copy()
    
    # Create a new image big enough to contain both images side by side
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    h = max(h1, h2)
    w = w1 + w2
    
    # Create empty image
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Copy images
    vis[:h1, :w1] = img1_color
    vis[:h2, w1:w1+w2] = img2_color
    
    # Draw lines between matched keypoints
    for i in range(len(pts1)):
        pt1 = (int(pts1[i, 0]), int(pts1[i, 1]))
        pt2 = (int(pts2[i, 0]) + w1, int(pts2[i, 1]))
        cv2.line(vis, pt1, pt2, (0, 255, 255), 1)
        cv2.circle(vis, pt1, 3, (0, 0, 255), -1)
        cv2.circle(vis, pt2, 3, (0, 0, 255), -1)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(f'Matched Features ({len(pts1)} matches)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('matched_features.png')
    plt.show()
    
    return vis