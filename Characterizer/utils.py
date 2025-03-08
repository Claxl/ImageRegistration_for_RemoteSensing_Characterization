#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for the image registration framework.
"""

import os
import re
import glob
import numpy as np
import cv2

def get_image_files(folder, extensions=['*.jpg', '*.png', '*.jpeg']):
    """
    Returns a sorted list of image file paths from the given folder.
    
    Args:
        folder (str): Path to the folder containing images
        extensions (list): List of file extensions to search for
        
    Returns:
        list: Sorted list of image file paths
    """
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)

def extract_number(filename):
    """
    Extracts the identifier part from filenames to match files across directories.
    
    This function tries different patterns to extract identifiers:
    1. Pattern like 'ROIs1970_fall_s1_8_p99.png' -> extracts '8_p99'
    2. Pattern with number before '_p' -> extracts that number
    3. Any number in the filename
    
    Args:
        filename (str): The filename to extract the identifier from
        
    Returns:
        str or None: The extracted identifier, or None if no match is found
    """
    # For filenames like 'ROIs1970_fall_s1_8_p99.png'
    match = re.search(r'_(\d+_p\d+)', filename)
    if match:
        return match.group(1)
    
    # Fallback to extracting just the number before '_p'
    match = re.search(r'_(\d+)_p', filename)
    if match:
        return match.group(1)
    
    # Final fallback to any number
    match = re.search(r'\d+', filename)
    if match:
        return match.group()
    
    return None

def find_matching_files_in_folder(folder_path, tag=None):
    """
    Finds matching files in a folder with the following naming pattern:
    - TAGN.mat: ground truth file (e.g., SO1.mat)
    - TAGNa.png: SAR image (e.g., SO1a.png)
    - TAGNb.png: optical image (e.g., SO1b.png)
    
    Args:
        folder_path (str): Path to the folder containing the files
        tag (str, optional): Specific tag/prefix to search for (e.g., 'SO', 'CS'), if None, searches all tags
        
    Returns:
        list: List of dictionaries containing file information for each matching set
    """
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return []
    
    # List of available tags
    available_tags = ["CS", "DN", "DO", "IO", "MO", "OO", "SO"]
    
    # Use only the specified tag if provided
    tags_to_use = [tag] if tag and tag in available_tags else available_tags
    
    # Get all files in the folder
    all_files = os.listdir(folder_path)
    
    matching_sets = []
    
    # For each potential tag
    for current_tag in tags_to_use:
        # Find .mat files with the current tag
        mat_pattern = re.compile(f"^{current_tag}(\\d+)\.mat$")
        mat_files = [f for f in all_files if mat_pattern.match(f)]
        
        for mat_file in mat_files:
            # Extract number from .mat filename
            match = mat_pattern.match(mat_file)
            if not match:
                continue
                
            number = match.group(1)
            
            # Look for corresponding image files
            sar_pattern = f"{current_tag}{number}a"  # e.g., SO1a
            opt_pattern = f"{current_tag}{number}b"  # e.g., SO1b
            
            # Find SAR file
            sar_files = [f for f in all_files if f.startswith(sar_pattern) and 
                         (f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg"))]
            
            # Find optical file
            opt_files = [f for f in all_files if f.startswith(opt_pattern) and 
                         (f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg"))]
            
            if sar_files and opt_files:
                sar_file = sar_files[0]
                opt_file = opt_files[0]
                
                matching_sets.append({
                    'tag': current_tag,
                    'number': number,
                    'gt_file': os.path.join(folder_path, mat_file),
                    'sar_file': os.path.join(folder_path, sar_file),
                    'opt_file': os.path.join(folder_path, opt_file)
                })
                
                print(f"Set found: {current_tag}{number}")
                print(f"  Ground Truth: {mat_file}")
                print(f"  SAR: {sar_file}")
                print(f"  Optical: {opt_file}")
    
    if not matching_sets:
        print(f"No matching sets found in folder {folder_path}")
        if tag:
            print(f"for tag {tag}")
    
    return matching_sets

def make_match_image(im1, im2, pts1, pts2, color=(0, 255, 255), radius=5, thickness=2):
    """
    Creates a visualization image showing matches between two images.
    
    Args:
        im1 (np.ndarray): First image (BGR or grayscale)
        im2 (np.ndarray): Second image (BGR or grayscale)
        pts1 (np.ndarray): Array of matched keypoints in image1 (x, y)
        pts2 (np.ndarray): Array of matched keypoints in image2 (x, y)
        color (tuple): BGR color for circles and lines
        radius (int): Radius of the circle for each keypoint
        thickness (int): Thickness of circles and lines
        
    Returns:
        np.ndarray: Visualization image with both images side-by-side and matches drawn
    """
    # Ensure both images are in BGR format for visualization
    if len(im1.shape) == 2:
        im1 = cv2.cvtColor(im1, cv2.COLOR_GRAY2BGR)
    if len(im2.shape) == 2:
        im2 = cv2.cvtColor(im2, cv2.COLOR_GRAY2BGR)

    # Get dimensions
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]

    # Create canvas for side-by-side display
    match_vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)

    # Place images on canvas
    match_vis[:h1, :w1] = im1
    match_vis[:h2, w1:w1 + w2] = im2

    # Draw each match
    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        # Adjust x2 because the second image is shifted to the right by w1
        x2_shifted = x2 + w1

        # Draw circles at the matched points
        cv2.circle(match_vis, (int(x1), int(y1)), radius, color, thickness)
        cv2.circle(match_vis, (int(x2_shifted), int(y2)), radius, color, thickness)

        # Draw a line between them
        cv2.line(match_vis, (int(x1), int(y1)), (int(x2_shifted), int(y2)), color, thickness)

    return match_vis

def load_ground_truth(mat_file_path):
    """
    Loads ground truth data from a .mat file.
    
    Args:
        mat_file_path (str): Path to the .mat file containing ground truth data
        
    Returns:
        tuple: (I_fix, I_move, landmarks_fix, landmarks_mov, T)
            - I_fix: Fixed (optical) image from ground truth
            - I_move: Moving (SAR) image from ground truth
            - landmarks_fix: Landmark points for optical image
            - landmarks_mov: Landmark points for SAR image
            - T: Ground truth transformation matrix (if available)
    """
    try:
        import scipy.io
        
        mat_data = scipy.io.loadmat(mat_file_path)
        I_fix = mat_data['I_fix']
        I_move = mat_data['I_move']
        landmarks = mat_data['Landmarks']
        
        # Extract landmark points
        landmarks_fix = landmarks[0][0][0]
        landmarks_mov = landmarks[0][0][1]
        
        # Extract transformation matrix if available
        T = None
        if 'T' in mat_data:
            T = mat_data['T']
            print(f"Ground truth transformation matrix extracted: shape {T.shape}")
        else:
            print("Warning: Ground truth transformation matrix 'T' not found in the .mat file")
        
        # Ensure images are grayscale
        if len(I_fix.shape) > 2:
            I_fix = cv2.cvtColor(I_fix.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            I_fix = I_fix.astype(np.uint8)
        
        if len(I_move.shape) > 2:
            I_move = cv2.cvtColor(I_move.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            I_move = I_move.astype(np.uint8)
        
        return I_fix, I_move, landmarks_fix, landmarks_mov, T
    except Exception as e:
        raise IOError(f"Error loading ground truth data: {e}")

def save_results(sar_path, opt_path, registered_img, matches_img, method, output_dir="output"):
    """
    Saves the registered image and matches visualization to the output directory.
    
    Args:
        sar_path (str): Path to the SAR image
        opt_path (str): Path to the optical image
        registered_img (np.ndarray): Registered (warped) SAR image
        matches_img (np.ndarray): Visualization of feature matches
        method (str): Method name used for registration (e.g., 'SIFT', 'RIFT')
        output_dir (str): Output directory for saving results
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    base_sar = os.path.splitext(os.path.basename(sar_path))[0]
    base_opt = os.path.splitext(os.path.basename(opt_path))[0]
    
    reg_filename = os.path.join(output_dir, f"{method}_registered_{base_sar}_to_{base_opt}.png")
    matches_filename = os.path.join(output_dir, f"{method}_matches_{base_sar}_to_{base_opt}.png")
    
    if registered_img is not None:
        cv2.imwrite(reg_filename, registered_img)
    if matches_img is not None:
        cv2.imwrite(matches_filename, matches_img)