#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for the image registration framework.

This module provides utility functions for:
- File handling and searching
- Image pair matching
- Ground truth data loading
- Result visualization
"""

import os
import re
import glob
import logging
import numpy as np
import cv2
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_image_files(folder, extensions=None):
    """
    Get a sorted list of image file paths from a folder.
    
    Args:
        folder (str): Path to the folder containing images
        extensions (list, optional): List of file extensions to search for.
            Defaults to ['.jpg', '.png', '.jpeg', '.tif', '.tiff']
            
    Returns:
        list: Sorted list of image file paths
    """
    if extensions is None:
        extensions = ['.jpg', '.png', '.jpeg', '.tif', '.tiff']
    
    folder_path = Path(folder)
    if not folder_path.exists():
        logger.warning(f"Folder not found: {folder}")
        return []
    
    # Find all files with specified extensions
    image_files = []
    for ext in extensions:
        # Remove leading dot if present
        if ext.startswith('.'):
            pattern = f"*{ext}"
        else:
            pattern = f"*.{ext}"
            
        image_files.extend(folder_path.glob(pattern))
    
    # Sort files by name
    return sorted(image_files)


def extract_number(filename):
    """
    Extract the identifier part from filenames to match files across directories.
    
    This function tries different patterns to extract identifiers:
    1. Pattern like 'ROIs1970_fall_s1_8_p99.png' -> extracts '8_p99'
    2. Pattern with number before '_p' -> extracts that number
    3. Any number in the filename
    
    Args:
        filename (str): The filename to extract the identifier from
        
    Returns:
        str or None: The extracted identifier, or None if no match is found
    """
    if not isinstance(filename, str):
        filename = str(filename)
    
    # Try different pattern matching strategies
    for pattern, group_idx in [
        (r'_(\d+_p\d+)', 1),  # 'ROIs1970_fall_s1_8_p99.png' -> '8_p99'
        (r'_(\d+)_p', 1),     # Extract number before '_p'
        (r'(\d+)', 0)         # Any number in the filename
    ]:
        match = re.search(pattern, filename)
        if match:
            return match.group(group_idx)
    
    logger.debug(f"No identifier found in filename: {filename}")
    return None


def find_matching_files_in_folder(folder_path, tag=None):
    """
    Find matching files in a folder with the following naming pattern:
    - TAGN.mat: ground truth file (e.g., SO1.mat)
    - TAGNa.png: SAR image (e.g., SO1a.png)
    - TAGNb.png: optical image (e.g., SO1b.png)
    
    Args:
        folder_path (str): Path to the folder containing the files
        tag (str, optional): Specific tag/prefix to search for (e.g., 'SO', 'CS')
            If None, searches all standard tags
            
    Returns:
        list: List of dictionaries containing file information for each matching set
    """
    folder = Path(folder_path)
    if not folder.exists():
        logger.error(f"The folder {folder_path} does not exist.")
        return []
    
    # Standard tags used in the dataset
    available_tags = ["CS", "DN", "DO", "IO", "MO", "OO", "SO"]
    
    # Use only the specified tag if provided and valid
    tags_to_use = [tag] if tag and tag in available_tags else available_tags
    
    # Get all files in the folder
    all_files = list(folder.iterdir())
    all_filenames = [f.name for f in all_files]
    
    matching_sets = []
    
    # Process each potential tag
    for current_tag in tags_to_use:
        # Find .mat files with the current tag using regex pattern
        mat_pattern = re.compile(f"^{current_tag}(\\d+)\.mat$")
        mat_files = [f for f in all_filenames if mat_pattern.match(f)]
        
        for mat_file in mat_files:
            # Extract number from .mat filename
            match = mat_pattern.match(mat_file)
            if not match:
                continue
                
            number = match.group(1)
            
            # Look for corresponding image files
            sar_pattern = f"{current_tag}{number}a"  # e.g., SO1a
            opt_pattern = f"{current_tag}{number}b"  # e.g., SO1b
            
            # Find matching SAR and optical files
            sar_files = _find_matching_image_files(all_filenames, sar_pattern)
            opt_files = _find_matching_image_files(all_filenames, opt_pattern)
            
            if sar_files and opt_files:
                #_log_matching_set(current_tag, number, mat_file, sar_files[0], opt_files[0])
                
                matching_sets.append({
                    'tag': current_tag,
                    'number': number,
                    'gt_file': str(folder / mat_file),
                    'sar_file': str(folder / sar_files[0]),
                    'opt_file': str(folder / opt_files[0])
                })
    
    if not matching_sets:
        _log_no_matches(folder_path, tag)
    
    return matching_sets


def _find_matching_image_files(all_filenames, pattern):
    """Find image files matching a pattern."""
    return [f for f in all_filenames if 
            f.startswith(pattern) and 
            any(f.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"])]


def _log_matching_set(tag, number, mat_file, sar_file, opt_file):
    """Log information about a matching set of files."""
    logger.info(f"Set found: {tag}{number}")
    logger.info(f"  Ground Truth: {mat_file}")
    logger.info(f"  SAR: {sar_file}")
    logger.info(f"  Optical: {opt_file}")


def _log_no_matches(folder_path, tag):
    """Log a message when no matching sets are found."""
    msg = f"No matching sets found in folder {folder_path}"
    if tag:
        msg += f" for tag {tag}"
    logger.warning(msg)


def make_match_image(im1, im2, pts1, pts2, color=(0, 255, 255), radius=5, thickness=2):
    """
    Create a visualization image showing matches between two images.
    
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
    # Ensure inputs are valid
    if im1 is None or im2 is None or pts1 is None or pts2 is None:
        logger.error("Invalid inputs to make_match_image")
        return None
    
    if len(pts1) != len(pts2):
        logger.error(f"Point count mismatch: {len(pts1)} vs {len(pts2)}")
        return None
    
    # Ensure both images are in BGR format for visualization
    im1_bgr = _ensure_bgr(im1)
    im2_bgr = _ensure_bgr(im2)

    # Get dimensions
    h1, w1 = im1_bgr.shape[:2]
    h2, w2 = im2_bgr.shape[:2]

    # Create canvas for side-by-side display
    height = max(h1, h2)
    width = w1 + w2
    match_vis = np.zeros((height, width, 3), dtype=np.uint8)

    # Place images on canvas
    match_vis[:h1, :w1] = im1_bgr
    match_vis[:h2, w1:w1 + w2] = im2_bgr

    # Draw each match
    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        # Adjust x2 because the second image is shifted to the right by w1
        x2_shifted = x2 + w1

        try:
            # Draw circles at the matched points
            cv2.circle(match_vis, (int(x1), int(y1)), radius, color, thickness)
            cv2.circle(match_vis, (int(x2_shifted), int(y2)), radius, color, thickness)

            # Draw a line between them
            cv2.line(match_vis, (int(x1), int(y1)), (int(x2_shifted), int(y2)), color, thickness)
        except Exception as e:
            logger.warning(f"Error drawing match from ({x1}, {y1}) to ({x2}, {y2}): {e}")
            continue

    return match_vis


def _ensure_bgr(img):
    """Ensure image is in BGR format."""
    if img is None:
        return None
        
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 3:
        return img.copy()
    elif img.shape[2] == 4:  # RGBA
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        logger.warning(f"Unexpected image format with shape {img.shape}")
        return img


def load_ground_truth(mat_file_path):
    """
    Load ground truth data from a .mat file.
    
    Args:
        mat_file_path (str): Path to the .mat file containing ground truth data
        
    Returns:
        tuple: (I_fix, I_move, landmarks_fix, landmarks_mov, T)
            - I_fix: Fixed (optical) image from ground truth
            - I_move: Moving (SAR) image from ground truth
            - landmarks_fix: Landmark points for optical image
            - landmarks_mov: Landmark points for SAR image
            - T: Ground truth transformation matrix (if available)
            
    Raises:
        IOError: If there's an error loading the ground truth data
    """
    try:
        import scipy.io
        
        logger.info(f"Loading ground truth from {mat_file_path}")
        mat_data = scipy.io.loadmat(mat_file_path)
        
        # Extract images
        I_fix = _extract_and_normalize_image(mat_data, 'I_fix')
        I_move = _extract_and_normalize_image(mat_data, 'I_move')
        
        # Extract landmark points
        landmarks = mat_data['Landmarks']
        landmarks_fix = landmarks[0][0][0]
        landmarks_mov = landmarks[0][0][1]
        
        logger.info(f"Loaded {len(landmarks_fix)} landmark pairs")
        
        # Extract transformation matrix if available
        T = _extract_transformation_matrix(mat_data)
        
        return I_fix, I_move, landmarks_fix, landmarks_mov, T
        
    except Exception as e:
        logger.error(f"Error loading ground truth data: {e}")
        raise IOError(f"Error loading ground truth data: {e}")


def _extract_and_normalize_image(mat_data, image_key):
    """Extract and normalize an image from the MAT data."""
    img = mat_data[image_key]
    
    # Ensure image is grayscale
    if len(img.shape) > 2:
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        img = img.astype(np.uint8)
        
    return img


def _extract_transformation_matrix(mat_data):
    """Extract transformation matrix from MAT data if available."""
    if 'T' in mat_data:
        T = mat_data['T']
        logger.info(f"Ground truth transformation matrix extracted: shape {T.shape}")
        return T
    else:
        logger.warning("Ground truth transformation matrix 'T' not found in the .mat file")
        return None


def save_results(sar_path, opt_path, registered_img, matches_img, method, output_dir="output"):
    """
    Save the registered image and matches visualization to the output directory.
    
    Args:
        sar_path (str): Path to the SAR image
        opt_path (str): Path to the optical image
        registered_img (np.ndarray): Registered (warped) SAR image
        matches_img (np.ndarray): Visualization of feature matches
        method (str): Method name used for registration (e.g., 'SIFT', 'RIFT')
        output_dir (str): Output directory for saving results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract base filenames
    base_sar = Path(sar_path).stem
    base_opt = Path(opt_path).stem
    
    # Create output filenames
    reg_filename = output_path / f"{method}_registered_{base_sar}_to_{base_opt}.png"
    matches_filename = output_path / f"{method}_matches_{base_sar}_to_{base_opt}.png"
    
    # Save images if available
    if registered_img is not None:
        try:
            cv2.imwrite(str(reg_filename), registered_img)
            logger.info(f"Saved registered image to {reg_filename}")
        except Exception as e:
            logger.error(f"Error saving registered image: {e}")
    
    if matches_img is not None:
        try:
            cv2.imwrite(str(matches_filename), matches_img)
            logger.info(f"Saved matches image to {matches_filename}")
        except Exception as e:
            logger.error(f"Error saving matches image: {e}")