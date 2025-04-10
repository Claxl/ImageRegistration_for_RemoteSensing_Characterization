#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for the image registration framework.

This module provides utility functions for:
- File handling and searching
- Ground truth data loading
- Result visualization (other funzioni ausiliarie sono definite in moduli specifici)
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
        mat_pattern = re.compile(f"^{current_tag}(\\d+)\\.mat$")
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


# La funzione load_ground_truth (e altre eventualmente utili) devono rimanere se
# sono usate dagli altri moduli. Qui si mostra ad esempio load_ground_truth se presente.

def load_ground_truth(gt_filepath):
    """
    Carica i dati di ground truth da un file .mat o altro formato previsto.
    
    Args:
        gt_filepath (str): percorso del file ground truth
        
    Returns:
        tuple: valori opportuni (ad esempio, landmark fissi, landmark mobili, matrice di trasformazione, etc.)
    """
    # Implementazione di esempio (da adattare in base al formato reale del file)
    # …
    # Per semplicità, restituiamo valori fittizi
    landmarks_fix = np.array([[100, 150], [200, 250]])
    landmarks_mov = np.array([[102, 148], [198, 252]])
    transform_gt = np.eye(3)
    return None, None, landmarks_fix, landmarks_mov, transform_gt
