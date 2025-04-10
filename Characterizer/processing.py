#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
High-level processing functions for handling different registration scenarios.

This module provides functions to process image sets from folders, handling
different file structures and automatically matching corresponding files.
"""

import os
import cv2
import numpy as np
import logging
from pathlib import Path
from .detectors import create_detector_and_matcher, RIFT_AVAILABLE, LGHD_AVAILABLE, SARSIFT_AVAILABLE, MINIMA_AVAILABLE
from .registration import process_image_pair_with_gt
from .utils import load_ground_truth, find_matching_files_in_folder
from .reporting import save_metrics, compare_methods, create_summary_report
from .visualization import visualize_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_from_folder(folder_path, methods, output_dir, tag=None, ratio_thresh=0.7, visualize=True, model=None):
    """
    Process sets of images from a folder containing SAR, optical images, and ground truth.
    
    Args:
        folder_path (str): Path to folder containing the files
        methods (list): List of registration methods to use
        output_dir (str): Output directory for results
        tag (str, optional): Specific tag/prefix to process (e.g., 'SO', 'CS')
        ratio_thresh (float): Threshold for Lowe's ratio test
        visualize (bool): Whether to generate visualizations
        
    Returns:
        dict: Results by set and method
    """
    # Find matching file sets
    matching_sets = find_matching_files_in_folder(folder_path, tag)
    
    if not matching_sets:
        logger.warning("No matching file sets found for processing.")
        return {}
    
    # Process each set
    results_by_set = {}
    
    for file_set in matching_sets:
        results_by_method = _process_single_set(
            file_set, methods, output_dir, ratio_thresh, visualize, model)
        
        
        set_name = f"{file_set['tag']}{file_set['number']}"
        results_by_set[set_name] = results_by_method
    if isinstance(model, list):
        # Crea una nuova lista dove per ogni elemento in methods
        # Se l'elemento è "MINIMA", concatenalo con ogni elemento di model
        # Altrimenti, mantieni l'elemento originale
        new_methods = []
        for m in methods:
            if m.upper() == "MINIMA":
                for mod in model:
                    new_methods.append(m + "_" + mod)
            else:
                new_methods.append(m)
        methods = new_methods    # Create summary report
    _create_summary(results_by_set, methods, output_dir)
    
    return results_by_set


def _process_single_set(file_set, methods, output_dir, ratio_thresh, visualize, model=None):
    """
    Process a single set of images (SAR, optical, ground truth).
    
    Args:
        file_set (dict): Information about the file set to process
        methods (list): List of registration methods to use
        output_dir (str): Base output directory
        ratio_thresh (float): Threshold for Lowe's ratio test
        visualize (bool): Whether to generate visualizations
        
    Returns:
        dict: Results by method
    """
    set_name = f"{file_set['tag']}{file_set['number']}"
   # logger.info(f"\nProcessing set: {set_name}")
    
    # Create dedicated output directory for this set
    set_output_dir = Path(output_dir) / set_name
    set_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load ground truth
    try:
        _, _, landmarks_fix, landmarks_mov, transform_gt = load_ground_truth(file_set['gt_file'])
        logger.info(f"Ground truth data loaded: {len(landmarks_fix)} landmark pairs")
        
    except Exception as e:
        logger.error(f"Error loading ground truth for set {set_name}: {e}")
        return {}
    
    # Process with each method
    results_by_method = {}
    
    for method in methods:
        logger.info(f"methods: {method}")
        if method.upper() == "MINIMA" and MINIMA_AVAILABLE and isinstance(model, list):
            for m in model:
                method_results = _process_with_method(
                    method, file_set, landmarks_mov, landmarks_fix, 
                    transform_gt, ratio_thresh, set_output_dir, visualize, model=m
                )
                if method_results:
                    results_by_method[method + "_" + m] = method_results
        else:
            method_results = _process_with_method(
                method, file_set, landmarks_mov, landmarks_fix, 
                transform_gt, ratio_thresh, set_output_dir, visualize, model=model
            )
            if method_results:
                results_by_method[method] = method_results
        print(results_by_method)
    
    # Compare methods for this set
    if results_by_method:
        compare_methods(results_by_method, str(set_output_dir))
    
    return results_by_method


def _process_with_method(method, file_set, landmarks_mov, landmarks_fix, 
                       transform_gt, ratio_thresh, output_dir, visualize, model):
    """
    Process a file set using a specific registration method.
    
    Args:
        method (str): Registration method to use
        file_set (dict): Information about the file set
        landmarks_mov, landmarks_fix: Ground truth landmarks
        transform_gt: Ground truth transformation matrix
        ratio_thresh (float): Threshold for Lowe's ratio test
        output_dir (Path): Output directory for results
        visualize (bool): Whether to generate visualizations
        
    Returns:
        dict: Registration results for this method
    """
    logger.info(f"\n==== Processing with {method} for set {file_set['tag']}{file_set['number']} ====")
    if method.upper() == "MINIMA" and MINIMA_AVAILABLE:
        logger.info(f"Using MINIMA method: {model}")
    try:
        # Skip unavailable methods
        if method.upper() == "RIFT" and not RIFT_AVAILABLE:
            logger.warning(f"Skipping RIFT method as it's not available")
            return None
        
        if method.upper() == "LGHD" and not LGHD_AVAILABLE:
            logger.warning(f"Skipping LGHD method as it's not available")
            return None
            
        if method.upper() == "SARSIFT" and not SARSIFT_AVAILABLE:
            logger.warning(f"Skipping SAR-SIFT method as it's not available")
            return None
        
        if method.upper() == "MINIMA" and not MINIMA_AVAILABLE:
            logger.warning(f"Skipping MINIMA method as it's not available")
            return None
        # Create detector and matcher
        logger.info(f"Creating detector and matcher for {method}")
        detector, matcher = create_detector_and_matcher(method)
        
        # Process the image pair
        results = process_image_pair_with_gt(
            file_set['sar_file'], file_set['opt_file'], detector, matcher,
            landmarks_mov, landmarks_fix, transform_gt, ratio_thresh, method, model
        )
        
        # Log results
        _log_results(results)
        
        # Save metrics
        save_metrics(results, method, str(output_dir))
        
        # Visualize results if requested
        if visualize:
            sar_img = cv2.imread(file_set['sar_file'], cv2.IMREAD_GRAYSCALE)
            opt_img = cv2.imread(file_set['opt_file'], cv2.IMREAD_GRAYSCALE)
            visualize_results(sar_img, opt_img, results, method, str(output_dir))
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing with {method}: {e}")
        import traceback
        traceback.print_exc()
        return None


def _log_results(results):
    """Log key metrics from registration results."""

    
    if 'matrix_rmse' in results and results['matrix_rmse'] is not None:
        # Handle numpy array case
        if isinstance(results['matrix_rmse'], np.ndarray):
            logger.info(f"Matrix RMSE: {float(results['matrix_rmse']):.6f}")
        else:
            logger.info(f"Matrix RMSE: {results['matrix_rmse']:.6f}")
    else:
        logger.info("Matrix RMSE: N/A")
        
    logger.info(f"Execution time: {results['execution_time']:.4f} sec")
    logger.info(f"Power_consumption: {results['power']:.4f} J")


def _create_summary(results_by_set, methods, output_dir):
    """Create a summary of all results."""
    logger.info("\n==== Processing Summary ====")
    logger.info(f"Sets processed: {len(results_by_set)}")
    logger.info(f"Methods used: {', '.join(methods)}")
    
    # Create summary report
    create_summary_report(results_by_set, methods, output_dir)