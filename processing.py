#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
High-level processing functions for handling different registration scenarios.
"""

import os
import cv2
import numpy as np
from detectors import create_detector_and_matcher, RIFT_AVAILABLE
from registration import process_image_pair_with_gt
from utils import load_ground_truth, find_matching_files_in_folder
from metrics import compute_transformation_error
from reporting import save_metrics, compare_methods, create_summary_report
from visualization import visualize_results

def process_with_ground_truth(mat_file, methods, output_dir, ratio_thresh=0.7, visualize=True):
    """
    Processes images using ground truth data from a .mat file.
    
    Args:
        mat_file (str): Path to the .mat file containing ground truth data
        methods (list): List of methods to use (e.g., ['SIFT', 'SURF', 'ORB'])
        output_dir (str): Output directory for results
        ratio_thresh (float): Threshold for Lowe's ratio test
        visualize (bool): Whether to generate visualizations
        
    Returns:
        dict: Dictionary of results by method
    """
    # Load ground truth data
    try:
        opt_img, sar_img, landmarks_fix, landmarks_mov, transform_gt = load_ground_truth(mat_file)
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        return
    
    print(f"Loaded ground truth data with {len(landmarks_fix)} landmark pairs")
    if transform_gt is not None:
        print(f"Ground truth transformation matrix loaded")
    
    # Process with each method
    results_by_method = {}
    for method in methods:
        print(f"\n==== Processing using {method} ====")
        try:
            # Skip RIFT if not available
            if method.upper() == "RIFT" and not RIFT_AVAILABLE:
                print(f"Skipping RIFT method as it's not available")
                continue
                
            detector, matcher = create_detector_and_matcher(method)
            results = process_image_pair_with_gt(sar_img, opt_img, detector, matcher, 
                                              landmarks_mov, landmarks_fix, ratio_thresh)
            
            # Add transformation error if ground truth is available
            if transform_gt is not None and results['transformation_matrix'] is not None:
                transform_errors = compute_transformation_error(
                    results['transformation_matrix'], transform_gt)
                results.update(transform_errors)
            
            results_by_method[method] = results
            
            # Print key metrics
            print(f"Total points detected: {results['total_points']}")
            print(f"Repeatable points: {results['repeatable_points']}")
            print(f"Ground truth matches: {results['gt_matches']}")
            print(f"Number of matches (NM): {results['NM']}")
            print(f"Number of correct matches (NCM): {results['NCM']}")
            print(f"Ratio NM/NCM: {results['ratio']:.4f}")
            
            if 'rmse' in results and results['rmse'] is not None:
                print(f"RMSE between landmarks: {results['rmse']:.4f}")
            elif 'rmse_landmarks' in results and results['rmse_landmarks'] is not None:
                print(f"RMSE between landmarks: {results['rmse_landmarks']:.4f}")
            else:
                print("RMSE between landmarks: N/A")
            
            if results['mutual_information'] is not None:
                print(f"Mutual Information: {results['mutual_information']:.4f}")
            else:
                print("Mutual Information: N/A")
            
            # Print transformation error metrics if available
            if 'transform_error' in results and results['transform_error'] is not None:
                print(f"Transform error: {results['transform_error']:.4f}")
                print(f"Matrix RMSE: {results['rmse']:.4f}")
                print(f"Frobenius norm: {results['frobenius']:.4f}")
            
            # Save metrics
            save_metrics(results, method, output_dir)
            
            # Visualize results
            if visualize:
                visualize_results(sar_img, opt_img, results, landmarks_mov, landmarks_fix, method, output_dir, transform_gt)
        
        except Exception as e:
            print(f"Error processing with {method}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create comparison of methods
    compare_methods(results_by_method, output_dir)
    
    return results_by_method

def process_from_folder(folder_path, methods, output_dir, tag=None, ratio_thresh=0.7, visualize=True):
    """
    Process images and ground truth files from a single folder.
    
    Args:
        folder_path (str): Path to the folder containing the files
        methods (list): List of methods to use (e.g., ['SIFT', 'SURF', 'ORB'])
        output_dir (str): Output directory for results
        tag (str, optional): Specific tag/prefix to process (e.g., 'SO', 'CS')
        ratio_thresh (float): Threshold for Lowe's ratio test
        visualize (bool): Whether to generate visualizations
        
    Returns:
        dict: Dictionary of results by set and method
    """
    # Find matching file sets
    matching_sets = find_matching_files_in_folder(folder_path, tag)
    
    if not matching_sets:
        print("No matching file sets found for processing.")
        return
    
    # Process each set
    results_by_set = {}
    
    for file_set in matching_sets:
        set_name = f"{file_set['tag']}{file_set['number']}"
        print(f"\nProcessing set: {set_name}")
        
        # Create dedicated output directory for this set
        set_output_dir = os.path.join(output_dir, set_name)
        if not os.path.exists(set_output_dir):
            os.makedirs(set_output_dir)
        
        # Load images
        sar_img = cv2.imread(file_set['sar_file'], cv2.IMREAD_GRAYSCALE)
        opt_img = cv2.imread(file_set['opt_file'], cv2.IMREAD_GRAYSCALE)
        
        if sar_img is None or opt_img is None:
            print(f"Error loading images for set {set_name}")
            continue
        
        # Load ground truth
        try:
            _, _, landmarks_fix, landmarks_mov, transform_gt = load_ground_truth(file_set['gt_file'])
        except Exception as e:
            print(f"Error loading ground truth for set {set_name}: {e}")
            continue
        
        # Process with each method
        results_by_method = {}
        
        for method in methods:
            print(f"\n==== Processing with {method} for set {set_name} ====")
            try:
                # Skip RIFT if not available
                if method.upper() == "RIFT" and not RIFT_AVAILABLE:
                    print(f"Skipping RIFT method as it's not available")
                    continue
                
                detector, matcher = create_detector_and_matcher(method)
                results = process_image_pair_with_gt(sar_img, opt_img, detector, matcher, 
                                                  landmarks_mov, landmarks_fix, ratio_thresh)
                
                # Add transformation error if ground truth is available
                if transform_gt is not None and results['transformation_matrix'] is not None:
                    transform_errors = compute_transformation_error(
                        results['transformation_matrix'], transform_gt)
                    results.update(transform_errors)
                
                results_by_method[method] = results
                
                # Print key metrics
                print(f"Total points detected: {results['total_points']}")
                print(f"Repeatable points: {results['repeatable_points']}")
                print(f"Ground truth matches: {results['gt_matches']}")
                print(f"Number of matches (NM): {results['NM']}")
                print(f"Number of correct matches (NCM): {results['NCM']}")
                print(f"Ratio NM/NCM: {results['ratio']:.4f}")
                
                if 'rmse' in results and results['rmse'] is not None:
                    print(f"RMSE between landmarks: {results['rmse']:.4f}")
                elif 'rmse_landmarks' in results and results['rmse_landmarks'] is not None:
                    print(f"RMSE between landmarks: {results['rmse_landmarks']:.4f}")
                else:
                    print("RMSE between landmarks: N/A")
                
                if results['mutual_information'] is not None:
                    print(f"Mutual Information: {results['mutual_information']:.4f}")
                else:
                    print("Mutual Information: N/A")
                
                # Print transformation error metrics if available
                if 'transform_error' in results and results['transform_error'] is not None:
                    print(f"Transform error: {results['transform_error']:.4f}")
                    print(f"Matrix RMSE: {results['rmse']:.4f}")
                    print(f"Frobenius norm: {results['frobenius']:.4f}")
                
                # Save metrics
                save_metrics(results, method, set_output_dir)
                
                # Visualize results
                if visualize:
                    visualize_results(sar_img, opt_img, results, landmarks_mov, landmarks_fix, 
                                     method, set_output_dir, transform_gt)
            
            except Exception as e:
                print(f"Error processing with {method} for set {set_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Compare methods for this set
        compare_methods(results_by_method, set_output_dir)
        
        # Add to global results
        results_by_set[set_name] = results_by_method
    
    # Overall summary
    print("\n==== Processing Summary ====")
    print(f"Sets processed: {len(results_by_set)}")
    print(f"Methods used: {', '.join(methods)}")
    
    # Create summary report
    create_summary_report(results_by_set, methods, output_dir)
    
    return results_by_set