#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image Registration Framework with Multiple Methods

This script provides a comprehensive framework for evaluating different image registration methods 
for remote sensing applications, particularly focusing on SAR-to-optical image registration.

Supported methods:
- SIFT (Scale-Invariant Feature Transform)
- SURF (Speeded-Up Robust Features) 
- ORB (Oriented FAST and Rotated BRIEF)
- AKAZE (Accelerated-KAZE)
- RIFT (Rotation Invariant Feature Transform) - if available

The script can operate in three modes:
1. Processing images from separate SAR and optical folders
2. Processing images and ground truth from a single folder
3. Processing using a specific ground truth .mat file

For evaluation, the script computes various metrics:
- Number of detected points in each image
- Number of repeatable points (keypoints common to both images)
- Number of matches corresponding to ground truth landmark pairs
- RMSE between transformed landmarks and ground truth
- Entropy and Mutual Information between registered images
- Transform error metrics when ground truth transformation is available
"""

import os
import re
import argparse
import cv2
import numpy as np
import traceback

from utils import get_image_files, extract_number, save_results
from detectors import create_detector_and_matcher, RIFT_AVAILABLE, SARSIFT_AVAILABLE, LGHD_AVAILABLE
from registration import process_image_pair
from processing import process_from_folder

def main():
    """
    Main function that parses command-line arguments and executes the appropriate processing pipeline.
    
    Supported processing modes:
    1. Process files from a single folder containing SAR, optical images, and ground truth files
    2. Process using a specific ground truth .mat file
    3. Process from separate folders for SAR and optical images
    """
    parser = argparse.ArgumentParser(description="Process SAR and Optical image pairs with ground truth evaluation.")
    parser.add_argument("--sar_folder", type=str, help="Path to the folder containing SAR images.")
    parser.add_argument("--opt_folder", type=str, help="Path to the folder containing Optical images.")
    parser.add_argument("--data_folder", type=str, help="Path to a single folder containing SAR, Optical images and ground truth files.")
    parser.add_argument("--methods", type=str, default="SIFT,SURF,ORB,AKAZE", help="Comma-separated list of methods to use (SIFT, SURF, ORB, AKAZE, RIFT).")
    parser.add_argument("--output_dir", type=str, default="output", help="Path to the output folder.")
    parser.add_argument("--ratio_thresh", type=float, default=0.7, help="Lowe's ratio test threshold.")
    parser.add_argument("--ground_truth", type=str, help="Path to ground truth .mat file.")
    parser.add_argument("--visualize", action="store_true", help="Create visualizations of results.")
    parser.add_argument("--tag", type=str, help="Specific tag/prefix to process (e.g., 'SO', 'CS', etc.).")
    parser.add_argument("--debug", action="store_true", help="Show debug information.")
    
    args = parser.parse_args()
    
    # Parse methods string into list
    methods = args.methods.split(',')
    
    # Check if RIFT is requested but not available
    if "RIFT" in methods and not RIFT_AVAILABLE:
        print("Warning: RIFT method was requested but is not available. It will be skipped.")
        methods = [m for m in methods if m != "RIFT"]
    if "LGHD" in methods and not LGHD_AVAILABLE:
        print("Warning: LGHD method was requested but is not available. It will be skipped.")
        methods = [m for m in methods if m != "LGHD"]

        # Check if SAR-SIFT is requested but not available
    if "SARSIFT" in methods and not SARSIFT_AVAILABLE:
        print("Warning: SAR-SIFT method was requested but is not available. It will be skipped.")
        methods = [m for m in methods if m != "SARSIFT"]
    # Process from a single folder containing both images and ground truth
    if args.data_folder:
        print(f"Processing files from folder: {args.data_folder}")
        process_from_folder(args.data_folder, methods, args.output_dir, 
                           args.tag, args.ratio_thresh, args.visualize)
    '''
    # Process with specific ground truth file if provided
    elif args.ground_truth:
        print(f"Using ground truth data from {args.ground_truth}")
        process_with_ground_truth(args.ground_truth, methods, args.output_dir, 
                                 args.ratio_thresh, args.visualize)
    
    # Process image folders if provided
    elif args.sar_folder and args.opt_folder:
        print(f"Processing images from folders: {args.sar_folder} and {args.opt_folder}")
        
        # Find matching files using modified function
        matching_files = []
        
        # Use find_matching_files_in_folder logic adapted for two folders
        sar_files = [f for f in os.listdir(args.sar_folder) 
                     if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg")]
        opt_files = [f for f in os.listdir(args.opt_folder) 
                     if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg")]
        
        # Extract identifiers using your pattern matching logic
        sar_dict = {}
        opt_dict = {}
        
        for f in sar_files:
            basename = os.path.basename(f)
            # Try to find patterns like TAGNa.png or other identifiers
            match = re.search(r'([A-Z]+\d+)a\.(png|jpg|jpeg)', basename)
            if match:
                key = match.group(1)
                sar_dict[key] = os.path.join(args.sar_folder, f)
                if args.debug:
                    print(f"SAR: {basename} -> key: {key}")
            else:
                # Try using the original extract_number function
                key = extract_number(basename)
                if key is not None:
                    sar_dict[key] = os.path.join(args.sar_folder, f)
                    if args.debug:
                        print(f"SAR: {basename} -> key: {key}")
        
        for f in opt_files:
            basename = os.path.basename(f)
            # Try to find patterns like TAGNb.png
            match = re.search(r'([A-Z]+\d+)b\.(png|jpg|jpeg)', basename)
            if match:
                key = match.group(1)
                opt_dict[key] = os.path.join(args.opt_folder, f)
                if args.debug:
                    print(f"OPT: {basename} -> key: {key}")
            else:
                # Try using the original extract_number function
                key = extract_number(basename)
                if key is not None:
                    opt_dict[key] = os.path.join(args.opt_folder, f)
                    if args.debug:
                        print(f"OPT: {basename} -> key: {key}")
        
        # Find common keys
        common_keys = sorted(set(sar_dict.keys()).intersection(set(opt_dict.keys())))
        
        if args.debug:
            print(f"\nFound {len(common_keys)} common keys: {common_keys}")
            print("\nMatching pairs:")
            for key in common_keys:
                sar_img_path = sar_dict[key]
                opt_img_path = opt_dict[key]
                print(f"  SAR: {os.path.basename(sar_img_path)} <-> Optical: {os.path.basename(opt_img_path)}")
        
        if not common_keys:
            print("No matching image pairs found.")
            return
        
        # Process each matching pair with each method
        for method in methods:
            print(f"\n==== Processing using {method} ====")
            try:
                # Skip RIFT if not available
                if method.upper() == "RIFT" and not RIFT_AVAILABLE:
                    print(f"Skipping RIFT method as it's not available")
                    continue
                    
                # Skip LGHD if not available
                if method.upper() == "LGHD" and not LGHD_AVAILABLE:
                    print(f"Skipping LGHD method as it's not available")
                    continue
                    
                detector, matcher = create_detector_and_matcher(method)
            except Exception as e:
                print(f"Skipping method {method} due to error: {e}")
                continue
            total_NM = 0
            total_NCM = 0
            registration_times = []
            

            for key in common_keys:
                sar_img_path = sar_dict[key]
                opt_img_path = opt_dict[key]
                print(f"Processing pair: SAR: {os.path.basename(sar_img_path)} <-> Optical: {os.path.basename(opt_img_path)}")
                try:
                    print(f"  Using {method} method")
                    # Process the images using the appropriate method
                    NM, NCM, ratio, reg_time, registered_img, matches_img = process_image_pair(
                        sar_img_path, opt_img_path, detector, matcher, args.ratio_thresh,method=method)
                    
                    print(f"  NM: {NM}, NCM: {NCM}, Ratio: {ratio:.2f}, Time: {reg_time:.3f} sec")
                    total_NM += NM
                    total_NCM += NCM
                    registration_times.append(reg_time)
                    
                    # Save results
                    pair_output_dir = os.path.join(args.output_dir, key)
                    if not os.path.exists(pair_output_dir):
                        os.makedirs(pair_output_dir)
                    
                    save_results(sar_img_path, opt_img_path, registered_img, matches_img, method, pair_output_dir)
                    
                except Exception as e:
                    print(f"  Error processing pair: {e}")
                    traceback.print_exc()
            
            overall_ratio = total_NM / total_NCM if total_NCM != 0 else 0
            print(f"Global results for {method}: Total NM: {total_NM}, Total NCM: {total_NCM}, Overall ratio: {overall_ratio:.2f}")
            if registration_times:
                average_time = np.mean(registration_times)
                median_time = np.median(registration_times)
                print(f"Registration times for {method} - Average: {average_time:.3f} sec, Median: {median_time:.3f} sec")
            
            # Write results to file
            results_path = os.path.join(args.output_dir, f"{method}_output.txt")
            with open(results_path, "w", encoding="utf-8") as f:
                f.write(f"Results for {method}\n")
                f.write(f"Found {len(common_keys)} common keys\n")
                f.write(f"Global results : Total NM: {total_NM}, Total NCM: {total_NCM}, Overall ratio: {overall_ratio:.2f}\n")
                if registration_times:
                    f.write(f"Registration times - Average: {average_time:.3f} sec, Median: {median_time:.3f} sec\n")
                print(f"Results written to {results_path}")
    else:
        print("Either --data_folder, --ground_truth, or both --sar_folder and --opt_folder must be provided.")
        parser.print_help()
    '''
if __name__ == "__main__":
    """
    Example usage:
    
    # Process from a single folder with tagged files:
    python main.py --data_folder=dataset/folder --methods=SIFT,SURF,ORB,AKAZE,RIFT --output_dir=results --visualize
    
    # Process specific tag only:
    python main.py --data_folder=dataset/folder --tag=SO --methods=SIFT,RIFT --output_dir=results --visualize
    
    # Process with a specific ground truth file:
    python main.py --ground_truth=dataset/SO1.mat --methods=SIFT,SURF,ORB,AKAZE --output_dir=results --visualize
    
    # Process from separate folders:
    python main.py --sar_folder=dataset/sar --opt_folder=dataset/optical --methods=SIFT,RIFT --output_dir=results
    """
    main()