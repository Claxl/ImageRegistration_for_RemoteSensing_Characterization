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
- BRISK (Binary Robust Invariant Scalable Keypoints)
- RIFT (Rotation Invariant Feature Transform) - if available
- LGHD (Local Gradient Histogram Descriptor) - if available
- SAR-SIFT - if available

The script processes images and ground truth from a folder with a consistent naming convention.
"""

import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
from Characterizer.detectors import RIFT_AVAILABLE, LGHD_AVAILABLE, SARSIFT_AVAILABLE, MINIMA_AVAILABLE
from Characterizer.processing import process_from_folder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Process SAR and Optical image pairs with ground truth evaluation."
    )
    
    # Input/output arguments
    parser.add_argument(
        "--data_folder", 
        type=str, 
        required=True,
        help="Path to a folder containing SAR, Optical images and ground truth files."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="output",
        help="Path to the output folder."
    )
    
    # Processing control arguments
    parser.add_argument(
        "--methods", 
        type=str, 
        default="SIFT,SURF,ORB,AKAZE",
        help="Comma-separated list of methods to use (SIFT, SURF, ORB, AKAZE, BRISK, RIFT, LGHD, SARSIFT)."
    )
    parser.add_argument(
        "--ratio_thresh", 
        type=float, 
        default=0.7,
        help="Lowe's ratio test threshold."
    )
    parser.add_argument(
        "--tag", 
        type=str, 
        help="Specific tag/prefix to process (e.g., 'SO', 'CS', etc.)."
    )
    
    # Visualization and debug flags
    parser.add_argument(
        "--visualize", 
        action="store_true",
        help="Create visualizations of results."
    )
    parser.add_argument(
        "--model", 
        type=str,
        default="loftr",
        help="Select model for MINIMA"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Show debug information."
    )
    
    return parser.parse_args()


def validate_methods(methods_list):
    """
    Validate and filter methods based on availability.
    
    Args:
        methods_list (list): List of methods to validate
        
    Returns:
        list: Filtered list of available methods
    """
    valid_methods = []
    
    for method in methods_list:
        method = method.strip().upper()
        
        if method == "RIFT" and not RIFT_AVAILABLE:
            logger.warning("RIFT method was requested but is not available. It will be skipped.")
            continue
            
        if method == "LGHD" and not LGHD_AVAILABLE:
            logger.warning("LGHD method was requested but is not available. It will be skipped.")
            continue
            
        if method == "SARSIFT" and not SARSIFT_AVAILABLE:
            logger.warning("SAR-SIFT method was requested but is not available. It will be skipped.")
            continue
        if method == "MINIMA" and not MINIMA_AVAILABLE:
            logger.warning("MINIMA method was requested but is not available. It will be skipped.")
            continue

        valid_methods.append(method)
    
    return valid_methods


def setup_output_directory(base_dir):
    """
    Set up the output directory with timestamp.
    
    Args:
        base_dir (str): Base output directory
        
    Returns:
        Path: Created output directory path
    """
    # Create timestamped output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(base_dir) / f"results_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Results will be saved to: {output_path}")
    return output_path


def main():
    """
    Main function to process image registration based on command-line arguments.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set debug logging level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse and validate methods
    methods = args.methods.split(',')
    methods = validate_methods(methods)
    
    if not methods:
        logger.error("No valid methods selected for processing.")
        return
    
    # Set up output directory
    output_dir = setup_output_directory(args.output_dir)
    
    # Save configuration
    with open(output_dir / "configuration.txt", "w") as f:
        f.write(f"Data folder: {args.data_folder}\n")
        f.write(f"Methods: {', '.join(methods)}\n")
        f.write(f"Ratio threshold: {args.ratio_thresh}\n")
        f.write(f"Tag: {args.tag if args.tag else 'All'}\n")
        f.write(f"Visualize: {args.visualize}\n")
        f.write(f"Debug: {args.debug}\n")
    
    # Process data from folder
    logger.info(f"Processing files from folder: {args.data_folder}")
    logger.info(f"Using methods: {', '.join(methods)}")
    
    process_from_folder(
        args.data_folder, 
        methods, 
        str(output_dir), 
        args.tag, 
        args.ratio_thresh, 
        args.visualize,
        args.model
    )
    
    logger.info(f"Processing complete. Results saved to {output_dir}")


if __name__ == "__main__":
    """
    Example usage:
    
    # Process all methods and all sets:
    python main.py --data_folder=dataset/folder --methods=SIFT,SURF,ORB,AKAZE,RIFT --output_dir=results --visualize
    
    # Process specific tag only:
    python main.py --data_folder=dataset/folder --tag=SO --methods=SIFT,RIFT --output_dir=results --visualize
    """
    try:
        main()
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)