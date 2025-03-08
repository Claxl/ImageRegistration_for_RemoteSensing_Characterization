#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Functions for reporting and summarizing registration results.
"""

import os
from .visualization import create_method_comparison_chart
import numpy as np
def save_metrics(results, method, output_dir):
    """
    Saves the computed metrics to a text file.
    
    Args:
        results (dict): Dictionary containing registration results and metrics
        method (str): Method name used for registration
        output_dir (str): Output directory for saving the metrics file
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(os.path.join(output_dir, f"{method}_metrics.txt"), 'w') as f:
        f.write(f"Metrics for {method}\n")
        f.write("=" * (len(f"Metrics for {method}")) + "\n\n")
        
        # Basic metrics - the ones you specifically wanted
        f.write(f"Number of keypoints detected in SAR image: {results['num_keypoints_sar']}\n")
        f.write(f"Number of keypoints detected in optical image: {results['num_keypoints_opt']}\n")
        f.write(f"Number of matches between images: {results['num_matches']}\n")
        f.write(f"Number of inlier matches after RANSAC: {results['num_inliers']}\n")
        
        if 'matrix_rmse' in results and results['matrix_rmse'] is not None:
            f.write(f"RMSE between calculated and ground truth matrices: {results['matrix_rmse']:.6f}\n")
        else:
            f.write("RMSE between calculated and ground truth matrices: N/A\n")
        
        f.write(f"Total execution time: {results['execution_time']:.4f} sec\n\n")
        
        # Save the transformation matrix if available
        if 'transformation_matrix' in results and results['transformation_matrix'] is not None:
            f.write("\nTransformation Matrix:\n")
            for row in results['transformation_matrix']:
                f.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f}\n")

def compare_methods(results_by_method, output_dir):
    """
    Creates a comparison table of results across different methods.
    
    Args:
        results_by_method (dict): Dictionary of results for each method
        output_dir (str): Output directory for saving the comparison
    """
    if not results_by_method:
        print("No results to compare.")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create comparison table focusing on the metrics you specifically asked for
    with open(os.path.join(output_dir, "method_comparison.txt"), 'w') as f:
        f.write("METHOD COMPARISON\n")
        f.write("===============\n\n")
        
        f.write("{:<10} {:<15} {:<15} {:<15} {:<15} {:<20} {:<15}\n".format(
            "Method", "SAR Keypoints", "OPT Keypoints", "Matches", "Inliers", "Matrix RMSE", "Time (sec)"))
        f.write("-" * 105 + "\n")
        
        for method, results in results_by_method.items():
            # Handle the case where matrix_rmse is a numpy array
            if 'matrix_rmse' in results and results['matrix_rmse'] is not None:
                if isinstance(results['matrix_rmse'], np.ndarray):
                    matrix_rmse_str = f"{float(results['matrix_rmse']):.6f}"
                else:
                    matrix_rmse_str = f"{results['matrix_rmse']:.6f}"
            else:
                matrix_rmse_str = "N/A"
            
            f.write("{:<10} {:<15} {:<15} {:<15} {:<15} {:<20} {:<15.4f}\n".format(
                method, 
                results['num_keypoints_sar'],
                results['num_keypoints_opt'],
                results['num_matches'],
                results['num_inliers'],
                matrix_rmse_str,
                results['execution_time']
            ))
    
    # Print comparison table
    print("\n==== Comparison of Methods ====")
    header = "{:<10} {:<15} {:<15} {:<15} {:<15} {:<20} {:<15}".format(
        "Method", "SAR Keypoints", "OPT Keypoints", "Matches", "Inliers", "Matrix RMSE", "Time (sec)")
    print(header)
    print("-" * len(header))
    
    for method, results in results_by_method.items():
        # Handle the case where matrix_rmse is a numpy array
        if 'matrix_rmse' in results and results['matrix_rmse'] is not None:
            if isinstance(results['matrix_rmse'], np.ndarray):
                matrix_rmse_str = f"{float(results['matrix_rmse']):.6f}"
            else:
                matrix_rmse_str = f"{results['matrix_rmse']:.6f}"
        else:
            matrix_rmse_str = "N/A"
        
        print("{:<10} {:<15} {:<15} {:<15} {:<15} {:<20} {:<15.4f}".format(
            method, 
            results['num_keypoints_sar'],
            results['num_keypoints_opt'],
            results['num_matches'],
            results['num_inliers'],
            matrix_rmse_str,
            results['execution_time']
        ))
    
    # Create a visual comparison of methods
    create_method_comparison_chart(results_by_method, output_dir)

# Also update the create_summary_report function:
def create_summary_report(results_by_set, methods, output_dir):
    """
    Create a summary report of results for all sets and methods.
    
    Args:
        results_by_set (dict): Dictionary with results for each set and method
        methods (list): List of methods used
        output_dir (str): Output directory for the report
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create the report
    report_path = os.path.join(output_dir, "summary_report.txt")
    with open(report_path, 'w') as f:
        f.write("SUMMARY OF RESULTS\n")
        f.write("=================\n\n")
        
        for set_name, set_results in results_by_set.items():
            f.write(f"Set: {set_name}\n")
            f.write("-" * 80 + "\n")
            
            # Metrics table
            f.write("{:<10} {:<15} {:<15} {:<15} {:<15} {:<20} {:<15}\n".format(
                "Method", "SAR Keypoints", "OPT Keypoints", "Matches", "Inliers", "Matrix RMSE", "Time (sec)"))
            f.write("-" * 105 + "\n")
            
            for method, results in set_results.items():
                # Handle the case where matrix_rmse is a numpy array
                if 'matrix_rmse' in results and results['matrix_rmse'] is not None:
                    if isinstance(results['matrix_rmse'], np.ndarray):
                        matrix_rmse_str = f"{float(results['matrix_rmse']):.6f}"
                    else:
                        matrix_rmse_str = f"{results['matrix_rmse']:.6f}"
                else:
                    matrix_rmse_str = "N/A"
                
                f.write("{:<10} {:<15} {:<15} {:<15} {:<15} {:<20} {:<15.4f}\n".format(
                    method, 
                    results['num_keypoints_sar'],
                    results['num_keypoints_opt'],
                    results['num_matches'],
                    results['num_inliers'],
                    matrix_rmse_str,
                    results['execution_time']
                ))
            
            f.write("\n\n")
        
        # Methods comparison section - average across all sets
        f.write("METHODS COMPARISON (average across all sets)\n")
        f.write("=========================================\n\n")
        
        # Calculate average metrics for each method
        avg_metrics = {}
        for method in methods:
            avg_metrics[method] = {
                'num_keypoints_sar': 0,
                'num_keypoints_opt': 0,
                'num_matches': 0,
                'num_inliers': 0,
                'matrix_rmse': 0,
                'execution_time': 0
            }
            
            # Counters for averaging
            count = 0
            matrix_rmse_count = 0
            
            for set_results in results_by_set.values():
                if method in set_results:
                    results = set_results[method]
                    count += 1
                    
                    avg_metrics[method]['num_keypoints_sar'] += results['num_keypoints_sar']
                    avg_metrics[method]['num_keypoints_opt'] += results['num_keypoints_opt']
                    avg_metrics[method]['num_matches'] += results['num_matches']
                    avg_metrics[method]['num_inliers'] += results['num_inliers']
                    avg_metrics[method]['execution_time'] += results['execution_time']
                    
                    if 'matrix_rmse' in results and results['matrix_rmse'] is not None:
                        # Handle numpy array case
                        if isinstance(results['matrix_rmse'], np.ndarray):
                            avg_metrics[method]['matrix_rmse'] += float(results['matrix_rmse'])
                        else:
                            avg_metrics[method]['matrix_rmse'] += results['matrix_rmse']
                        matrix_rmse_count += 1
            
            # Calculate averages
            if count > 0:
                avg_metrics[method]['num_keypoints_sar'] /= count
                avg_metrics[method]['num_keypoints_opt'] /= count
                avg_metrics[method]['num_matches'] /= count
                avg_metrics[method]['num_inliers'] /= count
                avg_metrics[method]['execution_time'] /= count
            
            if matrix_rmse_count > 0:
                avg_metrics[method]['matrix_rmse'] /= matrix_rmse_count
            else:
                avg_metrics[method]['matrix_rmse'] = None
        
        # Write average metrics table
        f.write("{:<10} {:<15} {:<15} {:<15} {:<15} {:<20} {:<15}\n".format(
            "Method", "SAR Keypoints", "OPT Keypoints", "Matches", "Inliers", "Matrix RMSE", "Time (sec)"))
        f.write("-" * 105 + "\n")
        
        for method, metrics in avg_metrics.items():
            if metrics['matrix_rmse'] is not None:
                matrix_rmse_str = f"{metrics['matrix_rmse']:.6f}"
            else:
                matrix_rmse_str = "N/A"
            
            f.write("{:<10} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f} {:<20} {:<15.4f}\n".format(
                method,
                metrics['num_keypoints_sar'],
                metrics['num_keypoints_opt'],
                metrics['num_matches'],
                metrics['num_inliers'],
                matrix_rmse_str,
                metrics['execution_time']
            ))
    
    print(f"Summary report saved to: {report_path}")

def create_summary_report(results_by_set, methods, output_dir):
    """
    Create a summary report of results for all sets and methods.
    
    Args:
        results_by_set (dict): Dictionary with results for each set and method
        methods (list): List of methods used
        output_dir (str): Output directory for the report
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create the report
    report_path = os.path.join(output_dir, "summary_report.txt")
    with open(report_path, 'w') as f:
        f.write("SUMMARY OF RESULTS\n")
        f.write("=================\n\n")
        
        for set_name, set_results in results_by_set.items():
            f.write(f"Set: {set_name}\n")
            f.write("-" * 80 + "\n")
            
            # Metrics table
            f.write("{:<10} {:<15} {:<15} {:<15} {:<15} {:<20} {:<15}\n".format(
                "Method", "SAR Keypoints", "OPT Keypoints", "Matches", "Inliers", "Matrix RMSE", "Time (sec)"))
            f.write("-" * 105 + "\n")
            
            for method, results in set_results.items():
                matrix_rmse_str = f"{results['matrix_rmse']:.6f}" if 'matrix_rmse' in results and results['matrix_rmse'] is not None else "N/A"
                
                f.write("{:<10} {:<15} {:<15} {:<15} {:<15} {:<20} {:<15.4f}\n".format(
                    method, 
                    results['num_keypoints_sar'],
                    results['num_keypoints_opt'],
                    results['num_matches'],
                    results['num_inliers'],
                    matrix_rmse_str,
                    results['execution_time']
                ))
            
            f.write("\n\n")
        
        # Methods comparison section - average across all sets
        f.write("METHODS COMPARISON (average across all sets)\n")
        f.write("=========================================\n\n")
        
        # Calculate average metrics for each method
        avg_metrics = {}
        for method in methods:
            avg_metrics[method] = {
                'num_keypoints_sar': 0,
                'num_keypoints_opt': 0,
                'num_matches': 0,
                'num_inliers': 0,
                'matrix_rmse': 0,
                'execution_time': 0
            }
            
            # Counters for averaging
            count = 0
            matrix_rmse_count = 0
            
            for set_results in results_by_set.values():
                if method in set_results:
                    results = set_results[method]
                    count += 1
                    
                    avg_metrics[method]['num_keypoints_sar'] += results['num_keypoints_sar']
                    avg_metrics[method]['num_keypoints_opt'] += results['num_keypoints_opt']
                    avg_metrics[method]['num_matches'] += results['num_matches']
                    avg_metrics[method]['num_inliers'] += results['num_inliers']
                    avg_metrics[method]['execution_time'] += results['execution_time']
                    
                    if 'matrix_rmse' in results and results['matrix_rmse'] is not None:
                        avg_metrics[method]['matrix_rmse'] += results['matrix_rmse']
                        matrix_rmse_count += 1
            
            # Calculate averages
            if count > 0:
                avg_metrics[method]['num_keypoints_sar'] /= count
                avg_metrics[method]['num_keypoints_opt'] /= count
                avg_metrics[method]['num_matches'] /= count
                avg_metrics[method]['num_inliers'] /= count
                avg_metrics[method]['execution_time'] /= count
            
            if matrix_rmse_count > 0:
                avg_metrics[method]['matrix_rmse'] /= matrix_rmse_count
            else:
                avg_metrics[method]['matrix_rmse'] = None
        
        # Write average metrics table
        f.write("{:<10} {:<15} {:<15} {:<15} {:<15} {:<20} {:<15}\n".format(
            "Method", "SAR Keypoints", "OPT Keypoints", "Matches", "Inliers", "Matrix RMSE", "Time (sec)"))
        f.write("-" * 105 + "\n")
        
        for method, metrics in avg_metrics.items():
            matrix_rmse_str = f"{metrics['matrix_rmse']:.6f}" if metrics['matrix_rmse'] is not None else "N/A"
            
            f.write("{:<10} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f} {:<20} {:<15.4f}\n".format(
                method,
                metrics['num_keypoints_sar'],
                metrics['num_keypoints_opt'],
                metrics['num_matches'],
                metrics['num_inliers'],
                matrix_rmse_str,
                metrics['execution_time']
            ))
    
    print(f"Summary report saved to: {report_path}")