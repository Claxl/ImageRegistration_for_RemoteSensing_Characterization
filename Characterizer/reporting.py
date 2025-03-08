#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Functions for reporting and summarizing registration results.

This module provides functionality to:
1. Save registration metrics to text files
2. Compare different registration methods
3. Create summary reports across multiple data sets
"""

import os
import logging
import numpy as np
from pathlib import Path
from .visualization import create_method_comparison_chart

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_metrics(results, method, output_dir):
    """
    Save registration metrics to a text file.
    
    Args:
        results (dict): Dictionary containing registration results and metrics
        method (str): Method name used for registration
        output_dir (str): Output directory for saving the metrics file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    metrics_file = output_path / f"{method}_metrics.txt"
    logger.info(f"Saving metrics to {metrics_file}")
    
    try:
        with open(metrics_file, 'w') as f:
            f.write(f"Metrics for {method}\n")
            f.write("=" * (len(f"Metrics for {method}")) + "\n\n")
            
            # Write basic metrics
            _write_basic_metrics(f, results)
            
            # Write transformation matrix if available
            _write_transformation_matrix(f, results)
            
    except Exception as e:
        logger.error(f"Error saving metrics to {metrics_file}: {e}")


def _write_basic_metrics(file, results):
    """Write basic metrics to an open file."""
    file.write(f"Number of keypoints detected in SAR image: {results['num_keypoints_sar']}\n")
    file.write(f"Number of keypoints detected in optical image: {results['num_keypoints_opt']}\n")
    file.write(f"Number of matches between images: {results['num_matches']}\n")
    file.write(f"Number of inlier matches after RANSAC: {results['num_inliers']}\n")
    
    # Handle matrix RMSE - either numpy array or scalar
    if 'matrix_rmse' in results and results['matrix_rmse'] is not None:
        matrix_rmse = results['matrix_rmse']
        if isinstance(matrix_rmse, np.ndarray):
            file.write(f"RMSE between calculated and ground truth matrices: {float(matrix_rmse):.6f}\n")
        else:
            file.write(f"RMSE between calculated and ground truth matrices: {matrix_rmse:.6f}\n")
    else:
        file.write("RMSE between calculated and ground truth matrices: N/A\n")
    
    file.write(f"Total execution time: {results['execution_time']:.4f} sec\n\n")


def _write_transformation_matrix(file, results):
    """Write transformation matrix to an open file if available."""
    if 'transformation_matrix' in results and results['transformation_matrix'] is not None:
        file.write("\nTransformation Matrix:\n")
        for row in results['transformation_matrix']:
            file.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f}\n")


def compare_methods(results_by_method, output_dir):
    """
    Create a comparison table of results across different methods.
    
    Args:
        results_by_method (dict): Dictionary of results for each method
        output_dir (str): Output directory for saving the comparison
    """
    if not results_by_method:
        logger.warning("No results to compare.")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    comparison_file = output_path / "method_comparison.txt"
    logger.info(f"Creating method comparison at {comparison_file}")
    
    try:
        with open(comparison_file, 'w') as f:
            f.write("METHOD COMPARISON\n")
            f.write("===============\n\n")
            
            # Write table header
            _write_comparison_header(f)
            
            # Write data for each method
            for method, results in results_by_method.items():
                _write_method_row(f, method, results)
        
        # Print comparison table to console
        _print_comparison_table(results_by_method)
        
        # Create visual comparison chart
        create_method_comparison_chart(results_by_method, str(output_path))
        
    except Exception as e:
        logger.error(f"Error creating comparison at {comparison_file}: {e}")


def _write_comparison_header(file):
    """Write the comparison table header to a file."""
    header = "{:<10} {:<15} {:<15} {:<15} {:<15} {:<20} {:<15}".format(
        "Method", "SAR Keypoints", "OPT Keypoints", "Matches", "Inliers", "Matrix RMSE", "Time (sec)")
    file.write(header + "\n")
    file.write("-" * 105 + "\n")


def _write_method_row(file, method, results):
    """Write a row of method results to the comparison table."""
    # Format matrix RMSE based on type
    matrix_rmse_str = _format_matrix_rmse(results)
    
    file.write("{:<10} {:<15} {:<15} {:<15} {:<15} {:<20} {:<15.4f}\n".format(
        method, 
        results['num_keypoints_sar'],
        results['num_keypoints_opt'],
        results['num_matches'],
        results['num_inliers'],
        matrix_rmse_str,
        results['execution_time']
    ))


def _format_matrix_rmse(results):
    """Format matrix RMSE value based on its type."""
    if 'matrix_rmse' in results and results['matrix_rmse'] is not None:
        if isinstance(results['matrix_rmse'], np.ndarray):
            return f"{float(results['matrix_rmse']):.6f}"
        else:
            return f"{results['matrix_rmse']:.6f}"
    else:
        return "N/A"


def _print_comparison_table(results_by_method):
    """Print the comparison table to console."""
    logger.info("\n==== Comparison of Methods ====")
    
    # Print header
    header = "{:<10} {:<15} {:<15} {:<15} {:<15} {:<20} {:<15}".format(
        "Method", "SAR Keypoints", "OPT Keypoints", "Matches", "Inliers", "Matrix RMSE", "Time (sec)")
    logger.info(header)
    logger.info("-" * len(header))
    
    # Print each row
    for method, results in results_by_method.items():
        matrix_rmse_str = _format_matrix_rmse(results)
        
        logger.info("{:<10} {:<15} {:<15} {:<15} {:<15} {:<20} {:<15.4f}".format(
            method, 
            results['num_keypoints_sar'],
            results['num_keypoints_opt'],
            results['num_matches'],
            results['num_inliers'],
            matrix_rmse_str,
            results['execution_time']
        ))


def create_summary_report(results_by_set, methods, output_dir):
    """
    Create a summary report of results for all sets and methods.
    
    Args:
        results_by_set (dict): Dictionary with results for each set and method
        methods (list): List of methods used
        output_dir (str): Output directory for the report
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report_path = output_path / "summary_report.txt"
    logger.info(f"Creating summary report at {report_path}")
    
    try:
        with open(report_path, 'w') as f:
            f.write("SUMMARY OF RESULTS\n")
            f.write("=================\n\n")
            
            # Results for each set
            for set_name, set_results in results_by_set.items():
                _write_set_results(f, set_name, set_results)
            
            # Overall methods comparison
            _write_method_comparison(f, results_by_set, methods)
            
        logger.info(f"Summary report saved to: {report_path}")
        
    except Exception as e:
        logger.error(f"Error creating summary report: {e}")


def _write_set_results(file, set_name, set_results):
    """Write results for a single set to the summary report."""
    file.write(f"Set: {set_name}\n")
    file.write("-" * 80 + "\n")
    
    # Metrics table header
    _write_comparison_header(file)
    
    # Data for each method
    for method, results in set_results.items():
        _write_method_row(file, method, results)
    
    file.write("\n\n")


def _write_method_comparison(file, results_by_set, methods):
    """Write method comparison section to the summary report."""
    file.write("METHODS COMPARISON (average across all sets)\n")
    file.write("=========================================\n\n")
    
    # Calculate average metrics for each method
    avg_metrics = _calculate_average_metrics(results_by_set, methods)
    
    # Write header
    _write_comparison_header(file)
    
    # Write average metrics for each method
    for method, metrics in avg_metrics.items():
        _write_avg_method_row(file, method, metrics)


def _calculate_average_metrics(results_by_set, methods):
    """Calculate average metrics for each method across all sets."""
    avg_metrics = {method: _initialize_avg_metrics() for method in methods}
    
    # Counters for averaging
    counts = {method: 0 for method in methods}
    matrix_rmse_counts = {method: 0 for method in methods}
    
    # Sum metrics across all sets
    for set_results in results_by_set.values():
        for method in methods:
            if method in set_results:
                _accumulate_metrics(avg_metrics, counts, matrix_rmse_counts, method, set_results[method])
    
    # Calculate averages
    for method in methods:
        _finalize_averages(avg_metrics, counts, matrix_rmse_counts, method)
    
    return avg_metrics


def _initialize_avg_metrics():
    """Initialize a dictionary for accumulating metrics."""
    return {
        'num_keypoints_sar': 0,
        'num_keypoints_opt': 0,
        'num_matches': 0,
        'num_inliers': 0,
        'matrix_rmse': 0,
        'execution_time': 0
    }


def _accumulate_metrics(avg_metrics, counts, matrix_rmse_counts, method, results):
    """Accumulate metrics for a method from a single result set."""
    counts[method] += 1
    
    # Accumulate standard metrics
    avg_metrics[method]['num_keypoints_sar'] += results['num_keypoints_sar']
    avg_metrics[method]['num_keypoints_opt'] += results['num_keypoints_opt']
    avg_metrics[method]['num_matches'] += results['num_matches']
    avg_metrics[method]['num_inliers'] += results['num_inliers']
    avg_metrics[method]['execution_time'] += results['execution_time']
    
    # Accumulate matrix RMSE if available
    if 'matrix_rmse' in results and results['matrix_rmse'] is not None:
        # Handle numpy array case
        if isinstance(results['matrix_rmse'], np.ndarray):
            avg_metrics[method]['matrix_rmse'] += float(results['matrix_rmse'])
        else:
            avg_metrics[method]['matrix_rmse'] += results['matrix_rmse']
        matrix_rmse_counts[method] += 1


def _finalize_averages(avg_metrics, counts, matrix_rmse_counts, method):
    """Calculate final averages for a method."""
    if counts[method] > 0:
        avg_metrics[method]['num_keypoints_sar'] /= counts[method]
        avg_metrics[method]['num_keypoints_opt'] /= counts[method]
        avg_metrics[method]['num_matches'] /= counts[method]
        avg_metrics[method]['num_inliers'] /= counts[method]
        avg_metrics[method]['execution_time'] /= counts[method]
    
    if matrix_rmse_counts[method] > 0:
        avg_metrics[method]['matrix_rmse'] /= matrix_rmse_counts[method]
    else:
        avg_metrics[method]['matrix_rmse'] = None


def _write_avg_method_row(file, method, metrics):
    """Write a row with average metrics for a method."""
    # Format matrix RMSE
    if metrics['matrix_rmse'] is not None:
        matrix_rmse_str = f"{metrics['matrix_rmse']:.6f}"
    else:
        matrix_rmse_str = "N/A"
    
    file.write("{:<10} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f} {:<20} {:<15.4f}\n".format(
        method,
        metrics['num_keypoints_sar'],
        metrics['num_keypoints_opt'],
        metrics['num_matches'],
        metrics['num_inliers'],
        matrix_rmse_str,
        metrics['execution_time']
    ))