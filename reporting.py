#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Functions for reporting and summarizing registration results.
"""

import os
from visualization import create_method_comparison_chart

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
        
        f.write(f"Number of detected points: {results['total_points']}\n")
        f.write(f"Number of repeatable points: {results['repeatable_points']}\n")
        f.write(f"Number of ground truth matches: {results['gt_matches']}\n")
        f.write(f"Number of matches (NM): {results['NM']}\n")
        f.write(f"Number of correct matches (NCM): {results['NCM']}\n")
        f.write(f"Ratio NM/NCM: {results['ratio']:.4f}\n")
        f.write(f"Registration time: {results['reg_time']:.4f} sec\n\n")
        
        # Handle RMSE from different sources
        if 'rmse' in results and results['rmse'] is not None:
            f.write(f"RMSE between landmarks: {results['rmse']:.4f}\n")
        elif 'rmse_landmarks' in results and results['rmse_landmarks'] is not None:
            f.write(f"RMSE between landmarks: {results['rmse_landmarks']:.4f}\n")
        else:
            f.write("RMSE between landmarks: N/A\n")
        
        if 'entropy_opt' in results and results['entropy_opt'] is not None:
            f.write(f"Entropy of optical image: {results['entropy_opt']:.4f}\n")
        else:
            f.write("Entropy of optical image: N/A\n")
        
        if 'entropy_reg' in results and results['entropy_reg'] is not None:
            f.write(f"Entropy of registered image: {results['entropy_reg']:.4f}\n")
        else:
            f.write("Entropy of registered image: N/A\n")
        
        if 'mutual_information' in results and results['mutual_information'] is not None:
            f.write(f"Mutual Information: {results['mutual_information']:.4f}\n")
        else:
            f.write("Mutual Information: N/A\n")
        
        # Add transformation error metrics if available
        if 'transform_error' in results and results['transform_error'] is not None:
            f.write(f"\nTransform error on sample points: {results['transform_error']:.4f}\n")
            f.write(f"Transformation matrix RMSE: {results['rmse']:.4f}\n")
            f.write(f"Transformation matrix Frobenius norm: {results['frobenius']:.4f}\n")
        else:
            f.write("\nTransformation matrix comparison: N/A\n")
            
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
    
    # Check if we have transformation error metrics
    has_transform_error = any('transform_error' in results and results['transform_error'] is not None 
                            for results in results_by_method.values())
    
    # Create comparison table
    with open(os.path.join(output_dir, "method_comparison.txt"), 'w') as f:
        f.write("METHOD COMPARISON\n")
        f.write("===============\n\n")
        
        if has_transform_error:
            f.write("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<15}\n".format(
                "Method", "Total Pts", "Repeatable", "GT Matches", "NM", "NCM", "RMSE Landmarks", "MI", "Transform Error"))
            f.write("-" * 105 + "\n")
        else:
            f.write("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}\n".format(
                "Method", "Total Pts", "Repeatable", "GT Matches", "NM", "NCM", "RMSE Landmarks", "MI"))
            f.write("-" * 90 + "\n")
        
        for method, results in results_by_method.items():
            # Handle RMSE from different sources
            if 'rmse' in results and results['rmse'] is not None:
                rmse_value = results['rmse']
            elif 'rmse_landmarks' in results and results['rmse_landmarks'] is not None:
                rmse_value = results['rmse_landmarks']
            else:
                rmse_value = None
                
            rmse_str = f"{rmse_value:.4f}" if rmse_value is not None else "N/A"
            mi_str = f"{results['mutual_information']:.4f}" if 'mutual_information' in results and results['mutual_information'] is not None else "N/A"
            transform_error_str = f"{results['transform_error']:.4f}" if 'transform_error' in results and results['transform_error'] is not None else "N/A"
            
            if has_transform_error:
                f.write("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<15}\n".format(
                    method, 
                    results['total_points'],
                    results['repeatable_points'],
                    results['gt_matches'],
                    results['NM'],
                    results['NCM'],
                    rmse_str,
                    mi_str,
                    transform_error_str
                ))
            else:
                f.write("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}\n".format(
                    method, 
                    results['total_points'],
                    results['repeatable_points'],
                    results['gt_matches'],
                    results['NM'],
                    results['NCM'],
                    rmse_str,
                    mi_str
                ))
    
    # Print comparison table
    print("\n==== Comparison of Methods ====")
    if has_transform_error:
        header = "{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<14} {:<10} {:<15}".format(
            "Method", "Total Pts", "Repeatable", "GT Matches", "NM", "NCM", "RMSE Landmarks", "MI", "Transform Error")
        print(header)
        print("-" * len(header))
    else:
        header = "{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<14} {:<10}".format(
            "Method", "Total Pts", "Repeatable", "GT Matches", "NM", "NCM", "RMSE Landmarks", "MI")
        print(header)
        print("-" * len(header))
    
    for method, results in results_by_method.items():
        # Handle RMSE from different sources
        if 'rmse' in results and results['rmse'] is not None:
            rmse_value = results['rmse']
        elif 'rmse_landmarks' in results and results['rmse_landmarks'] is not None:
            rmse_value = results['rmse_landmarks']
        else:
            rmse_value = None
            
        rmse_str = f"{rmse_value:.4f}" if rmse_value is not None else "N/A"
        mi_str = f"{results['mutual_information']:.4f}" if 'mutual_information' in results and results['mutual_information'] is not None else "N/A"
        transform_error_str = f"{results['transform_error']:.4f}" if 'transform_error' in results and results['transform_error'] is not None else "N/A"
        
        if has_transform_error:
            print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<14} {:<10} {:<15}".format(
                method, 
                results['total_points'],
                results['repeatable_points'],
                results['gt_matches'],
                results['NM'],
                results['NCM'],
                rmse_str,
                mi_str,
                transform_error_str
            ))
        else:
            print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<14} {:<10}".format(
                method, 
                results['total_points'],
                results['repeatable_points'],
                results['gt_matches'],
                results['NM'],
                results['NCM'],
                rmse_str,
                mi_str
            ))
    
    # Create a visual comparison of methods
    create_method_comparison_chart(results_by_method, output_dir)

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
    
    # Check if we have transformation error metrics
    has_transform_error = False
    for set_results in results_by_set.values():
        for method_results in set_results.values():
            if 'transform_error' in method_results and method_results['transform_error'] is not None:
                has_transform_error = True
                break
        if has_transform_error:
            break
    
    # Create the report
    report_path = os.path.join(output_dir, "summary_report.txt")
    with open(report_path, 'w') as f:
        f.write("SUMMARY OF RESULTS\n")
        f.write("=================\n\n")
        
        for set_name, set_results in results_by_set.items():
            f.write(f"Set: {set_name}\n")
            f.write("-" * 80 + "\n")
            
            # Metrics table
            if has_transform_error:
                f.write("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<15} {:<10} {:<15}\n".format(
                    "Method", "Total Pts", "Repeatable", "GT Matches", "NM", "NCM", "RMSE Landmarks", "MI", "Transform Err"))
                f.write("-" * 105 + "\n")
            else:
                f.write("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<15} {:<10}\n".format(
                    "Method", "Total Pts", "Repeatable", "GT Matches", "NM", "NCM", "RMSE Landmarks", "MI"))
                f.write("-" * 90 + "\n")
            
            for method, results in set_results.items():
                # Handle RMSE from different sources (RIFT vs others)
                if 'rmse' in results and results['rmse'] is not None:
                    rmse_value = results['rmse']
                elif 'rmse_landmarks' in results and results['rmse_landmarks'] is not None:
                    rmse_value = results['rmse_landmarks']
                else:
                    rmse_value = None
                    
                rmse_str = f"{rmse_value:.4f}" if rmse_value is not None else "N/A"
                mi_str = f"{results['mutual_information']:.4f}" if 'mutual_information' in results and results['mutual_information'] is not None else "N/A"
                transform_error_str = f"{results['transform_error']:.4f}" if 'transform_error' in results and results['transform_error'] is not None else "N/A"
                
                if has_transform_error:
                    f.write("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<15} {:<10} {:<15}\n".format(
                        method,
                        results['total_points'],
                        results['repeatable_points'],
                        results['gt_matches'],
                        results['NM'],
                        results['NCM'],
                        rmse_str,
                        mi_str,
                        transform_error_str
                    ))
                else:
                    f.write("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<15} {:<10}\n".format(
                        method,
                        results['total_points'],
                        results['repeatable_points'],
                        results['gt_matches'],
                        results['NM'],
                        results['NCM'],
                        rmse_str,
                        mi_str
                    ))
            
            f.write("\n\n")
        
        # Methods comparison section
        f.write("METHODS COMPARISON (average across all sets)\n")
        f.write("=========================================\n\n")
        
        # Calculate average metrics for each method
        avg_metrics = {}
        for method in methods:
            avg_metrics[method] = {
                'total_points': 0,
                'repeatable_points': 0,
                'gt_matches': 0,
                'NM': 0,
                'NCM': 0,
                'rmse': 0,
                'mutual_information': 0,
                'transform_error': 0
            }
            
            # Counters for averaging
            count = 0
            rmse_count = 0
            mi_count = 0
            transform_error_count = 0
            
            for set_results in results_by_set.values():
                if method in set_results:
                    results = set_results[method]
                    count += 1
                    
                    avg_metrics[method]['total_points'] += results['total_points']
                    avg_metrics[method]['repeatable_points'] += results['repeatable_points']
                    avg_metrics[method]['gt_matches'] += results['gt_matches']
                    avg_metrics[method]['NM'] += results['NM']
                    avg_metrics[method]['NCM'] += results['NCM']
                    
                    # Handle RMSE from different sources (RIFT vs others)
                    if 'rmse' in results and results['rmse'] is not None:
                        avg_metrics[method]['rmse'] += results['rmse']
                        rmse_count += 1
                    elif 'rmse_landmarks' in results and results['rmse_landmarks'] is not None:
                        avg_metrics[method]['rmse'] += results['rmse_landmarks']
                        rmse_count += 1
                    
                    if 'mutual_information' in results and results['mutual_information'] is not None:
                        avg_metrics[method]['mutual_information'] += results['mutual_information']
                        mi_count += 1
                    
                    if 'transform_error' in results and results['transform_error'] is not None:
                        avg_metrics[method]['transform_error'] += results['transform_error']
                        transform_error_count += 1
            
            # Calculate averages
            if count > 0:
                avg_metrics[method]['total_points'] /= count
                avg_metrics[method]['repeatable_points'] /= count
                avg_metrics[method]['gt_matches'] /= count
                avg_metrics[method]['NM'] /= count
                avg_metrics[method]['NCM'] /= count
            
            if rmse_count > 0:
                avg_metrics[method]['rmse'] /= rmse_count
            else:
                avg_metrics[method]['rmse'] = None
            
            if mi_count > 0:
                avg_metrics[method]['mutual_information'] /= mi_count
            else:
                avg_metrics[method]['mutual_information'] = None
            
            if transform_error_count > 0:
                avg_metrics[method]['transform_error'] /= transform_error_count
            else:
                avg_metrics[method]['transform_error'] = None
        
        # Write average metrics table
        if has_transform_error:
            f.write("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<15} {:<10} {:<15}\n".format(
                "Method", "Total Pts", "Repeatable", "GT Matches", "NM", "NCM", "RMSE Landmarks", "MI", "Transform Err"))
            f.write("-" * 105 + "\n")
        else:
            f.write("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<15} {:<10}\n".format(
                "Method", "Total Pts", "Repeatable", "GT Matches", "NM", "NCM", "RMSE Landmarks", "MI"))
            f.write("-" * 90 + "\n")
        
        for method, metrics in avg_metrics.items():
            rmse_str = f"{metrics['rmse']:.4f}" if metrics['rmse'] is not None else "N/A"
            mi_str = f"{metrics['mutual_information']:.4f}" if metrics['mutual_information'] is not None else "N/A"
            transform_error_str = f"{metrics['transform_error']:.4f}" if metrics['transform_error'] is not None else "N/A"
            
            if has_transform_error:
                f.write("{:<10} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<15} {:<10} {:<15}\n".format(
                    method,
                    metrics['total_points'],
                    metrics['repeatable_points'],
                    metrics['gt_matches'],
                    metrics['NM'],
                    metrics['NCM'],
                    rmse_str,
                    mi_str,
                    transform_error_str
                ))
            else:
                f.write("{:<10} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<15} {:<10}\n".format(
                    method,
                    metrics['total_points'],
                    metrics['repeatable_points'],
                    metrics['gt_matches'],
                    metrics['NM'],
                    metrics['NCM'],
                    rmse_str,
                    mi_str
                ))
    
    print(f"Summary report saved to: {report_path}")