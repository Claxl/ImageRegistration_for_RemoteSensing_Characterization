#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simplified visualization functions showing only matches and registered image.
Now includes mosaic image generation.
"""

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def visualize_results(sar_img, opt_img, results, landmarks_mov=None, landmarks_fix=None, method="", output_dir="output", gt_transform=None):
    """
    Creates simplified visualizations showing only matches and registered image.
    
    Args:
        sar_img (np.ndarray): SAR (moving) image
        opt_img (np.ndarray): Optical (fixed) image
        results (dict): Dictionary containing registration results
        landmarks_mov (list, optional): Landmarks in the moving image
        landmarks_fix (list, optional): Landmarks in the fixed image
        method (str): Method name used for registration
        output_dir (str): Output directory for saving visualizations
        gt_transform (np.ndarray, optional): Ground truth transformation matrix
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the matches image if available
    if 'matches_img' in results and results['matches_img'] is not None:
        # Check if matches_img is already in BGR format
        if len(results['matches_img'].shape) == 3 and results['matches_img'].shape[2] == 3:
            cv2.imwrite(os.path.join(output_dir, f"{method}_matches.png"), results['matches_img'])
        else:
            # Convert to BGR for saving if it's grayscale
            cv2.imwrite(os.path.join(output_dir, f"{method}_matches.png"), 
                      cv2.cvtColor(results['matches_img'], cv2.COLOR_GRAY2BGR))
    
    # Save the registered image if available
    if 'registered_img' in results and results['registered_img'] is not None:
        # Check if registered_img is already in BGR format
        if len(results['registered_img'].shape) == 3 and results['registered_img'].shape[2] == 3:
            cv2.imwrite(os.path.join(output_dir, f"{method}_registered.png"), results['registered_img'])
        else:
            # Convert to BGR for saving if it's grayscale
            cv2.imwrite(os.path.join(output_dir, f"{method}_registered.png"), 
                      cv2.cvtColor(results['registered_img'], cv2.COLOR_GRAY2BGR))
    
    # Generate and save mosaic image if not already present
    if 'mosaic_img' not in results or results['mosaic_img'] is None:
        # Create mosaic image combining optical and registered SAR image
        mosaic_img = create_mosaic_image(opt_img, results['registered_img'] if 'registered_img' in results else sar_img)
        results['mosaic_img'] = mosaic_img
        
        # Save the mosaic image
        if len(mosaic_img.shape) == 3 and mosaic_img.shape[2] == 3:
            cv2.imwrite(os.path.join(output_dir, f"{method}_mosaic.png"), mosaic_img)
        else:
            # Convert to BGR for saving if it's grayscale
            cv2.imwrite(os.path.join(output_dir, f"{method}_mosaic.png"), 
                      cv2.cvtColor(mosaic_img, cv2.COLOR_GRAY2BGR))
    else:
        # Save the existing mosaic image
        if len(results['mosaic_img'].shape) == 3 and results['mosaic_img'].shape[2] == 3:
            cv2.imwrite(os.path.join(output_dir, f"{method}_mosaic.png"), results['mosaic_img'])
        else:
            # Convert to BGR for saving if it's grayscale
            cv2.imwrite(os.path.join(output_dir, f"{method}_mosaic.png"), 
                      cv2.cvtColor(results['mosaic_img'], cv2.COLOR_GRAY2BGR))
    
    # Simple visualization using matplotlib (optional)
    if results['registered_img'] is not None:
        plt.figure(figsize=(18, 6))
        
        # Plot matches if available
        if 'matches_img' in results and results['matches_img'] is not None:
            plt.subplot(1, 3, 1)
            if len(results['matches_img'].shape) == 3:
                plt.imshow(cv2.cvtColor(results['matches_img'], cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(results['matches_img'], cmap='gray')
            plt.title('Feature Matches')
            plt.axis('off')
        
        # Plot registered image
        plt.subplot(1, 3, 2)
        if len(results['registered_img'].shape) == 3:
            plt.imshow(cv2.cvtColor(results['registered_img'], cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(results['registered_img'], cmap='gray')
        plt.title('Registered Image')
        plt.axis('off')
        
        # Plot mosaic image
        plt.subplot(1, 3, 3)
        if len(results['mosaic_img'].shape) == 3:
            plt.imshow(cv2.cvtColor(results['mosaic_img'], cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(results['mosaic_img'], cmap='gray')
        plt.title('Mosaic Image')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{method}_visualization.png"))
        plt.close()

def create_mosaic_image(fixed_img, registered_img):
    """
    Creates a mosaic image combining the fixed (optical) image and the registered (SAR) image.
    
    Args:
        fixed_img (np.ndarray): Fixed image (e.g., optical image)
        registered_img (np.ndarray): Registered image (e.g., registered SAR image)
        
    Returns:
        np.ndarray: Mosaic image combining both inputs
    """
    # Ensure both images have the same number of channels
    if len(fixed_img.shape) != len(registered_img.shape):
        if len(fixed_img.shape) == 2:  # fixed_img is grayscale
            fixed_img = cv2.cvtColor(fixed_img, cv2.COLOR_GRAY2BGR)
        if len(registered_img.shape) == 2:  # registered_img is grayscale
            registered_img = cv2.cvtColor(registered_img, cv2.COLOR_GRAY2BGR)
    
    # Ensure both images have the same dimensions
    h1, w1 = fixed_img.shape[:2]
    h2, w2 = registered_img.shape[:2]
    
    # Resize images to match in height
    if h1 != h2:
        if h1 > h2:
            registered_img = cv2.resize(registered_img, (int(w2 * h1 / h2), h1))
        else:
            fixed_img = cv2.resize(fixed_img, (int(w1 * h2 / h1), h2))
    
    # Create the mosaic by placing the images side by side
    if len(fixed_img.shape) == 3:  # Color images
        mosaic = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        mosaic[:fixed_img.shape[0], :fixed_img.shape[1], :] = fixed_img
        mosaic[:registered_img.shape[0], w1:w1+registered_img.shape[1], :] = registered_img
    else:  # Grayscale images
        mosaic = np.zeros((max(h1, h2), w1 + w2), dtype=np.uint8)
        mosaic[:fixed_img.shape[0], :fixed_img.shape[1]] = fixed_img
        mosaic[:registered_img.shape[0], w1:w1+registered_img.shape[1]] = registered_img
    
    # Add a vertical line to separate the images
    if len(mosaic.shape) == 3:  # Color image
        mosaic[:, w1:w1+1, :] = [255, 255, 255]  # White line
    else:  # Grayscale image
        mosaic[:, w1:w1+1] = 255  # White line
    
    return mosaic

def create_method_comparison_chart(results_by_method, output_dir):
    """
    Creates a visual comparison chart of different registration methods.
    
    Args:
        results_by_method (dict): Dictionary of results for each method
        output_dir (str): Output directory for saving the chart
    """
    if not results_by_method:
        print("No results to compare.")
        return
    
    try:
        methods = list(results_by_method.keys())
        
        # Extract metrics for plotting
        metrics = {
            'Matches': [results['num_matches'] for results in results_by_method.values()],
            'Inliers': [results['num_inliers'] for results in results_by_method.values()],
            'Execution Time (s)': [results['execution_time'] for results in results_by_method.values()]
        }
        
        # Add matrix RMSE if available
        has_matrix_rmse = any('matrix_rmse' in results and results['matrix_rmse'] is not None 
                            for results in results_by_method.values())
        
        if has_matrix_rmse:
            metrics['Matrix RMSE'] = [
                results['matrix_rmse'] if 'matrix_rmse' in results and results['matrix_rmse'] is not None else 0 
                for results in results_by_method.values()
            ]
        
        # Create bar charts for each metric
        plt.figure(figsize=(12, 8))
        
        # Layout depends on number of metrics
        n_metrics = len(metrics)
        n_rows = (n_metrics + 1) // 2  # Ceiling division
        
        for i, (metric_name, values) in enumerate(metrics.items(), 1):
            plt.subplot(n_rows, 2, i)
            bars = plt.bar(methods, values)
            plt.title(metric_name)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                if metric_name == 'Matrix RMSE' or metric_name == 'Execution Time (s)':
                    # Format with more decimal places for RMSE and time
                    label = f'{height:.4f}'
                else:
                    # Integer format for counts
                    label = f'{int(height)}'
                    
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(values) if max(values) > 0 else 0.01,
                        label, ha='center', va='bottom')
        
        plt.suptitle('Comparison of Registration Methods', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
        plt.savefig(os.path.join(output_dir, "method_comparison_chart.png"))
        plt.close()
        
        print(f"Saved visual comparison to {os.path.join(output_dir, 'method_comparison_chart.png')}")
    except Exception as e:
        print(f"Error creating visual comparison: {e}")
        import traceback
        traceback.print_exc()