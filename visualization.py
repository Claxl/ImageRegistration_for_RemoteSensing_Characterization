#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Functions for visualizing image registration results.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def visualize_results(sar_img, opt_img, results, landmarks_mov, landmarks_fix, method, output_dir, gt_transform=None):
    """
    Creates visualizations of the registration results with landmarks.
    
    Args:
        sar_img (np.ndarray): SAR (moving) image
        opt_img (np.ndarray): Optical (fixed) image
        results (dict): Dictionary containing registration results
        landmarks_mov (np.ndarray): Ground truth landmarks for SAR image
        landmarks_fix (np.ndarray): Ground truth landmarks for optical image
        method (str): Method name used for registration
        output_dir (str): Output directory for saving visualizations
        gt_transform (np.ndarray, optional): Ground truth transformation matrix
    """
    # Convert grayscale images to RGB for visualization
    sar_rgb = cv2.cvtColor(sar_img, cv2.COLOR_GRAY2RGB)
    opt_rgb = cv2.cvtColor(opt_img, cv2.COLOR_GRAY2RGB)
    
    # Draw landmarks on images
    sar_with_landmarks = sar_rgb.copy()
    opt_with_landmarks = opt_rgb.copy()
    
    for i, (x, y) in enumerate(landmarks_mov):
        cv2.circle(sar_with_landmarks, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.putText(sar_with_landmarks, str(i), (int(x) + 5, int(y) - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    for i, (x, y) in enumerate(landmarks_fix):
        cv2.circle(opt_with_landmarks, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.putText(opt_with_landmarks, str(i), (int(x) + 5, int(y) - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # If registration was successful, create more visualizations
    if results['registered_img'] is not None:
        # Ensure registered image is in RGB format
        if len(results['registered_img'].shape) == 2:
            reg_rgb = cv2.cvtColor(results['registered_img'], cv2.COLOR_GRAY2RGB)
        else:
            reg_rgb = results['registered_img'].copy()
            
        reg_with_landmarks = reg_rgb.copy()
        
        # Draw transformed landmarks on registered image
        if results['transformation_matrix'] is not None:
            M = results['transformation_matrix']
            h_landmarks_mov = np.ones((len(landmarks_mov), 3))
            h_landmarks_mov[:, :2] = landmarks_mov
            
            for i, pt in enumerate(h_landmarks_mov):
                tp = np.dot(M, pt)
                x, y = tp[0]/tp[2], tp[1]/tp[2]
                cv2.circle(reg_with_landmarks, (int(x), int(y)), 5, (0, 0, 255), -1)
                cv2.putText(reg_with_landmarks, str(i), (int(x) + 5, int(y) - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # If we have ground truth transform, also show those points in a different color
            if gt_transform is not None:
                for i, pt in enumerate(h_landmarks_mov):
                    tp = np.dot(gt_transform, pt)
                    x, y = tp[0]/tp[2], tp[1]/tp[2]
                    cv2.circle(reg_with_landmarks, (int(x), int(y)), 5, (0, 255, 0), -1)  # Green for ground truth
                    cv2.putText(reg_with_landmarks, f"{i}(GT)", (int(x) + 5, int(y) - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Create comparison image
        comparison = np.zeros((max(opt_img.shape[0], reg_rgb.shape[0]), 
                              opt_img.shape[1] + reg_rgb.shape[1], 3), dtype=np.uint8)
        comparison[:opt_rgb.shape[0], :opt_rgb.shape[1]] = opt_with_landmarks
        comparison[:reg_rgb.shape[0], opt_rgb.shape[1]:] = reg_with_landmarks
        
        # Draw correspondences between landmarks
        if results['transformation_matrix'] is not None:
            M = results['transformation_matrix']
            for i, ((x1, y1), (x2, y2)) in enumerate(zip(landmarks_fix, landmarks_mov)):
                # Transform the SAR landmark using the homography
                pt = np.dot(M, np.array([x2, y2, 1]))
                x2_trans, y2_trans = pt[0]/pt[2], pt[1]/pt[2]
                
                # Draw line connecting original and transformed landmarks
                cv2.line(comparison, (int(x1), int(y1)), 
                        (int(x2_trans) + opt_rgb.shape[1], int(y2_trans)), 
                        (0, 255, 0), 1)
                
                # If we have ground truth, also show those correspondences in a different color
                if gt_transform is not None:
                    pt_gt = np.dot(gt_transform, np.array([x2, y2, 1]))
                    x2_gt, y2_gt = pt_gt[0]/pt_gt[2], pt[1]/pt[2]
                    
                    cv2.line(comparison, (int(x1), int(y1)), 
                            (int(x2_gt) + opt_rgb.shape[1], int(y2_gt)), 
                            (255, 0, 0), 1)  # Blue for ground truth
        
        # Create error visualization
        if gt_transform is not None and results['transformation_matrix'] is not None:
            # Create a grid of points
            h, w = sar_img.shape
            grid_step = 50  # pixels
            grid_points = []
            for y in range(0, h, grid_step):
                for x in range(0, w, grid_step):
                    grid_points.append([x, y, 1])
            
            grid_points = np.array(grid_points)
            
            # Transform grid with both methods
            pred_points = np.dot(M, grid_points.T).T
            pred_points = pred_points[:, :2] / pred_points[:, 2:]
            
            gt_points = np.dot(gt_transform, grid_points.T).T
            gt_points = gt_points[:, :2] / gt_points[:, 2:]
            
            # Calculate distances
            distances = np.sqrt(np.sum((pred_points - gt_points) ** 2, axis=1))
            max_dist = max(distances)
            
            # Create error visualization
            error_vis = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Normalize the grid points back to image coordinates
            for i, ((x, y), dist) in enumerate(zip(grid_points[:, :2], distances)):
                x, y = int(x), int(y)
                if 0 <= x < w and 0 <= y < h:
                    # Color based on error (blue to red)
                    color = [int(255 * (dist / max_dist)), 0, int(255 * (1 - dist / max_dist))]
                    cv2.circle(error_vis, (x, y), 3, color, -1)
            
            # Add text with error statistics
            cv2.putText(error_vis, f"Max Error: {max_dist:.2f} px", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(error_vis, f"Mean Error: {np.mean(distances):.2f} px", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add color scale
            for i in range(100):
                val = i / 100.0
                color = [int(255 * val), 0, int(255 * (1 - val))]
                cv2.line(error_vis, (w - 120, h - 20 - i), (w - 80, h - 20 - i), color, 1)
            
            cv2.putText(error_vis, "0 px", (w - 70, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(error_vis, f"{max_dist:.1f} px", (w - 70, h - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imwrite(os.path.join(output_dir, f"{method}_transform_error.png"), error_vis)
        
        # Save images
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        cv2.imwrite(os.path.join(output_dir, f"{method}_sar_landmarks.png"), cv2.cvtColor(sar_with_landmarks, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, f"{method}_opt_landmarks.png"), cv2.cvtColor(opt_with_landmarks, cv2.COLOR_RGB2BGR))
        
        # Check if registered_img is already in BGR format
        if len(results['registered_img'].shape) == 3 and results['registered_img'].shape[2] == 3:
            cv2.imwrite(os.path.join(output_dir, f"{method}_registered.png"), results['registered_img'])
        else:
            # Convert to BGR for saving if it's grayscale
            cv2.imwrite(os.path.join(output_dir, f"{method}_registered.png"), 
                      cv2.cvtColor(results['registered_img'], cv2.COLOR_GRAY2BGR))
        
        cv2.imwrite(os.path.join(output_dir, f"{method}_reg_landmarks.png"), cv2.cvtColor(reg_with_landmarks, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, f"{method}_comparison.png"), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        
        if results['matches_img'] is not None:
            # Check if matches_img is already in BGR format
            if len(results['matches_img'].shape) == 3 and results['matches_img'].shape[2] == 3:
                cv2.imwrite(os.path.join(output_dir, f"{method}_matches.png"), results['matches_img'])
            else:
                # Convert to BGR for saving if it's grayscale
                cv2.imwrite(os.path.join(output_dir, f"{method}_matches.png"), 
                          cv2.cvtColor(results['matches_img'], cv2.COLOR_RGB2BGR))
        
        # Create matplotlib visualization
        if gt_transform is not None:
            plt.figure(figsize=(15, 12))
            
            plt.subplot(2, 3, 1)
            plt.imshow(sar_with_landmarks)
            plt.title('SAR Image with Landmarks')
            plt.axis('off')
            
            plt.subplot(2, 3, 2)
            plt.imshow(opt_with_landmarks)
            plt.title('Optical Image with Landmarks')
            plt.axis('off')
            
            plt.subplot(2, 3, 3)
            plt.imshow(reg_with_landmarks)
            plt.title('Registered SAR with Landmarks')
            plt.axis('off')
            
            plt.subplot(2, 3, 4)
            if results['matches_img'] is not None:
                plt.imshow(cv2.cvtColor(results['matches_img'], cv2.COLOR_BGR2RGB) 
                          if results['matches_img'].shape[2] == 3 else results['matches_img'])
                plt.title('Feature Matches')
            else:
                plt.imshow(comparison)
                plt.title('Comparison')
            plt.axis('off')
            
            plt.subplot(2, 3, 5)
            plt.imshow(comparison)
            plt.title('Landmarks Comparison (Green: Pred, Blue: GT)')
            plt.axis('off')
            
            plt.subplot(2, 3, 6)
            if os.path.exists(os.path.join(output_dir, f"{method}_transform_error.png")):
                error_vis = cv2.imread(os.path.join(output_dir, f"{method}_transform_error.png"))
                plt.imshow(cv2.cvtColor(error_vis, cv2.COLOR_BGR2RGB))
                plt.title('Transform Error')
            else:
                plt.imshow(np.zeros_like(sar_rgb))
                plt.title('Transform Error: N/A')
            plt.axis('off')
        else:
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 2, 1)
            plt.imshow(sar_with_landmarks)
            plt.title('SAR Image with Landmarks')
            plt.axis('off')
            
            plt.subplot(2, 2, 2)
            plt.imshow(opt_with_landmarks)
            plt.title('Optical Image with Landmarks')
            plt.axis('off')
            
            plt.subplot(2, 2, 3)
            plt.imshow(reg_with_landmarks)
            plt.title('Registered SAR with Transformed Landmarks')
            plt.axis('off')
            
            plt.subplot(2, 2, 4)
            if results['matches_img'] is not None:
                plt.imshow(cv2.cvtColor(results['matches_img'], cv2.COLOR_BGR2RGB) 
                          if results['matches_img'].shape[2] == 3 else results['matches_img'])
                plt.title('Feature Matches')
            else:
                plt.imshow(comparison)
                plt.title('Comparison')
            plt.axis('off')
        
        plt.suptitle(f'Registration Results for {method}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{method}_visualization.png"))
        plt.close()
    else:
        # Save basic images even if registration failed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        cv2.imwrite(os.path.join(output_dir, f"{method}_sar_landmarks.png"), cv2.cvtColor(sar_with_landmarks, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, f"{method}_opt_landmarks.png"), cv2.cvtColor(opt_with_landmarks, cv2.COLOR_RGB2BGR))

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
        
        # Check for transform error metrics
        has_transform_error = any('transform_error' in results and results['transform_error'] is not None 
                                 for results in results_by_method.values())
        
        # Extract metrics for plotting
        metrics = {
            'RMSE Landmarks': [],
            'MI': [],
            'Repeatable Points': [],
        }
        
        if has_transform_error:
            metrics['Transform Error'] = []
        
        # Collect metrics for each method
        for method, results in results_by_method.items():
            # Handle RMSE from different sources
            if 'rmse' in results and results['rmse'] is not None:
                metrics['RMSE Landmarks'].append(results['rmse'])
            elif 'rmse_landmarks' in results and results['rmse_landmarks'] is not None:
                metrics['RMSE Landmarks'].append(results['rmse_landmarks'])
            else:
                metrics['RMSE Landmarks'].append(0)
            
            if 'mutual_information' in results and results['mutual_information'] is not None:
                metrics['MI'].append(results['mutual_information'])
            else:
                metrics['MI'].append(0)
            
            if 'repeatable_points' in results:
                metrics['Repeatable Points'].append(results['repeatable_points'])
            else:
                metrics['Repeatable Points'].append(0)
            
            if has_transform_error:
                if 'transform_error' in results and results['transform_error'] is not None:
                    metrics['Transform Error'].append(results['transform_error'])
                else:
                    metrics['Transform Error'].append(0)
        
        # Create bar charts for each metric
        plt.figure(figsize=(15, 10))
        
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
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(values) if max(values) > 0 else 0.01,
                        f'{height:.2f}', ha='center', va='bottom')
        
        plt.suptitle('Comparison of Registration Methods', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
        plt.savefig(os.path.join(output_dir, "method_comparison_chart.png"))
        plt.close()
        
        print(f"Saved visual comparison to {os.path.join(output_dir, 'method_comparison_chart.png')}")
    except Exception as e:
        print(f"Error creating visual comparison: {e}")
        import traceback
        traceback.print_exc()