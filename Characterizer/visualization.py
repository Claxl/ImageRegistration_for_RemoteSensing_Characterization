#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization functions for image registration results.

This module provides functions to:
1. Visualize matching points between images
2. Visualize image transformations and registrations
3. Create fusion and checkerboard visualizations
4. Generate comparison charts across different methods
"""

import os
import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def visualize_matches(img1, img2, pts1, pts2, inliers=None, title="Matching Points", 
                     output_dir="output", method="default"):
    """
    Visualize matching points between two images with optional inlier highlighting.
    
    Args:
        img1, img2 (np.ndarray): Source images
        pts1, pts2 (np.ndarray): Matching points as Nx2 arrays
        inliers (np.ndarray): Boolean mask or indices of inlier matches
        title (str): Title for the visualization
        output_dir (str): Directory to save the visualization image
        method (str): Method name for file naming
    
    Returns:
        np.ndarray: Visualization image
    """
    # Validate inputs
    if img1 is None or img2 is None:
        logger.error("Invalid images provided for match visualization")
        return None
        
    if pts1 is None or pts2 is None:
        logger.error("Invalid points provided for match visualization")
        return None
        
    if len(pts1) != len(pts2):
        logger.error(f"Points count mismatch: {len(pts1)} vs {len(pts2)}")
        return None
    
    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path / f"{method}_matches.png"
    
    # Prepare images for visualization
    img1_bgr = _ensure_bgr(img1)
    img2_bgr = _ensure_bgr(img2)
    
    # Create side-by-side image
    vis = _create_side_by_side(img1_bgr, img2_bgr)
    
    # Prepare inlier mask
    inlier_mask = _prepare_inlier_mask(inliers, len(pts1))
    
    # Draw matches
    _draw_matches_on_image(vis, pts1, pts2, img1_bgr.shape[1], inlier_mask)
    
    # Add text information
    inlier_count = np.sum(inlier_mask)
    outlier_count = len(pts1) - inlier_count
    info_text = f"Inliers: {inlier_count}, Outliers: {outlier_count}"
    cv2.putText(vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add title
    cv2.putText(vis, title, (vis.shape[1]//2 - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.9, (255, 255, 255), 2)
    
    # Save the visualization image
    try:
        cv2.imwrite(str(save_path), vis)
        logger.info(f"Saved match visualization to {save_path}")
    except Exception as e:
        logger.error(f"Error saving match visualization: {e}")
    
    return vis


def _ensure_bgr(img):
    """Convert grayscale image to BGR if needed."""
    if img is None:
        return None
        
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 3:
        return img.copy()
    elif img.shape[2] == 4:  # RGBA
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        logger.warning(f"Unexpected image format with shape {img.shape}")
        return img


def _create_side_by_side(img1, img2):
    """Create a side-by-side visualization image."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    height = max(h1, h2)
    width = w1 + w2
    vis = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Place images side by side
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    
    return vis


def _prepare_inlier_mask(inliers, num_points):
    """Prepare a boolean mask of inliers."""
    if inliers is None:
        return np.ones(num_points, dtype=bool)
        
    if not isinstance(inliers, np.ndarray):
        inliers = np.array(inliers)
    
    # If inliers is a list of indices, convert to boolean mask
    if inliers.dtype != bool:
        mask = np.zeros(num_points, dtype=bool)
        mask[inliers] = True
        return mask
    
    return inliers


def _draw_matches_on_image(vis, pts1, pts2, width_offset, inlier_mask):
    """Draw match lines and points on the visualization image."""
    for i, (pt1, pt2) in enumerate(zip(pts1, pts2)):
        is_inlier = i < len(inlier_mask) and inlier_mask[i]
        
        # Set color and thickness based on inlier status
        color = (0, 255, 0) if is_inlier else (0, 0, 255)  # Green for inliers, red for outliers
        thickness = 2 if is_inlier else 1
        
        # Convert to integers for OpenCV
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0]) + width_offset, int(pt2[1])
        
        try:
            # Draw line
            cv2.line(vis, (x1, y1), (x2, y2), color, thickness)
            
            # Draw circles at keypoints
            cv2.circle(vis, (x1, y1), 4, color, thickness)
            cv2.circle(vis, (x2, y2), 4, color, thickness)
        except Exception as e:
            logger.warning(f"Error drawing match {i}: {e}")


def visualize_transformation(img1, img2, H, title="Image Transformation", 
                           output_dir="output", method="default"):
    """
    Visualize the effect of the transformation matrix H on the second image.
    
    Args:
        img1 (np.ndarray): First (reference) image
        img2 (np.ndarray): Second image to be transformed
        H (np.ndarray): 3x3 transformation matrix
        title (str): Title for the visualization
        output_dir (str): Directory to save the visualization image
        method (str): Method name for file naming
    
    Returns:
        np.ndarray: Visualization image
    """
    # Validate inputs
    if img1 is None or img2 is None:
        logger.error("Invalid images provided for transformation visualization")
        return None
        
    if H is None:
        logger.error("Invalid transformation matrix provided")
        return None
    
    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path / f"{method}_transformation.png"
    
    # Prepare images for visualization
    img1_bgr = _ensure_bgr(img1)
    img2_bgr = _ensure_bgr(img2)
    
    # Apply transformations to create visualizations
    warped_img1, warped_img2 = _apply_transformations(img1_bgr, img2_bgr, H)
    
    # Create edge images to visualize alignment
    edges_combined = _create_edge_visualization(warped_img1, warped_img2)
    
    # Create overlay of transformed images
    overlay = cv2.addWeighted(warped_img1, 0.5, warped_img2, 0.5, 0)
    
    # Create a 2x2 grid layout for display
    grid = _create_transformation_grid(img1_bgr, img2_bgr, overlay, edges_combined, title)
    
    # Save the visualization image
    try:
        cv2.imwrite(str(save_path), grid)
        logger.info(f"Saved transformation visualization to {save_path}")
    except Exception as e:
        logger.error(f"Error saving transformation visualization: {e}")
    
    return grid


def _apply_transformations(img1, img2, H):
    """Apply transformations to images for visualization."""
    # Get dimensions
    h1, w1 = img1.shape[:2]
    
    # Create larger canvas to show full transformed image
    # Using 3x the size of the reference image as a reasonable estimate
    result_h, result_w = 3*h1, 3*w1
    
    # Create offset transform to center the result
    offset_matrix = np.array([
        [1, 0, w1],
        [0, 1, h1],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Apply transformations
    warped_img1 = cv2.warpPerspective(img1, offset_matrix, (result_w, result_h))
    warped_img2 = cv2.warpPerspective(img2, offset_matrix @ H, (result_w, result_h))
    
    return warped_img1, warped_img2


def _create_edge_visualization(img1, img2):
    """Create a colored edge visualization to show alignment."""
    # Convert to grayscale for edge detection
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Detect edges
    edges1 = cv2.Canny(gray1, 50, 150)
    edges2 = cv2.Canny(gray2, 50, 150)
    
    # Create color version of edges
    edges1_color = np.zeros_like(img1)
    edges2_color = np.zeros_like(img2)
    
    edges1_color[edges1 > 0] = [255, 0, 0]  # Blue
    edges2_color[edges2 > 0] = [0, 0, 255]  # Red
    
    # Combine edge images
    return cv2.add(edges1_color, edges2_color)


def _create_transformation_grid(img1, img2, overlay, edges, title):
    """Create a grid visualization of original and transformed images."""
    # Create a 2x2 grid layout for display
    grid_h = max(img1.shape[0], img2.shape[0])
    grid_w = img1.shape[1] + img2.shape[1]
    grid = np.zeros((grid_h * 2, grid_w, 3), dtype=np.uint8)
    
    # Original images (top row)
    grid[:img1.shape[0], :img1.shape[1]] = img1
    grid[:img2.shape[0], img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2
    
    # Add titles for the original images
    cv2.putText(grid, "Original Image 1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(grid, "Original Image 2", (img1.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Transformed results (bottom row)
    # Resize transformed visualizations to fit in grid
    max_w = grid_w // 2
    max_h = grid_h
    
    # Resize and place overlay
    overlay_resized = _resize_to_fit(overlay, max_w, max_h)
    h_overlay, w_overlay = overlay_resized.shape[:2]
    grid[grid_h:grid_h+h_overlay, :w_overlay] = overlay_resized
    
    # Resize and place edge visualization
    edges_resized = _resize_to_fit(edges, max_w, max_h)
    h_edges, w_edges = edges_resized.shape[:2]
    grid[grid_h:grid_h+h_edges, w_overlay:w_overlay+w_edges] = edges_resized
    
    # Add titles for the transformed images
    cv2.putText(grid, "Overlay of Transformed Images", (10, grid_h + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(grid, "Edge Alignment (Blue: Im1, Red: Im2)", (w_overlay + 10, grid_h + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Main title
    cv2.putText(grid, title, (grid_w//2 - 150, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return grid


def _resize_to_fit(img, max_width, max_height):
    """Resize an image to fit within maximum dimensions while preserving aspect ratio."""
    h, w = img.shape[:2]
    
    # Calculate scale to fit largest dimension
    scale = min(max_width / w, max_height / h)
    
    # Resize image
    if scale < 1:
        return cv2.resize(img, None, fx=scale, fy=scale)
    return img


def create_fusion_images(image1, image2, H):
    """
    Create fusion and checkerboard mosaic images using perspective transform.
    
    Args:
        image1 (np.ndarray): First (reference) image
        image2 (np.ndarray): Second image to be transformed
        H (np.ndarray): 3x3 transformation matrix
    
    Returns:
        tuple: (fusion_img, checkerboard_mosaic) - Fused image and checkerboard mosaic
    """
    # Validate inputs
    if image1 is None or image2 is None or H is None:
        logger.error("Invalid inputs for fusion image creation")
        return None, None
    
    # Ensure both images are 3-channel for better visualization
    image1_bgr = _ensure_bgr(image1)
    image2_bgr = _ensure_bgr(image2)
    
    # Apply transformations
    warped1, warped2 = _apply_transformations(image1_bgr, image2_bgr, H)
    
    # Create fusion image
    fusion = _create_blended_fusion(warped1, warped2)
    
    # Create checkerboard mosaic
    mosaic = _create_checkerboard_mosaic(warped1, warped2)
    
    # Crop images to remove unnecessary black borders
    fusion, mosaic = _crop_to_content(fusion, mosaic)
    
    return fusion, mosaic


def _create_blended_fusion(img1, img2):
    """Create a blended fusion of two images."""
    # Create empty canvas
    fusion = np.zeros_like(img1, dtype=np.float64)
    
    # Create masks for blending
    mask1 = (img1 > 0)
    mask2 = (img2 > 0)
    both = mask1 & mask2
    only1 = mask1 & (~mask2)
    only2 = mask2 & (~mask1)
    
    # Blend images
    fusion[both] = (img1[both].astype(np.float64) + img2[both].astype(np.float64)) / 2
    fusion[only1] = img1[only1]
    fusion[only2] = img2[only2]
    
    return np.clip(fusion, 0, 255).astype(np.uint8)


def _create_checkerboard_mosaic(img1, img2, block_size=64):
    """Create a checkerboard mosaic from two images."""
    mosaic = np.zeros_like(img1)
    
    # Create checkerboard pattern
    y_blocks = img1.shape[0] // block_size + 1
    x_blocks = img1.shape[1] // block_size + 1
    
    for y in range(y_blocks):
        for x in range(x_blocks):
            y1 = y * block_size
            y2 = min((y + 1) * block_size, img1.shape[0])
            x1 = x * block_size
            x2 = min((x + 1) * block_size, img1.shape[1])
            
            if (y + x) % 2 == 0:
                mosaic[y1:y2, x1:x2] = img1[y1:y2, x1:x2]
            else:
                mosaic[y1:y2, x1:x2] = img2[y1:y2, x1:x2]
    
    return mosaic


def _crop_to_content(img1, img2, border=10):
    """Crop images to non-zero content with a small border."""
    # Find non-zero pixels
    non_zero = np.where(img1 > 0)
    if len(non_zero[0]) > 0 and len(non_zero[1]) > 0:
        y_min, y_max = np.min(non_zero[0]), np.max(non_zero[0])
        x_min, x_max = np.min(non_zero[1]), np.max(non_zero[1])
        
        # Add a small border
        y_min = max(0, y_min - border)
        y_max = min(img1.shape[0] - 1, y_max + border)
        x_min = max(0, x_min - border)
        x_max = min(img1.shape[1] - 1, x_max + border)
        
        # Crop both images
        img1_cropped = img1[y_min:y_max+1, x_min:x_max+1]
        img2_cropped = img2[y_min:y_max+1, x_min:x_max+1]
        
        return img1_cropped, img2_cropped
    
    return img1, img2


def create_side_by_side_mosaic(image1, image2):
    """
    Create a side-by-side mosaic of two images.
    
    Args:
        image1 (np.ndarray): First image
        image2 (np.ndarray): Second image
    
    Returns:
        np.ndarray: Side-by-side mosaic image
    """
    # Validate inputs
    if image1 is None or image2 is None:
        logger.error("Invalid inputs for side-by-side mosaic")
        return None
    
    # Ensure both images are 3-channel for better visualization
    image1_bgr = _ensure_bgr(image1)
    image2_bgr = _ensure_bgr(image2)
    
    # Get initial dimensions
    h1, w1 = image1_bgr.shape[:2]
    h2, w2 = image2_bgr.shape[:2]
    
    # Resize images to match in height
    if h1 != h2:
        if h1 > h2:
            image2_bgr = cv2.resize(image2_bgr, (int(w2 * h1 / h2), h1))
            # Update dimensions after resize
            h2, w2 = image2_bgr.shape[:2]
        else:
            image1_bgr = cv2.resize(image1_bgr, (int(w1 * h2 / h1), h2))
            # Update dimensions after resize
            h1, w1 = image1_bgr.shape[:2]
    
    # Create the mosaic by placing the images side by side
    mosaic = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    mosaic[:h1, :w1, :] = image1_bgr
    mosaic[:h2, w1:w1+w2, :] = image2_bgr
    
    # Add a vertical line to separate the images
    mosaic[:, w1:w1+1, :] = [255, 255, 255]  # White line
    
    return mosaic


def visualize_fusion_results(fusion_image, mosaic_image, title="Fusion Results", 
                           output_dir="output", method="default"):
    """
    Create a visualization of fusion and mosaic results.
    
    Args:
        fusion_image (np.ndarray): Fused result image
        mosaic_image (np.ndarray): Checkerboard mosaic image
        title (str): Title for the visualization
        output_dir (str): Directory to save the visualization image
        method (str): Method name for file naming
    
    Returns:
        np.ndarray: Visualization image
    """
    # Validate inputs
    if fusion_image is None or mosaic_image is None:
        logger.error("Invalid inputs for fusion visualization")
        return None
    
    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path / f"{method}_fusion_vis.png"
    
    # Ensure images are in BGR format
    fusion_bgr = _ensure_bgr(fusion_image)
    mosaic_bgr = _ensure_bgr(mosaic_image)
    
    # Create side-by-side display
    h1, w1 = fusion_bgr.shape[:2]
    h2, w2 = mosaic_bgr.shape[:2]
    
    max_h = max(h1, h2)
    total_w = w1 + w2
    
    display = np.zeros((max_h, total_w, 3), dtype=np.uint8)
    
    # Place images side by side
    display[:h1, :w1] = fusion_bgr
    display[:h2, w1:w1+w2] = mosaic_bgr
    
    # Add titles
    cv2.putText(display, "Fusion Result", (w1//2 - 80, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display, "Checkerboard Mosaic", (w1 + w2//2 - 120, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add main title
    cv2.putText(display, title, (total_w//2 - 100, max_h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    # Save the visualization image
    try:
        cv2.imwrite(str(save_path), display)
        logger.info(f"Saved fusion visualization to {save_path}")
    except Exception as e:
        logger.error(f"Error saving fusion visualization: {e}")
    
    return display


def visualize_results(fixed_img, moving_img, results, method="", output_dir="output"):
    """
    Create and save visualizations of registration results.
    
    Args:
        fixed_img (np.ndarray): Fixed (reference) image
        moving_img (np.ndarray): Moving image to be registered
        results (dict): Dictionary containing registration results
        method (str): Name of registration method for file naming
        output_dir (str): Directory for saving results
    
    Returns:
        dict: Dictionary containing visualization images
    """
    if fixed_img is None or moving_img is None or results is None:
        logger.error("Invalid inputs for results visualization")
        return {}
    
    # Prepare output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving visualizations to {output_path}")
    
    vis_results = {}
    
    try:
        # Generate transformation visualization if transformation matrix available
        if _has_transformation_matrix(results):
            vis_results.update(_create_transformation_visualizations(
                fixed_img, moving_img, results, method, output_path))
            
        # Generate registered image visualization if available
        if _has_registered_image(results):
            vis_results.update(_create_registration_visualizations(
                fixed_img, results, method, output_path))
    
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
    
    return vis_results


def _has_transformation_matrix(results):
    """Check if results contain a valid transformation matrix."""
    return ('transformation_matrix' in results and 
            results['transformation_matrix'] is not None)


def _has_registered_image(results):
    """Check if results contain a registered image."""
    return ('registered_img' in results and 
            results['registered_img'] is not None)


def _create_transformation_visualizations(fixed_img, moving_img, results, method, output_path):
    """Create visualizations related to the transformation matrix."""
    vis_results = {}
    
    # Get transformation matrix
    H = results['transformation_matrix']
    
    # Create and save transformation visualization
    transform_img = visualize_transformation(
        fixed_img, moving_img, H, 
        title=f"{method} - Transformation", 
        output_dir=str(output_path),
        method=method
    )
    vis_results['transform_img'] = transform_img
    
    # Create fusion images
    fusion_img, checkerboard_img = create_fusion_images(fixed_img, moving_img, H)
    vis_results['fusion_img'] = fusion_img
    vis_results['checkerboard_img'] = checkerboard_img
    
    # Save individual fusion images
    cv2.imwrite(str(output_path / f"{method}_fusion.png"), fusion_img)
    cv2.imwrite(str(output_path / f"{method}_checkerboard.png"), checkerboard_img)
    
    # Create fusion visualization
    fusion_vis = visualize_fusion_results(
        fusion_img, checkerboard_img, 
        title=f"{method} - Fusion Results", 
        output_dir=str(output_path),
        method=method
    )
    vis_results['fusion_vis'] = fusion_vis
    
    return vis_results


def _create_registration_visualizations(fixed_img, results, method, output_path):
    """Create visualizations related to the registered image."""
    vis_results = {}
    
    registered_img = results['registered_img']
    
    # Create side-by-side mosaic
    mosaic_img = create_side_by_side_mosaic(fixed_img, registered_img)
    vis_results['mosaic_img'] = mosaic_img
    
    # Save mosaic
    cv2.imwrite(str(output_path / f"{method}_mosaic.png"), mosaic_img)
    
    return vis_results


def create_method_comparison_chart(results_by_method, output_dir="output"):
    """
    Create a visual comparison chart of different registration methods.
    
    Args:
        results_by_method (dict): Dictionary of results for each method
        output_dir (str): Directory for saving the chart
    
    Returns:
        np.ndarray: Comparison chart image
    """
    if not results_by_method:
        logger.warning("No results to compare.")
        return None
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    chart_path = output_path / "method_comparison_chart.png"
    
    try:
        # Extract metrics for plotting
        methods = list(results_by_method.keys())
        metrics = _extract_metrics_for_chart(results_by_method)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Layout depends on number of metrics
        n_metrics = len(metrics)
        n_rows = (n_metrics + 1) // 2  # Ceiling division
        
        # Create subplots for each metric
        for i, (metric_name, values) in enumerate(metrics.items(), 1):
            _create_metric_subplot(plt.subplot(n_rows, 2, i), methods, metric_name, values)
        
        # Finalize and save chart
        plt.suptitle('Comparison of Registration Methods', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
        
        plt.savefig(chart_path)
        logger.info(f"Saved visual comparison to {chart_path}")
        plt.close()
        
        # Convert to OpenCV image for consistent return type
        chart_img = cv2.imread(str(chart_path))
        return chart_img
        
    except Exception as e:
        logger.error(f"Error creating visual comparison: {e}")
        import traceback
        traceback.print_exc()
        return None


def _extract_metrics_for_chart(results_by_method):
    """Extract metrics from results for charting."""
    metrics = {
        'Matches': [results.get('num_matches', 0) for results in results_by_method.values()],
        'Inliers': [results.get('num_inliers', 0) for results in results_by_method.values()],
        'Execution Time (s)': [results.get('execution_time', 0) for results in results_by_method.values()]
    }
    
    # Add matrix RMSE if available
    if any('matrix_rmse' in results and results['matrix_rmse'] is not None 
           for results in results_by_method.values()):
        metrics['Matrix RMSE'] = []
        for results in results_by_method.values():
            if 'matrix_rmse' in results and results['matrix_rmse'] is not None:
                # Handle numpy array case
                if isinstance(results['matrix_rmse'], np.ndarray):
                    metrics['Matrix RMSE'].append(float(results['matrix_rmse']))
                else:
                    metrics['Matrix RMSE'].append(results['matrix_rmse'])
            else:
                metrics['Matrix RMSE'].append(0)
    
    return metrics


def _create_metric_subplot(ax, methods, metric_name, values):
    """Create a subplot for a specific metric."""
    # Use numerical x positions for the bars
    x = np.arange(len(methods))
    bars = ax.bar(x, values)
    ax.set_title(metric_name)
    
    # Set tick positions explicitly before setting labels
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        if metric_name == 'Matrix RMSE' or metric_name == 'Execution Time (s)':
            # Format with more decimal places for RMSE and time
            label = f'{height:.4f}'
        else:
            # Integer format for counts
            label = f'{int(height)}'
            
        ax.annotate(label,
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom')