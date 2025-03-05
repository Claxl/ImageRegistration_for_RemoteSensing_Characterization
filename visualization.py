import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def visualize_matches(img1, img2, pts1, pts2, inliers=None, title="Matching Points", output_dir="output", method="default"):
    """
    Visualize matching points between two images with optional inlier highlighting and save the result.
    
    Args:
        img1, img2: Source images
        pts1, pts2: Matching points as Nx2 arrays
        inliers: Boolean mask or indices of inlier matches
        title: Title to include in the image
        output_dir: Directory to save the visualization image
        method: Method name for file naming
    
    Returns:
        np.ndarray: Visualization image
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Construct save path
    save_path = os.path.join(output_dir, f"{method}_matches.png")
    # Ensure both images are 3-channel for visualization
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    
    # Create side-by-side image
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    height = max(h1, h2)
    width = w1 + w2
    vis = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Place images side by side
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    
    # Create inlier mask if not provided
    if inliers is None:
        inliers = np.ones(len(pts1), dtype=bool)
    elif not isinstance(inliers, np.ndarray):
        inliers = np.array(inliers)
    
    # If inliers is a list of indices, convert to boolean mask
    if inliers.dtype != bool:
        mask = np.zeros(len(pts1), dtype=bool)
        mask[inliers] = True
        inliers = mask
    
    # Draw matches
    for i, (pt1, pt2) in enumerate(zip(pts1, pts2)):
        if i < len(inliers) and inliers[i]:
            # Green lines for inliers
            color = (0, 255, 0)
            thickness = 2
        else:
            # Red lines for outliers
            color = (0, 0, 255)
            thickness = 1
        
        # Convert to integers for OpenCV
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0]) + w1, int(pt2[1])
        
        # Draw line
        cv2.line(vis, (x1, y1), (x2, y2), color, thickness)
        
        # Draw circles at keypoints
        cv2.circle(vis, (x1, y1), 4, color, thickness)
        cv2.circle(vis, (x2, y2), 4, color, thickness)
    
    # Add text information
    inlier_count = np.sum(inliers)
    outlier_count = len(pts1) - inlier_count
    info_text = f"Inliers: {inlier_count}, Outliers: {outlier_count}"
    cv2.putText(vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Save the visualization image
    cv2.imwrite(save_path, vis)
    
    return vis

def visualize_transformation(im1, im2, H, title="Image Transformation", output_dir="output", method="default"):
    """
    Visualize the effect of the transformation matrix H on the second image and save the result.
    
    Args:
        im1: First (reference) image
        im2: Second image to be transformed
        H: 3x3 transformation matrix
        title: Title to include in the image
        output_dir: Directory to save the visualization image
        method: Method name for file naming
    
    Returns:
        np.ndarray: Visualization image
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Construct save path
    save_path = os.path.join(output_dir, f"{method}_transformation.png")
    # Ensure both images are 3-channel for visualization
    if len(im1.shape) == 2:
        im1 = cv2.cvtColor(im1, cv2.COLOR_GRAY2BGR)
    if len(im2.shape) == 2:
        im2 = cv2.cvtColor(im2, cv2.COLOR_GRAY2BGR)
    
    # Get dimensions
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    
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
    warped_im1 = cv2.warpPerspective(im1, offset_matrix, (result_w, result_h))
    warped_im2 = cv2.warpPerspective(im2, offset_matrix @ H, (result_w, result_h))
    
    # Create edge images to better visualize the alignment
    gray1 = cv2.cvtColor(warped_im1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(warped_im2, cv2.COLOR_BGR2GRAY)
    
    edges1 = cv2.Canny(gray1, 50, 150)
    edges2 = cv2.Canny(gray2, 50, 150)
    
    # Create color version of edges
    edges1_color = np.zeros_like(warped_im1)
    edges2_color = np.zeros_like(warped_im2)
    
    edges1_color[edges1 > 0] = [255, 0, 0]  # Blue
    edges2_color[edges2 > 0] = [0, 0, 255]  # Red
    
    # Combine images for visualization
    overlay = cv2.addWeighted(warped_im1, 0.5, warped_im2, 0.5, 0)
    edges_combined = cv2.add(edges1_color, edges2_color)
    
    # Create a 2x2 grid layout for display
    grid_h = max(warped_im1.shape[0], warped_im2.shape[0])
    grid_w = warped_im1.shape[1] + warped_im2.shape[1]
    grid = np.zeros((grid_h * 2, grid_w, 3), dtype=np.uint8)
    
    # Original images (top row)
    grid[:h1, :w1] = im1
    grid[:h2, w1:w1+w2] = im2
    
    # Add titles for the original images
    cv2.putText(grid, "Original Image 1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(grid, "Original Image 2", (w1 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Transformed results (bottom row)
    # Resize overlay and edges to fit in grid layout
    max_w = grid_w // 2
    max_h = grid_h
    
    # Calculate scale to fit largest dimension
    scale_overlay = min(max_w / overlay.shape[1], max_h / overlay.shape[0])
    scale_edges = min(max_w / edges_combined.shape[1], max_h / edges_combined.shape[0])
    
    # Resize images
    overlay_resized = cv2.resize(overlay, None, fx=scale_overlay, fy=scale_overlay)
    edges_resized = cv2.resize(edges_combined, None, fx=scale_edges, fy=scale_edges)
    
    # Calculate positions to place in grid
    h_overlay, w_overlay = overlay_resized.shape[:2]
    h_edges, w_edges = edges_resized.shape[:2]
    
    # Place in bottom row
    grid[grid_h:grid_h+h_overlay, :w_overlay] = overlay_resized
    grid[grid_h:grid_h+h_edges, w_overlay:w_overlay+w_edges] = edges_resized
    
    # Add titles for the transformed images
    cv2.putText(grid, "Overlay of Transformed Images", (10, grid_h + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(grid, "Edge Alignment (Blue: Im1, Red: Im2)", (w_overlay + 10, grid_h + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Main title
    cv2.putText(grid, title, (grid_w//2 - 150, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Save the visualization image
    cv2.imwrite(save_path, grid)
    
    return grid

def create_fusion_images(image1, image2, H):
    """
    Create fusion and checkerboard mosaic images using perspective transform.
    
    Args:
        image1: First (reference) image
        image2: Second image to be transformed
        H: 3x3 transformation matrix
    
    Returns:
        tuple: (fusion_img, checkerboard_mosaic) - Fused image and checkerboard mosaic
    """
    # Ensure both images are 3-channel for better visualization
    if len(image1.shape) == 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    if len(image2.shape) == 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    # Get image dimensions
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    # Create a canvas large enough to hold both images
    output_size = (3*w1, 3*h1)
    
    # Create the offset transformation to center the result
    offset = np.array([[1, 0, w1], [0, 1, h1], [0, 0, 1]], dtype=np.float64)
    
    # Combine transformations
    final_transform = offset @ H
    
    # Warp images onto the canvas
    warped1 = cv2.warpPerspective(image1, offset, output_size)
    warped2 = cv2.warpPerspective(image2, final_transform, output_size)
    
    # Create fusion image
    fusion = np.zeros_like(warped1, dtype=np.float64)
    
    # Blend where both images exist
    mask1 = (warped1 > 0)
    mask2 = (warped2 > 0)
    both = mask1 & mask2
    only1 = mask1 & (~mask2)
    only2 = mask2 & (~mask1)
    
    fusion[both] = (warped1[both].astype(np.float64) + warped2[both].astype(np.float64)) / 2
    fusion[only1] = warped1[only1]
    fusion[only2] = warped2[only2]
    
    fusion = np.clip(fusion, 0, 255).astype(np.uint8)
    
    # Create a simple checkerboard mosaic
    mosaic = np.zeros_like(fusion)
    block_size = 64
    
    # Create checkerboard masks
    y_blocks = fusion.shape[0] // block_size + 1
    x_blocks = fusion.shape[1] // block_size + 1
    
    for y in range(y_blocks):
        for x in range(x_blocks):
            y1 = y * block_size
            y2 = min((y + 1) * block_size, fusion.shape[0])
            x1 = x * block_size
            x2 = min((x + 1) * block_size, fusion.shape[1])
            
            if (y + x) % 2 == 0:
                mosaic[y1:y2, x1:x2] = warped1[y1:y2, x1:x2]
            else:
                mosaic[y1:y2, x1:x2] = warped2[y1:y2, x1:x2]
    
    # Crop the images to remove unnecessary black borders
    # Find non-zero pixels
    non_zero = np.where(fusion > 0)
    if len(non_zero[0]) > 0 and len(non_zero[1]) > 0:
        y_min, y_max = np.min(non_zero[0]), np.max(non_zero[0])
        x_min, x_max = np.min(non_zero[1]), np.max(non_zero[1])
        
        # Add a small border
        border = 10
        y_min = max(0, y_min - border)
        y_max = min(fusion.shape[0] - 1, y_max + border)
        x_min = max(0, x_min - border)
        x_max = min(fusion.shape[1] - 1, x_max + border)
        
        # Crop both images
        fusion = fusion[y_min:y_max+1, x_min:x_max+1]
        mosaic = mosaic[y_min:y_max+1, x_min:x_max+1]
    
    return fusion, mosaic

def create_side_by_side_mosaic(image1, image2):
    """
    Create a side-by-side mosaic of two images.
    
    Args:
        image1: First image
        image2: Second image
    
    Returns:
        np.ndarray: Side-by-side mosaic image
    """
    # Ensure both images are 3-channel for better visualization
    if len(image1.shape) == 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    if len(image2.shape) == 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    
    # Get initial dimensions
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    # Resize images to match in height
    if h1 != h2:
        if h1 > h2:
            image2 = cv2.resize(image2, (int(w2 * h1 / h2), h1))
            # Update dimensions after resize
            h2, w2 = image2.shape[:2]
        else:
            image1 = cv2.resize(image1, (int(w1 * h2 / h1), h2))
            # Update dimensions after resize
            h1, w1 = image1.shape[:2]
    
    # Create the mosaic by placing the images side by side
    mosaic = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    mosaic[:h1, :w1, :] = image1
    mosaic[:h2, w1:w1+w2, :] = image2
    
    # Add a vertical line to separate the images
    mosaic[:, w1:w1+1, :] = [255, 255, 255]  # White line
    
    return mosaic

def visualize_fusion_results(fusion_image, mosaic_image, title="Fusion Results", output_dir="output", method="default"):
    """
    Create a visualization of fusion and mosaic results and save the result.
    
    Args:
        fusion_image: Fused result image
        mosaic_image: Checkerboard mosaic image
        title: Title to include in the image
        output_dir: Directory to save the visualization image
        method: Method name for file naming
    
    Returns:
        np.ndarray: Visualization image
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Construct save path
    save_path = os.path.join(output_dir, f"{method}_fusion_vis.png")
    # Create a side-by-side display
    h1, w1 = fusion_image.shape[:2]
    h2, w2 = mosaic_image.shape[:2]
    
    max_h = max(h1, h2)
    total_w = w1 + w2
    
    display = np.zeros((max_h, total_w, 3), dtype=np.uint8)
    
    # Convert images to BGR if they are grayscale
    if len(fusion_image.shape) == 2:
        fusion_image = cv2.cvtColor(fusion_image, cv2.COLOR_GRAY2BGR)
    if len(mosaic_image.shape) == 2:
        mosaic_image = cv2.cvtColor(mosaic_image, cv2.COLOR_GRAY2BGR)
    
    # Place images side by side
    display[:h1, :w1] = fusion_image
    display[:h2, w1:w1+w2] = mosaic_image
    
    # Add titles
    cv2.putText(display, "Fusion Result", (w1//2 - 80, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display, "Checkerboard Mosaic", (w1 + w2//2 - 120, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Save the visualization image
    cv2.imwrite(save_path, display)
    
    return display

def visualize_results(fixed_img, moving_img, results, method="", output_dir="output"):
    """
    Visualize registration results and save to files.
    
    Args:
        fixed_img: Fixed (reference) image
        moving_img: Moving image to be registered
        results: Dictionary containing registration results
        method: Name of registration method for file naming
        output_dir: Directory for saving results
    
    Returns:
        dict: Dictionary containing visualization images
    """
    vis_results = {}
    
    # Handle output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Prepare base paths for saving
    base_path = os.path.join(output_dir, method)
    
    # Visualize matches if available
    if 'keypoints1' in results and 'keypoints2' in results and 'matches' in results:
        # Extract keypoints and matches
        kp1, kp2, matches = results['keypoints1'], results['keypoints2'], results['matches']
        
        # Convert keypoints to points arrays
        pts1 = np.array([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.array([kp2[m.trainIdx].pt for m in matches])
        
        # Get inliers if available, otherwise use all matches as inliers
        inliers = results.get('inliers', np.ones(len(matches), dtype=bool))
        
        # Visualize matches
        matches_img = visualize_matches(fixed_img, moving_img, pts1, pts2, inliers, 
                                        title=f"{method} - Matches", 
                                        output_dir=output_dir,
                                        method=method)
        vis_results['matches_img'] = matches_img
    
    # Visualize transformation if available
    if 'transformation_matrix' in results and results['transformation_matrix'] is not None:
        # Get transformation matrix
        H = results['transformation_matrix']
        
        # Visualize transformation
        transform_img = visualize_transformation(fixed_img, moving_img, H, 
                                               title=f"{method} - Transformation", 
                                               output_dir=output_dir,
                                               method=method)
        vis_results['transform_img'] = transform_img
        
        # Create fusion images
        fusion_img, checkerboard_img = create_fusion_images(fixed_img, moving_img, H)
        vis_results['fusion_img'] = fusion_img
        vis_results['checkerboard_img'] = checkerboard_img
        
        # Save fusion images directly
        fusion_path = os.path.join(output_dir, f"{method}_fusion.png")
        checkerboard_path = os.path.join(output_dir, f"{method}_checkerboard.png")
        
        # Always save as color images
        cv2.imwrite(fusion_path, fusion_img)
        cv2.imwrite(checkerboard_path, checkerboard_img)
        
        # Create fusion visualization
        fusion_vis = visualize_fusion_results(fusion_img, checkerboard_img, 
                                           title=f"{method} - Fusion Results", 
                                           output_dir=output_dir,
                                           method=method)
        vis_results['fusion_vis'] = fusion_vis
    
    # Create a side-by-side mosaic if registered image is available
    if 'registered_img' in results and results['registered_img'] is not None:
        registered_img = results['registered_img']
        mosaic_img = create_side_by_side_mosaic(fixed_img, registered_img)
        vis_results['mosaic_img'] = mosaic_img
        
        # Save mosaic directly
        mosaic_path = os.path.join(output_dir, f"{method}_mosaic.png")
        cv2.imwrite(mosaic_path, mosaic_img)
    
    return vis_results

def create_method_comparison_chart(results_by_method, output_dir="output"):
    """
    Creates a visual comparison chart of different registration methods and saves it.
    
    Args:
        results_by_method: Dictionary of results for each method
        output_dir: Directory for saving the chart
    
    Returns:
        np.ndarray: Comparison chart image
    """
    if not results_by_method:
        print("No results to compare.")
        return None
    
    try:
        methods = list(results_by_method.keys())
        
        # Extract metrics for plotting
        metrics = {
            'Matches': [results.get('num_matches', 0) for results in results_by_method.values()],
            'Inliers': [results.get('num_inliers', 0) for results in results_by_method.values()],
            'Execution Time (s)': [results.get('execution_time', 0) for results in results_by_method.values()]
        }
        
        # Add matrix RMSE if available
        has_matrix_rmse = any('matrix_rmse' in results and results['matrix_rmse'] is not None 
                            for results in results_by_method.values())
        
        if has_matrix_rmse:
            metrics['Matrix RMSE'] = [
                results.get('matrix_rmse', 0) if 'matrix_rmse' in results and results['matrix_rmse'] is not None else 0 
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
        
        # Save chart
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        chart_path = os.path.join(output_dir, "method_comparison_chart.png")
        plt.savefig(chart_path)
        print(f"Saved visual comparison to {chart_path}")
        plt.close()
            
        # Convert matplotlib figure to OpenCV image for consistent return type
        fig = plt.gcf()
        fig.canvas.draw()
        
        # Use the newer buffer_rgba method or fall back to alternative approach
        try:
            # Modern Matplotlib versions
            chart_img = np.asarray(fig.canvas.buffer_rgba())
            chart_img = cv2.cvtColor(chart_img, cv2.COLOR_RGBA2BGR)
        except AttributeError:
            try:
                # Older Matplotlib versions
                w, h = fig.canvas.get_width_height()
                chart_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
                chart_img = chart_img.reshape((h, w, 3))
                chart_img = cv2.cvtColor(chart_img, cv2.COLOR_RGB2BGR)
            except AttributeError:
                # Fallback method
                print("Using fallback method to convert Matplotlib figure to image")
                chart_path = os.path.join(output_dir, "temp_chart.png")
                plt.savefig(chart_path)
                chart_img = cv2.imread(chart_path)
                if os.path.exists(chart_path):
                    os.remove(chart_path)  # Clean up temporary file
        
        return chart_img
        
    except Exception as e:
        print(f"Error creating visual comparison: {e}")
        import traceback
        traceback.print_exc()
        return None

def debug_registration(fixed_img, moving_img, pts1, pts2, inliers, H, output_dir="output", method="debug", registered_img=None, fusion_img=None, mosaic_img=None):
    """
    Debug the image registration pipeline by saving visualization images.
    
    Args:
        fixed_img, moving_img: Input images
        pts1, pts2: Matched keypoints
        inliers: Indices or boolean mask of inliers
        H: Transformation matrix
        output_dir: Directory for saving debug images
        method: Method name for file naming
        registered_img: Optional registered image
        fusion_img, mosaic_img: Optional fusion results
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Ensure input images are in color
    if len(fixed_img.shape) == 2:
        fixed_img = cv2.cvtColor(fixed_img, cv2.COLOR_GRAY2BGR)
    if len(moving_img.shape) == 2:
        moving_img = cv2.cvtColor(moving_img, cv2.COLOR_GRAY2BGR)
    
    # Visualize matches
    vis_matches = visualize_matches(fixed_img, moving_img, pts1, pts2, inliers, 
                    title="Matches", output_dir=output_dir, method=method)
    
    # Visualize transformation
    vis_transform = visualize_transformation(fixed_img, moving_img, H, 
                        title="Transformation Result", output_dir=output_dir, method=method)
    
    # Create fusion images if not provided
    if fusion_img is None or mosaic_img is None:
        fusion_img, checkerboard_img = create_fusion_images(fixed_img, moving_img, H)
        
        # Save individual fusion images
        fusion_path = os.path.join(output_dir, f"{method}_fusion.png")
        checkerboard_path = os.path.join(output_dir, f"{method}_checkerboard.png")
        
        cv2.imwrite(fusion_path, fusion_img)
        cv2.imwrite(checkerboard_path, checkerboard_img)
    else:
        # If fusion_img is provided but mosaic_img isn't, create a dummy checkerboard
        # (this case shouldn't happen in practice)
        checkerboard_img = mosaic_img if mosaic_img is not None else np.zeros_like(fusion_img)
        
        # Ensure fusion_img and checkerboard_img are in color
        if len(fusion_img.shape) == 2:
            fusion_img = cv2.cvtColor(fusion_img, cv2.COLOR_GRAY2BGR)
        if len(checkerboard_img.shape) == 2:
            checkerboard_img = cv2.cvtColor(checkerboard_img, cv2.COLOR_GRAY2BGR)
    
    # Visualize fusion
    vis_fusion = visualize_fusion_results(fusion_img, checkerboard_img, 
                    title="Fusion Results", output_dir=output_dir, method=method)
    
    # Additional text output
    if isinstance(inliers, np.ndarray):
        num_inliers = np.sum(inliers)
    else:
        num_inliers = len(inliers)
    
    # Save metrics to text file
    with open(os.path.join(output_dir, f"{method}_metrics.txt"), "w") as f:
        f.write(f"Number of matches: {len(pts1)}\n")
        f.write(f"Number of inliers: {num_inliers}\n")
        f.write(f"Inlier ratio: {num_inliers/len(pts1)*100:.2f}%\n")
        f.write("Transformation matrix:\n")
        f.write(f"{H}\n")
    
    # Also print to console
    print(f"Number of matches: {len(pts1)}")
    print(f"Number of inliers: {num_inliers}")
    print(f"Inlier ratio: {num_inliers/len(pts1)*100:.2f}%")
    print("Transformation matrix:")
    print(H)