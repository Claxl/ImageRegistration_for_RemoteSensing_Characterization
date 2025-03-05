import numpy as np
import cv2

def visualize_matches(img1, img2, pts1, pts2, inliers=None, title="Matching Points"):
    """
    Visualize matching points between two images with optional inlier highlighting using OpenCV.
    
    Args:
        img1, img2: Source images
        pts1, pts2: Matching points as Nx2 arrays
        inliers: Boolean mask or indices of inlier matches
        title: Window title
    """
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
    
    # Create a resizable window
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, vis)
    cv2.waitKey(0)
    
    return vis

def visualize_transformation(im1, im2, H, title="Image Transformation"):
    """
    Visualize the effect of the transformation matrix H on the second image using OpenCV.
    
    Args:
        im1: First (reference) image
        im2: Second image to be transformed
        H: 3x3 transformation matrix
        title: Window title
    """
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
    
    # Display the result
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, grid)
    cv2.waitKey(0)
    
    return grid

def visualize_fusion_results(fusion_image, mosaic_image, title="Fusion Results"):
    """
    Display fusion and mosaic results using OpenCV.
    
    Args:
        fusion_image: Fused result image
        mosaic_image: Checkerboard mosaic image
        title: Window title
    """
    # Create a side-by-side display
    h1, w1 = fusion_image.shape[:2]
    h2, w2 = mosaic_image.shape[:2]
    
    max_h = max(h1, h2)
    total_w = w1 + w2
    
    display = np.zeros((max_h, total_w, 3), dtype=np.uint8)
    
    # Place images side by side
    display[:h1, :w1] = fusion_image
    display[:h2, w1:w1+w2] = mosaic_image
    
    # Add titles
    cv2.putText(display, "Fusion Result", (w1//2 - 80, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display, "Checkerboard Mosaic", (w1 + w2//2 - 120, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, display)
    cv2.waitKey(0)
    
    return display

# Example usage:
# This would be called from the main script after getting matches and transformation
def debug_rift_transformation(im1, im2, pts1, pts2, inliers, H, fusion_image=None, mosaic_image=None):
    """
    Debug the RIFT transformation pipeline using visualizations.
    
    Args:
        im1, im2: Input images
        pts1, pts2: Matched keypoints
        inliers: Indices or boolean mask of inliers
        H: Transformation matrix
        fusion_image, mosaic_image: Optional fusion results
    """
    # Visualize matches
    vis_matches = visualize_matches(im1, im2, pts1, pts2, inliers, 
                     title=f"Matches")
    
    # Visualize transformation
    vis_transform = visualize_transformation(im1, im2, H, 
                         title=f"Transformation Result")
    
    # Visualize fusion if available
    if fusion_image is not None and mosaic_image is not None:
        vis_fusion = visualize_fusion_results(fusion_image, mosaic_image, 
                        title="Fusion Results")
    
    # Additional text output
    if isinstance(inliers, np.ndarray):
        num_inliers = np.sum(inliers)
    else:
        num_inliers = len(inliers)
    
    print(f"Number of matches: {len(pts1)}")
    print(f"Number of inliers: {num_inliers}")
    print(f"Inlier ratio: {num_inliers/len(pts1)*100:.2f}%")
    print("Transformation matrix:")
    print(H)
    
    cv2.destroyAllWindows()