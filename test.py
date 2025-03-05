import numpy as np
import cv2
# Import the visualization functions
from rift2.visualization import visualize_matches, visualize_transformation, visualize_fusion_results

# Import from your original rift2 package
from rift2.FeatureDetection import FeatureDetection
from rift2.kptsOrientation import kptsOrientation
from rift2.FeatureDescribe import FeatureDescribe

def demo_RIFT2(path1: str, path2: str, transform_type='similarity', visualize=True):
    """
    Demo RIFT2 pipeline with simplified transformation using OpenCV.
    
    Args:
        path1, path2: Paths to the input images
        transform_type: Type of transformation ('similarity', 'affine', 'perspective')
        visualize: Whether to show visualizations
        
    Returns:
        Tuple containing:
        - H: 3x3 transformation matrix
        - fusion_image: Fused image result
        - mosaic_image: Checkerboard mosaic image
        - inliers_kp1: Inlier keypoints from image 1
        - inliers_kp2: Inlier keypoints from image 2
        - rmse: Root mean square error of inliers
    """
    # 1) Read images
    im1 = cv2.imread(path1, cv2.IMREAD_COLOR)
    im2 = cv2.imread(path2, cv2.IMREAD_COLOR)
    if im1 is None or im2 is None:
        print("Error reading input images.")
        return None, None, None, None, None, None

    # If single channel, replicate to 3-ch for consistency
    if im1.ndim == 2:
        im1 = cv2.cvtColor(im1, cv2.COLOR_GRAY2BGR)
    if im2.ndim == 2:
        im2 = cv2.cvtColor(im2, cv2.COLOR_GRAY2BGR)

    print("Feature detection")
    # 2) Feature detection - Keep this part unchanged
    key1, m1, eo1 = FeatureDetection(im1, 4, 6, 5000)
    key2, m2, eo2 = FeatureDetection(im2, 4, 6, 5000)

    print("Orientation")
    # 3) Orientation - Keep this part unchanged
    kpts1 = kptsOrientation(key1, m1, True, 96)
    kpts2 = kptsOrientation(key2, m2, True, 96)

    # 4) Feature description - Keep this part unchanged
    print("Feature description")
    des1 = FeatureDescribe(im1, eo1, kpts1, 96, 6, 6)
    des2 = FeatureDescribe(im2, eo2, kpts2, 96, 6, 6)
    des1 = des1.T  # so it's (numKeypoints1, descriptorDimension)
    des2 = des2.T  # so it's (numKeypoints2, descriptorDimension)

    # 5) Match the descriptors - Keep this part unchanged
    print("Matching features")
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1.astype(np.float32),
                          des2.astype(np.float32),
                          k=2)

    # We can mimic 'matchFeatures' with ratio test etc.
    good = []
    ratio_thresh = 1.0   # or 0.8, etc., as you like
    for m in matches:
        if len(m) == 2:
            if m[0].distance < ratio_thresh * m[1].distance:
                good.append(m[0])
        elif len(m) == 1:
            good.append(m[0])

    if not good:
        print("No good matches found!")
        return None, None, None, None, None, None

    matchedPoints1 = []
    matchedPoints2 = []
    for g in good:
        matchedPoints1.append(kpts1[:2, g.queryIdx])  # (x, y)
        matchedPoints2.append(kpts2[:2, g.trainIdx])

    matchedPoints1 = np.array(matchedPoints1)
    matchedPoints2 = np.array(matchedPoints2)

    # 6) Remove duplicates, etc. - Keep this part unchanged
    matchedPoints2_unique, idxs = np.unique(matchedPoints2, axis=0, return_index=True)
    matchedPoints1_unique = matchedPoints1[idxs]
    
    print(f"Number of unique matches: {len(matchedPoints1_unique)}")
    
    if len(matchedPoints1_unique) < 3:
        print("Not enough unique matches for transformation estimation!")
        return None, None, None, None, None, None

    # 7) MODIFIED: Use OpenCV for transformation estimation instead of FSC
    print(f"Estimating {transform_type} transformation")
    ransac_threshold = 3.0
    
    # OpenCV expects source points, then destination points
    if transform_type == 'similarity':
        # Similarity transform is a subset of affine with preservation of aspect ratio
        H, mask = cv2.estimateAffinePartial2D(
            matchedPoints2_unique, matchedPoints1_unique,  # Note the order: source, then destination
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold
        )
        # Convert to 3x3 form for compatibility
        H_full = np.eye(3)
        if H is not None:
            H_full[0:2, :] = H
    elif transform_type == 'affine':
        H, mask = cv2.estimateAffine2D(
            matchedPoints2_unique, matchedPoints1_unique,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold
        )
        # Convert to 3x3 form
        H_full = np.eye(3)
        if H is not None:
            H_full[0:2, :] = H
    elif transform_type == 'perspective':
        H_full, mask = cv2.findHomography(
            matchedPoints2_unique, matchedPoints1_unique,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold
        )
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")
    
    # If transformation estimation failed, return identity
    if H_full is None:
        print("Transformation estimation failed!")
        H_full = np.eye(3)
        mask = np.zeros(len(matchedPoints1_unique), dtype=np.uint8)
    
    # Extract inlier points
    inlier_idxs = np.where(mask.ravel() == 1)[0]
    c1 = matchedPoints1_unique[inlier_idxs]
    c2 = matchedPoints2_unique[inlier_idxs]
    
    # Calculate RMSE for inliers
    rmse = 0.0
    if len(c1) > 0:
        if transform_type == 'perspective':
            # For homography, we need to apply perspective division
            ones = np.ones((len(c2), 1))
            pts_homogeneous = np.hstack((c2, ones))
            transformed = np.dot(H_full, pts_homogeneous.T).T
            transformed = transformed[:, :2] / transformed[:, 2:]
            errors = transformed - c1
        else:
            # For affine transforms
            transformed = cv2.transform(c2.reshape(-1, 1, 2), H_full[:2]).reshape(-1, 2)
            errors = transformed - c1
        
        rmse = np.sqrt(np.mean(np.sum(errors**2, axis=1)))
    
    print(f"Found {len(c1)} inliers with RMSE: {rmse:.4f}")
    print("Transformation matrix:")
    print(H_full)
    
    # 8) MODIFIED: Simplified image fusion
    def simplified_image_fusion(image1, image2, H):
        """
        Perform simplified image fusion by warping image2 to image1's space.
        """
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
    
    # 9) Apply image fusion
    fusion_image, mosaic_image = simplified_image_fusion(im1, im2, H_full)
    
    # 10) Visualize results if requested
    if visualize:
        # Visualize matches
        visualize_matches(im1, im2, matchedPoints1_unique, matchedPoints2_unique, 
                         mask.ravel().astype(bool), 
                         title=f"{transform_type.upper()} - Matching Points")
        
        # Visualize transformation
        visualize_transformation(im1, im2, H_full, 
                               title=f"{transform_type.upper()} - Transformation (RMSE: {rmse:.4f})")
        
        # Visualize fusion results
        visualize_fusion_results(fusion_image, mosaic_image, 
                               title=f"{transform_type.upper()} - Fusion Results")
        
        # Close all windows when done
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return H_full, fusion_image, mosaic_image, c1, c2, rmse


if __name__ == '__main__':
    # Paths to your images - update these to your actual paths
    path2 = 'DATASET/RemoteSensing/SAR_Optical/SO1a.png'
    path1 = 'DATASET/RemoteSensing/SAR_Optical/SO1b.png'
    
    # Test all transformation types
    for transform_type in ['perspective']:
        print(f"\n=== Testing {transform_type.upper()} transformation ===\n")
        
        H, fusion, mosaic, inliers1, inliers2, rmse = demo_RIFT2(
            path1, path2, 
            transform_type=transform_type,
            visualize=True
        )
        
        if H is not None:
            print(f"Registration completed with RMSE: {rmse:.4f}")
        else:
            print(f"Registration failed for {transform_type} transformation")
        
        # Wait for a key press before continuing to the next transformation type
        print("Press any key to continue to the next transformation type...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()