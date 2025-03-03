import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

# Make sure the current directory is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules - using explicit imports to avoid file naming conflicts
from rift import RIFT  # Make sure this is the optimized version
from fsc import FSC
from imagefusion import image_fusion

def main():
    """
    Optimized RIFT demonstration script.
    """
    # Check if data directory exists
    data_dir = "sar-optical"
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist. Please create it and add image pairs.")
        return
    
    # Load image pair
    img1_path = os.path.join(data_dir, "pair1.jpg")
    img2_path = os.path.join(data_dir, "pair2.jpg")
    
    print("Loading images...")
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print(f"Error loading images from {img1_path} or {img2_path}")
        return
    
    # Initialize RIFT
    print("Initializing RIFT...")
    rift = RIFT(scales=4, orientations=6, patch_size=96)
    
    # Process images using optimized pipeline
    print("Processing first image...")
    start_time = time.time()
    keypoints1, descriptors1, mim1 = rift.detect_and_compute(img1)
    img1_time = time.time() - start_time
    print(f"First image processing time: {img1_time:.2f} seconds")
    
    print("Processing second image...")
    start_time = time.time()
    keypoints2, descriptors2, mim2 = rift.detect_and_compute(img2)
    img2_time = time.time() - start_time
    print(f"Second image processing time: {img2_time:.2f} seconds")
    
    print(f"Detected {len(keypoints1)} keypoints in first image")
    print(f"Detected {len(keypoints2)} keypoints in second image")
    
    # For rotation invariance, compute descriptors using rotated MIMs
    print("Computing rotation-invariant descriptors...")
    start_time = time.time()
    
    # Get phase congruency and filter response for second image
    # (needed for creating rotated MIMs)
    pc2, _, _, eo2 = rift.compute_phase_congruency(img2)
    
    # Create rotated MIMs for second image
    mims2 = rift.create_rotated_mims(img2, eo2)
    
    # Compute descriptors for each rotated MIM
    all_descriptors2 = []
    for rotated_mim in mims2:
        rotated_descriptors = rift.compute_descriptor(img2, keypoints2, rotated_mim)
        all_descriptors2.append(rotated_descriptors)
    
    rotation_time = time.time() - start_time
    print(f"Rotation-invariant descriptor computation time: {rotation_time:.2f} seconds")
    
    # Match features using multiple descriptors and keep the best matches
    print("Matching features...")
    start_time = time.time()
    
    best_matches = []
    best_match_count = 0
    
    # Match with original descriptors first
    matches = rift.match_features(descriptors1, descriptors2)
    best_matches = matches
    best_match_count = len(matches)
    
    # Try matching with rotated descriptors to find better matches
    for i, rot_descriptors2 in enumerate(all_descriptors2):
        if rot_descriptors2.size > 0:
            matches = rift.match_features(descriptors1, rot_descriptors2)
            
            if len(matches) > best_match_count:
                best_matches = matches
                best_match_count = len(matches)
                print(f"Found better matches with rotation {i}")
    
    matching_time = time.time() - start_time
    print(f"Feature matching time: {matching_time:.2f} seconds")
    print(f"Found {len(best_matches)} initial matches")
    
    # Extract matched points for outlier removal
    if len(best_matches) < 4:
        print("Not enough matches for transformation estimation")
        return
    
    # Convert keypoints to numpy arrays
    matched_points1 = np.array([keypoints1[m.queryIdx].pt for m in best_matches])
    matched_points2 = np.array([keypoints2[m.trainIdx].pt for m in best_matches])
    
    # Add dummy third column (needed for FSC)
    pts1 = np.hstack([matched_points1, np.zeros((len(matched_points1), 1))])
    pts2 = np.hstack([matched_points2, np.zeros((len(matched_points2), 1))])
    
    print("Removing outliers using FSC...")
    start_time = time.time()
    # Use FSC for outlier removal
    transformation, rmse, inlier_pts1, inlier_pts2 = FSC(pts1, pts2, 'affine', 2)
    fsc_time = time.time() - start_time
    print(f"FSC outlier removal time: {fsc_time:.2f} seconds")
    
    print(f"Found {len(inlier_pts1)} inlier matches after FSC")
    print(f"RMSE: {rmse:.2f}")
    
    print("Visualizing results...")
    # Draw matches
    if len(best_matches) > 0:
        matches_img = cv2.drawMatches(
            img1, keypoints1, img2, keypoints2, best_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        # Draw inlier matches
        if len(inlier_pts1) > 0:
            inlier_kp1 = [cv2.KeyPoint(x, y, size=10) for x, y in inlier_pts1[:, :2]]
            inlier_kp2 = [cv2.KeyPoint(x, y, size=10) for x, y in inlier_pts2[:, :2]]
            
            inlier_matches = [cv2.DMatch(i, i, 0) for i in range(len(inlier_kp1))]
            
            inlier_matches_img = cv2.drawMatches(
                img1, inlier_kp1, img2, inlier_kp2, inlier_matches, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            
            print("Performing image fusion...")
            start_time = time.time()
            # Image fusion
            fusion_image, checkerboard = image_fusion(img2, img1, transformation)
            fusion_time = time.time() - start_time
            print(f"Image fusion time: {fusion_time:.2f} seconds")
            
            # Display results
            plt.figure(figsize=(15, 10))
            
            plt.subplot(221)
            plt.imshow(cv2.cvtColor(matches_img, cv2.COLOR_BGR2RGB))
            plt.title(f'Initial Matches ({len(best_matches)})')
            plt.axis('off')
            
            plt.subplot(222)
            plt.imshow(cv2.cvtColor(inlier_matches_img, cv2.COLOR_BGR2RGB))
            plt.title(f'Inlier Matches ({len(inlier_pts1)})')
            plt.axis('off')
            
            plt.subplot(223)
            plt.imshow(cv2.cvtColor(fusion_image, cv2.COLOR_BGR2RGB))
            plt.title('Fused Image')
            plt.axis('off')
            
            plt.subplot(224)
            plt.imshow(cv2.cvtColor(checkerboard, cv2.COLOR_BGR2RGB))
            plt.title('Checkerboard Visualization')
            plt.axis('off')
            
            plt.tight_layout()
            
            # Save results
            results_dir = "results"
            os.makedirs(results_dir, exist_ok=True)
            
            plt.savefig(os.path.join(results_dir, "rift_results.png"), dpi=300)
            cv2.imwrite(os.path.join(results_dir, "matches.jpg"), matches_img)
            cv2.imwrite(os.path.join(results_dir, "inlier_matches.jpg"), inlier_matches_img)
            cv2.imwrite(os.path.join(results_dir, "fusion.jpg"), fusion_image)
            cv2.imwrite(os.path.join(results_dir, "checkerboard.jpg"), checkerboard)
            
            plt.show()
        else:
            print("No inliers found after FSC.")
    else:
        print("No matches found.")

    # Print performance summary
    total_time = img1_time + img2_time + rotation_time + matching_time + fsc_time
    if 'fusion_time' in locals():
        total_time += fusion_time
    
    print("\nPerformance Summary:")
    print(f"First image processing: {img1_time:.2f}s ({img1_time/total_time*100:.1f}%)")
    print(f"Second image processing: {img2_time:.2f}s ({img2_time/total_time*100:.1f}%)")
    print(f"Rotation-invariant descriptors: {rotation_time:.2f}s ({rotation_time/total_time*100:.1f}%)")
    print(f"Feature matching: {matching_time:.2f}s ({matching_time/total_time*100:.1f}%)")
    print(f"FSC outlier removal: {fsc_time:.2f}s ({fsc_time/total_time*100:.1f}%)")
    if 'fusion_time' in locals():
        print(f"Image fusion: {fusion_time:.2f}s ({fusion_time/total_time*100:.1f}%)")
    print(f"Total processing time: {total_time:.2f}s")
    
    print("Done!")

if __name__ == "__main__":
    main()