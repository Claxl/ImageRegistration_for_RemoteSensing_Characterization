import os
import argparse
import cv2
import numpy as np
import time
from typing import Tuple, List

from sar_sift import SarSift, Keypoint
from matching import match_descriptors, match, DistanceCriterion


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='SAR-SIFT Image Registration')
    
    parser.add_argument('reference_image', help='Path to reference image')
    parser.add_argument('target_image', help='Path to target image')
    parser.add_argument('transform_model', choices=['similarity', 'affine', 'perspective'],
                        help='Transformation model type')
    parser.add_argument('--output_dir', default='./image_save',
                        help='Directory to save output images (default: ./image_save)')
    parser.add_argument('--feature_density', type=float, default=0.008,
                       help='Feature density per pixel (default: 0.008)')
    
    return parser.parse_args()


def ensure_output_dir(output_dir: str) -> None:
    """Ensure the output directory exists."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def process_images(reference_image_path: str, target_image_path: str, 
                  transform_model: str, output_dir: str,
                  feature_density: float = 0.008) -> None:
    """
    Process images using SAR-SIFT algorithm.
    
    Args:
        reference_image_path: Path to reference image
        target_image_path: Path to target image
        transform_model: Transformation model ('similarity', 'affine', or 'perspective')
        output_dir: Directory to save output images
        feature_density: Feature density per pixel
    """
    # Load images
    image_1 = cv2.imread(reference_image_path, cv2.IMREAD_UNCHANGED)
    image_2 = cv2.imread(target_image_path, cv2.IMREAD_UNCHANGED)
    
    if image_1 is None or image_2 is None:
        raise ValueError("Failed to load images.")
    
    # Calculate feature count based on image size
    num_pixels_1 = image_1.shape[0] * image_1.shape[1]
    num_pixels_2 = image_2.shape[0] * image_2.shape[1]
    nFeatures_1 = int(round(num_pixels_1 * feature_density))
    nFeatures_2 = int(round(num_pixels_2 * feature_density))
    
    # Initialize SAR-SIFT detectors
    sar_sift_1 = SarSift(nFeatures_1, 8, 2, 2**(1.0/3.0), 0.8/5, 0.04)
    sar_sift_2 = SarSift(nFeatures_2, 8, 2, 2**(1.0/3.0), 0.8/5, 0.04)
    
    # Detect keypoints in reference image
    total_start = time.time()
    
    detect1_start = time.time()
    keypoints_1, sar_harris_fun_1, amplit_1, orient_1 = sar_sift_1.detect_keys(image_1)
    detect1_time = time.time() - detect1_start
    print(f"Reference image feature detection time: {detect1_time:.2f}s")
    print(f"Number of features detected in reference image: {len(keypoints_1)}")
    
    # Compute descriptors for reference image
    des1_start = time.time()
    descriptors_1 = sar_sift_1.compute_descriptors(keypoints_1, amplit_1, orient_1)
    des1_time = time.time() - des1_start
    print(f"Reference image descriptor computation time: {des1_time:.2f}s")
    
    # Detect keypoints in target image
    detect2_start = time.time()
    keypoints_2, sar_harris_fun_2, amplit_2, orient_2 = sar_sift_2.detect_keys(image_2)
    detect2_time = time.time() - detect2_start
    print(f"Target image feature detection time: {detect2_time:.2f}s")
    print(f"Number of features detected in target image: {len(keypoints_2)}")
    
    # Compute descriptors for target image
    des2_start = time.time()
    descriptors_2 = sar_sift_2.compute_descriptors(keypoints_2, amplit_2, orient_2)
    des2_time = time.time() - des2_start
    print(f"Target image descriptor computation time: {des2_time:.2f}s")
    
    # Match descriptors
    match_start = time.time()
    dmatchs = match_descriptors(descriptors_1, descriptors_2, DistanceCriterion.COS)
    
    # Find transformation and eliminate outliers
    homography, right_matchs, matched_line = match(
        image_1, image_2, dmatchs, keypoints_1, keypoints_2, transform_model
    )
    match_time = time.time() - match_start
    print(f"Feature matching time: {match_time:.2f}s")
    
    # Print transformation matrix
    print(f"Transformation matrix ({transform_model}):")
    print(homography)
    
    # Save visualization of matches if successful
    if matched_line is not None:
        cv2.imwrite(os.path.join(output_dir, "correct_matches.jpg"), matched_line)
    else:
        print("Warning: Failed to create match visualization")
    
    # Create keypoint visualizations
    keys_image_1 = draw_keypoints(image_1, keypoints_1)
    keys_image_2 = draw_keypoints(image_2, keypoints_2)
    cv2.imwrite(os.path.join(output_dir, "reference_image_keypoints.jpg"), keys_image_1)
    cv2.imwrite(os.path.join(output_dir, "target_image_keypoints.jpg"), keys_image_2)
    
    # Analyze keypoints by layer
    analyze_keypoint_layers(keypoints_1, keypoints_2, right_matchs, output_dir)
    
    # Image fusion
    from matching import image_fusion
    fusion_image, mosaic_image = image_fusion(image_1, image_2, homography)
    cv2.imwrite(os.path.join(output_dir, "fused_image.jpg"), fusion_image)
    cv2.imwrite(os.path.join(output_dir, "mosaic_image.jpg"), mosaic_image)
    
    total_time = time.time() - total_start
    print(f"Total processing time: {total_time:.2f}s")


def draw_keypoints(image: np.ndarray, keypoints: List[Keypoint]) -> np.ndarray:
    """Draw keypoints on image."""
    # Convert custom keypoints to OpenCV keypoints
    cv_keypoints = [
        cv2.KeyPoint(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave)
        for kp in keypoints
    ]
    
    # Draw keypoints
    return cv2.drawKeypoints(image, cv_keypoints, None, color=(0, 255, 0), 
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


def analyze_keypoint_layers(keypoints_1: List[Keypoint], keypoints_2: List[Keypoint], 
                           right_matches: List, output_path: str) -> None:
    """
    Analyze keypoint distribution across scale space layers.
    
    Args:
        keypoints_1: Keypoints from reference image
        keypoints_2: Keypoints from target image
        right_matches: Correct matches
        output_path: Directory to save analysis
    """
    # Check if we have any matches to analyze
    if not right_matches:
        print("No matches to analyze")
        return
    """
    Analyze keypoint distribution across scale space layers.
    
    Args:
        keypoints_1: Keypoints from reference image
        keypoints_2: Keypoints from target image
        right_matches: Correct matches
        output_path: Directory to save analysis
    """
    from sar_sift import SAR_SIFT_LATERS
    
    # Count keypoints per layer
    keys1_num = [0] * SAR_SIFT_LATERS
    keys2_num = [0] * SAR_SIFT_LATERS
    
    for kp in keypoints_1:
        keys1_num[kp.octave] += 1
        
    for kp in keypoints_2:
        keys2_num[kp.octave] += 1
    
    # Count correct matches per layer
    right_nums1 = [0] * SAR_SIFT_LATERS
    right_nums2 = [0] * SAR_SIFT_LATERS
    
    for match in right_matches:
        right_nums1[keypoints_1[match.queryIdx].octave] += 1
        right_nums2[keypoints_2[match.trainIdx].octave] += 1
    
    # Save results to file
    with open(os.path.join(output_path, "position.txt"), "w") as f:
        # Write header
        f.write(f"{'Index':<6} {'Reference coords':<20} {'Layer':<8} {'Strength':<10} "
                f"{'Target coords':<20} {'Layer':<8} {'Strength':<10}\n")
        
        # Write match details
        for i, match in enumerate(right_matches):
            ref_kp = keypoints_1[match.queryIdx]
            tgt_kp = keypoints_2[match.trainIdx]
            
            f.write(f"{i:<6} ({ref_kp.pt[0]:.1f}, {ref_kp.pt[1]:.1f}){' ':<5} "
                    f"{ref_kp.octave:<8} {ref_kp.response:<10.4f} "
                    f"({tgt_kp.pt[0]:.1f}, {tgt_kp.pt[1]:.1f}){' ':<5} "
                    f"{tgt_kp.octave:<8} {tgt_kp.response:<10.4f}\n")
        
        # Write summary by layer
        f.write("\n" + "-" * 70 + "\n")
        f.write(f"{'Layer':<6} {'Ref points':<12} {'Target points':<15} "
                f"{'Ref correct':<12} {'Target correct':<15}\n")
        
        for i in range(SAR_SIFT_LATERS):
            f.write(f"{i:<6} {keys1_num[i]:<12} {keys2_num[i]:<15} "
                    f"{right_nums1[i]:<12} {right_nums2[i]:<15}\n")


if __name__ == "__main__":
    args = parse_arguments()
    ensure_output_dir(args.output_dir)
    
    try:
        process_images(
            args.reference_image,
            args.target_image,
            args.transform_model,
            args.output_dir,
            args.feature_density
        )
        print("Processing completed successfully.")
    except Exception as e:
        print(f"Error: {e}")