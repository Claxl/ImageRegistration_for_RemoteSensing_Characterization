import cv2
import numpy as np
import time
import os
import glob
import re
from rift2.FeatureDetection import FeatureDetection
from rift2.kptsOrientation import kptsOrientation
from rift2.FeatureDescribe import FeatureDescribe
from rift2.FSC import FSC
from rift2.image_fusion import image_fusion

def get_image_files(folder, extensions=['*.jpg', '*.png', '*.jpeg']):
    """Returns a sorted list of image file paths from the given folder."""
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)

def extract_number(filename):
    """
    Extracts the first group of digits from the filename.
    Returns the number as a string, or None if no digits are found.
    """
    match = re.search(r'\d+', filename)
    if match:
        return match.group()
    return None


def save_results(sar_path, opt_path, registered_img, matches_img):
    """
    Saves the registered SAR image and the image with drawn matches into an output folder.
    The filenames include the method name and the base filenames of the input images.
    """
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    base_sar = os.path.splitext(os.path.basename(sar_path))[0]
    base_opt = os.path.splitext(os.path.basename(opt_path))[0]
    
    reg_filename = os.path.join(output_dir, f"RIFT2_registered_{base_sar}_to_{base_opt}.png")
    matches_filename = os.path.join(output_dir, f"RIFT2_matches_{base_sar}_to_{base_opt}.png")
    
    if registered_img is not None:
        cv2.imwrite(reg_filename, registered_img)
    if matches_img is not None:
        cv2.imwrite(matches_filename, matches_img)
def make_match_image(im1, im2, pts1, pts2, color=(0, 255, 255), radius=5, thickness=2):
    """
    Creates and returns a single image visualizing matches between two images.
    
    Args:
        im1 (np.ndarray): First image (BGR or grayscale).
        im2 (np.ndarray): Second image (BGR or grayscale).
        pts1 (np.ndarray): Nx2 array of matched keypoints in image1 (x, y).
        pts2 (np.ndarray): Nx2 array of matched keypoints in image2 (x, y).
        color (tuple): BGR color for circles and lines.
        radius (int): Radius of the circle representing each keypoint match.
        thickness (int): Thickness of both the circle border and the line.
    
    Returns:
        match_vis (np.ndarray): Composite image with matches drawn.
    """

    # Ensure both images are color so we can draw in color
    if len(im1.shape) == 2:
        im1 = cv2.cvtColor(im1, cv2.COLOR_GRAY2BGR)
    if len(im2.shape) == 2:
        im2 = cv2.cvtColor(im2, cv2.COLOR_GRAY2BGR)

    # Dimensions
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]

    # Create a blank canvas for side-by-side display
    # Height is max of both images, width is sum of widths
    match_vis = np.zeros((max(h1,h2), w1 + w2, 3), dtype=np.uint8)

    # Place im1 (left) and im2 (right) on the canvas
    match_vis[:h1, :w1] = im1
    match_vis[:h2, w1:w1 + w2] = im2

    # Draw each match
    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        # Adjust x2 because the second image is shifted to the right by w1
        x2_shifted = x2 + w1

        # Draw circles at the matched points
        cv2.circle(match_vis, (int(x1), int(y1)), radius, color, thickness)
        cv2.circle(match_vis, (int(x2_shifted), int(y2)), radius, color, thickness)

        # Draw a line between them
        cv2.line(match_vis, (int(x1), int(y1)), (int(x2_shifted), int(y2)), color, thickness)

    return match_vis
def process_image_pair(sar_img_path, opt_img_path):
    """
    Processes a pair of images:
      - Reads the images (in grayscale)
      - Extracts keypoints and descriptors using the provided detector
      - Matches descriptors using the provided matcher with k=2 and applies Lowe’s ratio test
      - Computes the homography using RANSAC (if there are enough matches)
      - Draws the inlier matches and registers the SAR image using the homography.
    Returns:
      NM: number of matches after ratio test,
      NCM: number of correct matches (inliers),
      ratio: NM/NCM (or 0 if NCM==0),
      reg_time: processing time,
      registered_img: the warped (registered) SAR image,
      matches_img: the drawn matches image.
    """
    start_reg_time = time.time()
    
    # Load images in grayscale
    sar_img = cv2.imread(sar_img_path, cv2.IMREAD_GRAYSCALE)
    opt_img = cv2.imread(opt_img_path, cv2.IMREAD_GRAYSCALE)
    if sar_img is None:
        raise IOError(f"Error loading images: {sar_img_path}")
    if opt_img is None:
        raise IOError(f"Error loading images: {opt_img_path}")

    # If single channel, replicate to 3-ch for consistency
    if sar_img.ndim == 2:
        sar_img = cv2.cvtColor(sar_img, cv2.COLOR_GRAY2BGR)
    if opt_img.ndim == 2:
        opt_img = cv2.cvtColor(opt_img, cv2.COLOR_GRAY2BGR)

    print("Feature detection")
    # 2) Feature detection
    key1, m1, eo1 = FeatureDetection(sar_img, 4, 6, 5000)
    key2, m2, eo2 = FeatureDetection(opt_img, 4, 6, 5000)


    print("Orientation")

    # 3) Orientation
    kpts1 = kptsOrientation(key1, m1, True, 96)
    kpts2 = kptsOrientation(key2, m2, True, 96)

    # 4) Feature description
    print("Feature description")
    des1 = FeatureDescribe(sar_img, eo1, kpts1, 96, 6, 6)  # not shown here
    des2 = FeatureDescribe(opt_img, eo2, kpts2, 96, 6, 6)  # not shown here
    des1 = des1.T  # so it's (numKeypoints1, descriptorDimension)
    des2 = des2.T  # so it's (numKeypoints2, descriptorDimension)


    # 5) Match the descriptors
    #   Suppose they’re arrays shape (N1, D) and (N2, D).
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1.astype(np.float32),
                          des2.astype(np.float32),
                          k=2)

    # We can mimic 'matchFeatures' with ratio test etc.
    good = []
    ratio_thresh = 1   # I take it all matches
    for m in matches:
        if len(m) == 2:
            if m[0].distance < ratio_thresh * m[1].distance:
                good.append(m[0])
        elif len(m) == 1:
            good.append(m[0])

    matchedPoints1 = []
    matchedPoints2 = []
    for g in good:
        matchedPoints1.append(kpts1[:2, g.queryIdx])  # (x, y)
        matchedPoints2.append(kpts2[:2, g.trainIdx])

    matchedPoints1 = np.array(matchedPoints1)
    matchedPoints2 = np.array(matchedPoints2)

    # 6) Remove duplicates, etc.
    #   For example:
    matchedPoints2_unique, idxs = np.unique(matchedPoints2, axis=0, return_index=True)
    matchedPoints1_unique = matchedPoints1[idxs]

    # 7) RANSAC or FSC
    H, rmse, c1, c2 = FSC(matchedPoints1_unique, matchedPoints2_unique,
                          change_form='similarity',
                          error_t=3.0)

    registered_img = image_fusion(sar_img, opt_img, H)
    NM = matchedPoints2_unique.shape[0]
    NCM = c2.shape[0]
    ratio = NM/ NCM if NCM != 0 else 0
    reg_time = time.time() - start_reg_time
    matches_img = make_match_image(sar_img, opt_img, c1, c2)
    return NM, NCM, ratio, reg_time, registered_img, matches_img, rmse

def main():
    # Ask for input folders for SAR and Optical images
    sar_folder = "DATASET/OSdataset/512/test/sar"
    opt_folder = "DATASET/OSdataset/512/test/opt"
    
    sar_files = get_image_files(sar_folder)
    opt_files = get_image_files(opt_folder)
    
    if len(sar_files) == 0 or len(opt_files) == 0:
        print("No images found in one of the folders.")
        return
    
    # Build dictionaries mapping the extracted number from filename to file path
    sar_dict = {}
    for f in sar_files:
        key = extract_number(os.path.basename(f))
        if key is not None:
            sar_dict[key] = f

    opt_dict = {}
    for f in opt_files:
        key = extract_number(os.path.basename(f))
        if key is not None:
            opt_dict[key] = f

    # Find common keys (images with the same number)
    common_keys = sorted(set(sar_dict.keys()).intersection(set(opt_dict.keys())), key=lambda x: int(x))
    if not common_keys:
        print("No matching image pairs (by number) found.")
        return
    total_NM = 0
    total_NCM = 0
    registration_times = []
        
    for key in common_keys:
            sar_img_path = sar_dict[key]
            opt_img_path = opt_dict[key]
            print(f"Processing pair: SAR: {os.path.basename(sar_img_path)} <-> Optical: {os.path.basename(opt_img_path)}")
            try:
                NM, NCM, ratio, reg_time, registered_img, matches_img, rmse = process_image_pair(
                    sar_img_path, opt_img_path)
                print(f"  NM: {NM}, NCM: {NCM}, Ratio: {ratio:.2f}, Time: {reg_time:.3f} sec, RMSE: {rmse:.3f}")
                total_NM += NM
                total_NCM += NCM
                registration_times.append(reg_time)
                save_results(sar_img_path, opt_img_path, registered_img, matches_img)
            except Exception as e:
                print(f"  Error processing pair for key {key}: {e}")
        
    overall_ratio = total_NM / total_NCM if total_NCM != 0 else 0
    print(f"Global results : Total NM: {total_NM}, Total NCM: {total_NCM}, Overall ratio: {overall_ratio:.2f}")
    if registration_times:
        average_time = np.mean(registration_times)
        median_time = np.median(registration_times)
        print(f"Registration times - Average: {average_time:.3f} sec, Median: {median_time:.3f} sec")
    print("\n")

if __name__ == "__main__":
    main()
