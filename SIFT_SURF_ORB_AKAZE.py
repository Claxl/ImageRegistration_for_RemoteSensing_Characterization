import cv2
import numpy as np
import time
import os
import glob
import re
import argparse

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

def create_detector_and_matcher(method):
    """
    Given a method name ('SIFT', 'SURF', or 'ORB'), returns a tuple (detector, matcher)
    properly configured for that method.
    """
    if method.upper() == "SIFT":
        detector = cv2.SIFT_create()
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    elif method.upper() == "SURF":
        try:
            detector = cv2.xfeatures2d.SURF_create()
        except AttributeError:
            raise AttributeError("SURF is not available. Please install opencv-contrib-python.")
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    elif method.upper() == "ORB":
        detector = cv2.ORB_create()
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    elif method.upper() == "AKAZE":
        detector = cv2.AKAZE_create()
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        raise ValueError(f"Method {method} not recognized.")
    return detector, matcher

def save_results(sar_path, opt_path, registered_img, matches_img, method):
    """
    Saves the registered SAR image and the image with drawn matches into an output folder.
    The filenames include the method name and the base filenames of the input images.
    """
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    base_sar = os.path.splitext(os.path.basename(sar_path))[0]
    base_opt = os.path.splitext(os.path.basename(opt_path))[0]
    
    reg_filename = os.path.join(output_dir, f"{method}_registered_{base_sar}_to_{base_opt}.png")
    matches_filename = os.path.join(output_dir, f"{method}_matches_{base_sar}_to_{base_opt}.png")
    
    if registered_img is not None:
        cv2.imwrite(reg_filename, registered_img)
    if matches_img is not None:
        cv2.imwrite(matches_filename, matches_img)

def process_image_pair(sar_img_path, opt_img_path, detector, matcher, ratio_thresh=0.7):
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
    if sar_img is None or opt_img is None:
        raise IOError(f"Error loading images: {sar_img_path} or {opt_img_path}")
    # Extract keypoints and descriptors
    kp_sar, desc_sar = detector.detectAndCompute(sar_img, None)
    kp_opt, desc_opt = detector.detectAndCompute(opt_img, None)
    if desc_sar is None or desc_opt is None:
        raise ValueError("No descriptors found in one or both images.")
    
    # Perform matching with k=2 and apply Lowe's ratio test
    matches = matcher.knnMatch(desc_sar, desc_opt, k=2)
    good_matches = []
    for m_n in matches:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    NM = len(good_matches)
    
    # Initialize output images
    registered_img = None
    matches_img = None
    
    # Compute homography if enough good matches are found
    if NM >= 4:
        src_pts = np.float32([kp_sar[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_opt[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0)
        if M is not None:
            mask = mask.ravel()
            NCM = int(np.sum(mask))  # number of correct (inlier) matches
            height, width = opt_img.shape
            registered_img = cv2.warpPerspective(sar_img, M, (width, height))
            matches_img = cv2.drawMatches(sar_img, kp_sar, opt_img, kp_opt, good_matches, None, 
                                          matchesMask=mask.tolist(), flags=2)
        else:
            NCM = 0
    else:
        NCM = 0

    ratio = NM / NCM if NCM != 0 else 0
    reg_time = time.time() - start_reg_time
    
    return NM, NCM, ratio, reg_time, registered_img, matches_img

def main():
    def parse_arguments():
        parser = argparse.ArgumentParser(description="Process SAR and Optical image pairs.")
        parser.add_argument("--sar_folder", type=str, required=True, help="Path to the folder containing SAR images.")
        parser.add_argument("--opt_folder", type=str, required=True, help="Path to the folder containing Optical images.")
        return parser.parse_args()

    args = parse_arguments()
    sar_folder = args.sar_folder
    opt_folder = args.opt_folder

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

    methods = [ "SIFT", "SURF", "ORB", "AKAZE"]
    for method in methods:
        print(f"\n==== Processing using {method} ====")
        try:
            detector, matcher = create_detector_and_matcher(method)
        except Exception as e:
            print(f"Skipping method {method} due to error: {e}")
            continue

        total_NM = 0
        total_NCM = 0
        registration_times = []
        
        for key in common_keys:
            sar_img_path = sar_dict[key]
            opt_img_path = opt_dict[key]
            print(f"Processing pair: SAR: {os.path.basename(sar_img_path)} <-> Optical: {os.path.basename(opt_img_path)}")
            try:
                NM, NCM, ratio, reg_time, registered_img, matches_img = process_image_pair(
                    sar_img_path, opt_img_path, detector, matcher)
                print(f"  NM: {NM}, NCM: {NCM}, Ratio: {ratio:.2f}, Time: {reg_time:.3f} sec")
                total_NM += NM
                total_NCM += NCM
                registration_times.append(reg_time)
                save_results(sar_img_path, opt_img_path, registered_img, matches_img, method)
            except Exception as e:
                print(f"  Error processing pair for key {key}: {e}")
        
        overall_ratio = total_NM / total_NCM if total_NCM != 0 else 0
        print(f"Global results for {method}: Total NM: {total_NM}, Total NCM: {total_NCM}, Overall ratio: {overall_ratio:.2f}")
        if registration_times:
            average_time = np.mean(registration_times)
            median_time = np.median(registration_times)
            print(f"Registration times for {method} - Average: {average_time:.3f} sec, Median: {median_time:.3f} sec")
        print("\n")
        with open("output/"+method+"_output.txt", "w", encoding="utf-8") as f:
            f.write(f"Global results : Total NM: {total_NM}, Total NCM: {total_NCM}, Overall ratio: {overall_ratio:.2f}\n")
            f.write(f"Registration times - Average: {average_time:.3f} sec, Median: {median_time:.3f} sec\n")
            f.close()

if __name__ == "__main__":
    main()
