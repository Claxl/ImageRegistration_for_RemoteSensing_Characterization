import numpy as np
import cv2
from skimage import feature
from phasecong3 import phasecong  # Make sure phasecong3.py is in your path


###############################################################################
#                               BLOCK 1: Log-Gabor Convolution Filtering
###############################################################################

def log_gabor_filtering(image, scales=4, orientations=6):
    """
    Perform Log-Gabor filtering on the input image to compute:
      - Maximum moment (m)
      - Minimum moment (min_moment)
      - Filter responses (eo)
    which will be used for subsequent feature detection.

    Args:
        image (ndarray): Input image (grayscale or color).
        scales (int): Number of wavelet scales in the Log-Gabor bank.
        orientations (int): Number of orientations in the Log-Gabor bank.

    Returns:
        m (ndarray): The maximum moment map from phase congruency.
        min_moment (ndarray): The minimum moment map from phase congruency.
        eo (list): The complex filter responses from phase congruency.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image

    # Convert to float64
    image_gray = image_gray.astype(np.float64)

    # Run phase congruency
    m, min_moment, _, _, _, eo, _ = phasecong(
        image_gray,
        nscale=scales,
        norient=orientations,
        minWaveLength=3,
        mult=1.6,
        sigmaOnf=0.75,
        k=3,
        cutOff=1
    )

    return m, min_moment, eo


###############################################################################
#                               BLOCK 2: Feature Detection
###############################################################################

def detect_features(m, min_moment, num_points=5000):
    """
    Detect features (corners and edges) using phase congruency maps:
      - Use 'min_moment' with FAST corner detection.
      - Use 'm' with Canny or other edge detector if needed.
      - Combine corner and edge points, limit to num_points.

    Args:
        m (ndarray): Maximum moment (edge strength).
        min_moment (ndarray): Minimum moment (corner strength).
        num_points (int): Maximum number of keypoints to retain.

    Returns:
        keypoints (ndarray): Array of shape (N, 2) with [x, y] coordinates.
    """
    # Normalize the moments
    m_norm = (m - m.min()) / (m.max() - m.min() + 1e-10)
    min_moment_norm = (min_moment - min_moment.min()) / (min_moment.max() - min_moment.min() + 1e-10)

    # Detect corners (FAST) on min_moment_norm
    corners = feature.corner_fast(min_moment_norm, threshold=0.0001)
    corner_y, corner_x = np.where(corners > 0)

    # Convert to list of points
    keypoints_x = corner_x
    keypoints_y = corner_y

    # If not enough corners, also detect edges in m
    if len(keypoints_x) < num_points:
        edges = cv2.Canny(np.uint8(m_norm * 255), 50, 150)
        edge_y, edge_x = np.where(edges > 0)

        # Add as many edge points as needed
        remaining = num_points - len(keypoints_x)
        if len(edge_x) > remaining:
            idx = np.random.choice(len(edge_x), remaining, replace=False)
            edge_x = edge_x[idx]
            edge_y = edge_y[idx]

        keypoints_x = np.concatenate([keypoints_x, edge_x])
        keypoints_y = np.concatenate([keypoints_y, edge_y])

    # If too many keypoints, randomly downsample to num_points
    if len(keypoints_x) > num_points:
        idx = np.random.choice(len(keypoints_x), num_points, replace=False)
        keypoints_x = keypoints_x[idx]
        keypoints_y = keypoints_y[idx]

    # Combine into Nx2 array
    keypoints = np.vstack((keypoints_x, keypoints_y)).T

    return keypoints


###############################################################################
#                               BLOCK 3: Feature Description
###############################################################################

def compute_orientation(keypoints, magnitude, angle, patch_size=96):
    """
    Compute a dominant orientation for each keypoint by analyzing gradient
    histograms in a local patch (similar to SIFT's orientation assignment).

    Args:
        keypoints (ndarray): (N, 2) array of keypoint coordinates [x, y].
        magnitude (ndarray): Gradient magnitude of the image.
        angle (ndarray): Gradient angle (0-360 degrees).
        patch_size (int): Size of the local patch for orientation.

    Returns:
        oriented_kpts (ndarray): (M, 3) array of [x, y, orientation].
    """
    n_bins = 24
    ORI_PEAK_RATIO = 0.8
    oriented_kpts = []
    half_size = patch_size // 2

    for (x, y) in keypoints:
        x, y = int(x), int(y)
        # Extract local patch
        x1, x2 = x - half_size, x + half_size
        y1, y2 = y - half_size, y + half_size

        # Check boundaries
        if (x1 < 0 or y1 < 0 or x2 >= magnitude.shape[1] or y2 >= magnitude.shape[0]):
            continue

        patch_mag = magnitude[y1:y2, x1:x2]
        patch_ang = angle[y1:y2, x1:x2]

        # Build orientation histogram
        hist = np.zeros(n_bins, dtype=np.float64)
        rows, cols = patch_mag.shape
        for i in range(rows):
            for j in range(cols):
                bin_idx = int((patch_ang[i, j] / 360.0) * n_bins) % n_bins
                hist[bin_idx] += patch_mag[i, j]

        # Smooth histogram
        kernel = np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16.0
        # Pad for circular convolution
        hist_padded = np.concatenate([hist[-2:], hist, hist[:2]])
        hist_smooth = np.convolve(hist_padded, kernel, mode='valid')  # length = n_bins
        max_idx = np.argmax(hist_smooth)
        max_val = hist_smooth[max_idx]

        # Primary orientation
        angles = [max_idx * (360.0 / n_bins)]

        # Check secondary peaks
        for i in range(n_bins):
            if i == max_idx:
                continue
            if hist_smooth[i] > ORI_PEAK_RATIO * max_val:
                angles.append(i * (360.0 / n_bins))

        for a in angles:
            oriented_kpts.append([x, y, a])

    return np.array(oriented_kpts)


def compute_max_index_map(eo):
    """
    From the list of Log-Gabor responses (eo), compute the orientation
    that has the maximum amplitude at each pixel (the MIM: Maximum Index Map).

    Args:
        eo (list): A list of shape [num_orientations][num_scales][H, W]
                   containing complex filter responses.

    Returns:
        mim (ndarray): 2D array of the same size as the image indicating
                       the orientation index (1-based) that is strongest
                       at each pixel.
    """
    # Suppose eo is: eo[orientation][scale][height, width]
    orientations = len(eo)
    scales = len(eo[0])
    # Stack amplitude sums for each orientation
    # amplitude[o, y, x] = sum of abs(eo[o][s][y, x]) over all scales
    amplitude = []
    for o in range(orientations):
        sum_abs = np.sum([np.abs(eo[o][s]) for s in range(scales)], axis=0)
        amplitude.append(sum_abs)

    amplitude = np.stack(amplitude, axis=0)  # shape: (orientations, H, W)
    # Argmax along orientation dimension
    mim = np.argmax(amplitude, axis=0) + 1  # 1-based index
    return mim.astype(np.int32)


def describe_features(image, oriented_kpts, eo, patch_size=96, grid_x=6, grid_y=6):
    """
    Build a descriptor for each keypoint using the Maximum Index Map (MIM),
    recoding it based on the dominant orientation, and then computing
    a histogram-based descriptor (SIFT-like).

    Args:
        image (ndarray): The original (grayscale) image.
        oriented_kpts (ndarray): (N, 3) array of [x, y, angle].
        eo (list): Log-Gabor filter responses.
        patch_size (int): Size of the patch for descriptor.
        grid_x (int): Number of cells horizontally in the patch.
        grid_y (int): Number of cells vertically in the patch.

    Returns:
        descriptors (ndarray): A 2D array, one row per keypoint descriptor.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    mim = compute_max_index_map(eo)
    h, w = gray.shape
    half_size = patch_size // 2
    n_orientations = len(eo)

    descriptors = []
    hist_bins = np.arange(1, n_orientations + 2)

    for (x, y, angle) in oriented_kpts:
        x, y = int(x), int(y)

        # Skip if too close to boundary
        if x < half_size or x >= w - half_size or y < half_size or y >= h - half_size:
            continue

        patch_mim = mim[y - half_size:y + half_size, x - half_size:x + half_size]
        if patch_mim.shape != (patch_size, patch_size):
            continue

        # Compute global histogram for this patch
        flat_patch = patch_mim.ravel()
        global_hist = np.bincount(flat_patch, minlength=n_orientations + 1)[1:]  # skip bin 0
        dominant_idx = np.argmax(global_hist) + 1  # 1-based
        max_val = global_hist[dominant_idx - 1]

        # Check for second peak
        global_hist_copy = global_hist.copy()
        global_hist_copy[dominant_idx - 1] = 0
        second_idx = np.argmax(global_hist_copy) + 1
        create_second_descriptor = (global_hist_copy[second_idx - 1] > 0.8 * max_val)

        # Recode patch based on dominant_idx
        recoded_mim = np.where(
            patch_mim >= dominant_idx,
            patch_mim - dominant_idx + 1,
            patch_mim + n_orientations - dominant_idx + 1
        )

        # Divide into grid cells and compute histograms
        cell_h = patch_size // grid_y
        cell_w = patch_size // grid_x
        descriptor = []
        for gy in range(grid_y):
            for gx in range(grid_x):
                cell = recoded_mim[gy*cell_h:(gy+1)*cell_h, gx*cell_w:(gx+1)*cell_w]
                cell_hist = np.bincount(cell.ravel(), minlength=n_orientations+1)[1:]
                if cell_hist.sum() > 0:
                    cell_hist = cell_hist / cell_hist.sum()
                descriptor.extend(cell_hist.tolist())
        descriptors.append(descriptor)

        # Possibly create a second descriptor if the second peak is significant
        if create_second_descriptor:
            recoded_mim2 = np.where(
                patch_mim >= second_idx,
                patch_mim - second_idx + 1,
                patch_mim + n_orientations - second_idx + 1
            )
            descriptor2 = []
            for gy in range(grid_y):
                for gx in range(grid_x):
                    cell = recoded_mim2[gy*cell_h:(gy+1)*cell_h, gx*cell_w:(gx+1)*cell_w]
                    cell_hist = np.bincount(cell.ravel(), minlength=n_orientations+1)[1:]
                    if cell_hist.sum() > 0:
                        cell_hist = cell_hist / cell_hist.sum()
                    descriptor2.extend(cell_hist.tolist())
            descriptors.append(descriptor2)

    return np.array(descriptors)


###############################################################################
#                               BLOCK 4: Feature Matching
###############################################################################

def match_descriptors(des1, des2, match_threshold=100, ratio_threshold=1.0):
    """
    Match two sets of descriptors by nearest-neighbor distance.

    Args:
        des1 (ndarray): Descriptors from the first image (N1 x D).
        des2 (ndarray): Descriptors from the second image (N2 x D).
        match_threshold (float): Maximum allowed distance for a match.
        ratio_threshold (float): Ratio test threshold (if < 1.0).

    Returns:
        matches (list): List of (i, j) pairs of matched indices.
    """
    if des1.size == 0 or des2.size == 0:
        return []

    distances = np.zeros((des1.shape[0], des2.shape[0]), dtype=np.float64)
    for i in range(des1.shape[0]):
        diff = des2 - des1[i]
        # L2 distance
        distances[i] = np.sum(diff**2, axis=1)

    matches = []
    for i in range(distances.shape[0]):
        sorted_idx = np.argsort(distances[i])
        best_idx = sorted_idx[0]
        best_dist = distances[i, best_idx]
        if best_dist > match_threshold:
            continue
        # Optional ratio test
        if ratio_threshold < 1.0 and len(sorted_idx) > 1:
            second_best_idx = sorted_idx[1]
            second_best_dist = distances[i, second_best_idx]
            if best_dist / max(second_best_dist, 1e-10) > ratio_threshold:
                continue
        matches.append((i, best_idx))

    return matches


def ransac_transform(pts1, pts2, transform_type='similarity', error_threshold=3, max_iterations=1000):
    """
    Estimate a robust transformation between matched points using RANSAC.
    """
    best_model = None
    best_inliers = []
    n_points = pts1.shape[0]

    # Minimum points for each transform
    if transform_type == 'similarity':
        min_points = 2
    elif transform_type == 'affine':
        min_points = 3
    else:
        min_points = 4

    if n_points < min_points:
        return None, []

    for _ in range(max_iterations):
        indices = np.random.choice(n_points, min_points, replace=False)
        sample1 = pts1[indices]
        sample2 = pts2[indices]

        # Estimate model
        if transform_type == 'similarity':
            model = cv2.estimateAffinePartial2D(sample1, sample2)[0]
            if model is None:
                continue
            model_3x3 = np.vstack([model, [0, 0, 1]])
        elif transform_type == 'affine':
            model = cv2.estimateAffine2D(sample1, sample2)[0]
            if model is None:
                continue
            model_3x3 = np.vstack([model, [0, 0, 1]])
        else:
            # Homography (perspective)
            model_3x3, _ = cv2.findHomography(sample1, sample2)
            if model_3x3 is None:
                continue

        # Compute errors
        if transform_type == 'perspective':
            ones = np.ones((n_points, 1))
            homogeneous = np.hstack([pts1, ones])
            projected = (model_3x3 @ homogeneous.T).T
            proj_xy = projected[:, :2] / (projected[:, 2:] + 1e-10)
        else:
            # For similarity/affine
            A = model_3x3[:2, :2]
            t = model_3x3[:2, 2]
            proj_xy = (pts1 @ A.T) + t

        errors = np.sqrt(np.sum((proj_xy - pts2)**2, axis=1))
        inliers = np.where(errors < error_threshold)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_model = model_3x3

    # Refine model using all inliers
    if len(best_inliers) >= min_points:
        inlier_pts1 = pts1[best_inliers]
        inlier_pts2 = pts2[best_inliers]
        if transform_type == 'similarity':
            model = cv2.estimateAffinePartial2D(inlier_pts1, inlier_pts2)[0]
            best_model = np.vstack([model, [0, 0, 1]])
        elif transform_type == 'affine':
            model = cv2.estimateAffine2D(inlier_pts1, inlier_pts2)[0]
            best_model = np.vstack([model, [0, 0, 1]])
        else:
            best_model, _ = cv2.findHomography(inlier_pts1, inlier_pts2)

    return best_model, best_inliers


###############################################################################
#                               PIPELINE DEMO
###############################################################################

def rift2_pipeline(image1, image2, num_points=5000, patch_size=96):
    """
    Demonstrate the complete pipeline of:
      1. Log-Gabor filtering
      2. Feature detection
      3. Feature description
      4. Feature matching + RANSAC

    Args:
        image1 (ndarray): First image (BGR or grayscale).
        image2 (ndarray): Second image (BGR or grayscale).
        num_points (int): Maximum number of features to detect.
        patch_size (int): Local patch size for orientation & descriptors.

    Returns:
        (matched_pts1, matched_pts2, transform, inliers)
    """
    # ------------------------- BLOCK 1: Log-Gabor Filtering -------------------------
    m1, min_m1, eo1 = log_gabor_filtering(image1)
    m2, min_m2, eo2 = log_gabor_filtering(image2)

    # ------------------------- BLOCK 2: Feature Detection ---------------------------
    kpts1 = detect_features(m1, min_m1, num_points=num_points)
    kpts2 = detect_features(m2, min_m2, num_points=num_points)

    # ------------------------- Prepare for orientation calculation ------------------
    # We need gradient magnitude and angle for each image
    def compute_gradients(img):
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sobelx**2 + sobely**2)
        ang = np.arctan2(sobely, sobelx) * 180.0 / np.pi
        ang[ang < 0] += 360.0
        return mag, ang

    mag1, ang1 = compute_gradients(image1)
    mag2, ang2 = compute_gradients(image2)

    # ------------------------- BLOCK 3: Feature Description -------------------------
    # 3a. Assign orientation
    oriented_kpts1 = compute_orientation(kpts1, mag1, ang1, patch_size=patch_size)
    oriented_kpts2 = compute_orientation(kpts2, mag2, ang2, patch_size=patch_size)

    # 3b. Build descriptors (MIM-based)
    des1 = describe_features(image1, oriented_kpts1, eo1, patch_size=patch_size)
    des2 = describe_features(image2, oriented_kpts2, eo2, patch_size=patch_size)

    # ------------------------- BLOCK 4: Feature Matching ---------------------------
    matches = match_descriptors(des1, des2, match_threshold=100, ratio_threshold=1.0)
    if not matches:
        print("No matches found.")
        return None, None, None, None

    # Convert match indices to point coordinates
    pts1 = np.array([oriented_kpts1[i][:2] for (i, _) in matches])
    pts2 = np.array([oriented_kpts2[j][:2] for (_, j) in matches])

    # Remove duplicates
    _, unique_idx1 = np.unique(pts1, axis=0, return_index=True)
    pts1 = pts1[unique_idx1]
    pts2 = pts2[unique_idx1]

    _, unique_idx2 = np.unique(pts2, axis=0, return_index=True)
    pts1 = pts1[unique_idx2]
    pts2 = pts2[unique_idx2]

    # RANSAC to remove outliers and get transformation
    transform, inliers = ransac_transform(pts1, pts2, transform_type='similarity', error_threshold=3)
    if transform is None or len(inliers) == 0:
        print("No valid transformation found.")
        return pts1, pts2, None, None

    matched_pts1 = pts1[inliers]
    matched_pts2 = pts2[inliers]
    return matched_pts1, matched_pts2, transform, inliers


###############################################################################
#                             EXAMPLE USAGE
###############################################################################
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example usage with two images
    img1_path = "pair1.jpg"
    img2_path = "pair2.jpg"

    image1 = cv2.imread(img1_path)
    image2 = cv2.imread(img2_path)

    if image1 is None or image2 is None:
        print(f"Could not load images: {img1_path}, {img2_path}")
    else:
        matched_pts1, matched_pts2, transform, inliers = rift2_pipeline(image1, image2)
        if matched_pts1 is not None:
            print(f"Number of inlier matches: {len(inliers)}")

            # Simple visualization of matched points (side by side)
            h1, w1 = image1.shape[:2]
            h2, w2 = image2.shape[:2]
            out = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
            out[:h1, :w1] = image1
            out[:h2, w1:w1+w2] = image2

            for (x1, y1), (x2, y2) in zip(matched_pts1, matched_pts2):
                x2_shifted = x2 + w1
                cv2.circle(out, (int(x1), int(y1)), 3, (0, 0, 255), -1)
                cv2.circle(out, (int(x2_shifted), int(y2)), 3, (0, 255, 0), -1)
                cv2.line(out, (int(x1), int(y1)), (int(x2_shifted), int(y2)), (0, 255, 255), 1)

            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            plt.title(f"Matches: {len(inliers)} inliers")
            plt.axis('off')
            plt.show()
