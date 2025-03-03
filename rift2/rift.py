import numpy as np
import cv2

# Import the RIFT2 pipeline functions already defined
# (Assuming these are all in your rift2/ package or similar)
from .FeatureDetection import FeatureDetection
from .kptsOrientation import kptsOrientation
from .FeatureDescribe import FeatureDescribe
from .FSC import FSC
from .image_fusion import image_fusion

class RIFT:
    """
    RIFT encapsulates the main pipeline for detecting, describing, and
    matching keypoints between two multimodal images, as well as optionally
    fusing them with the estimated transformation.

    Attributes:
        s (int): Number of scales for feature detection.
        o (int): Number of orientations for log-Gabor filters.
        max_keypoints (int): Maximum number of keypoints to detect.
        patch_size (int): Patch size used during orientation and description.
        use_orientation (bool): If True, computes orientation for each keypoint.
        ratio_thresh (float): Ratio threshold for the ratio test in matching.
        change_form (str): Transformation model (e.g., "similarity", "affine", "perspective").
        error_thresh (float): Threshold for the inlier test in FSC.
    """
    
    def __init__(
        self,
        s=4,
        o=6,
        max_keypoints=5000,
        patch_size=96,
        use_orientation=True,
        ratio_thresh=0.8,
        change_form="similarity",
        error_thresh=3.0
    ):
        """
        Initialize RIFT parameters.

        Args:
            s (int): Number of scales for feature detection (phasecong3).
            o (int): Number of orientations for feature detection (phasecong3).
            max_keypoints (int): Max number of keypoints to detect.
            patch_size (int): Patch size for orientation and descriptor extraction.
            use_orientation (bool): If True, compute orientation for each keypoint.
            ratio_thresh (float): Lowe's ratio test threshold for matching.
            change_form (str): Transformation model for FSC. Options: ['similarity','affine','perspective'].
            error_thresh (float): Inlier threshold for FSC.
        """
        self.s = s
        self.o = o
        self.max_keypoints = max_keypoints
        self.patch_size = patch_size
        self.use_orientation = use_orientation
        self.ratio_thresh = ratio_thresh
        self.change_form = change_form
        self.error_thresh = error_thresh


    def detect_features(self, image):
        """
        Detect keypoints using RIFT’s approach (phase congruency + FAST edges/corners).

        Args:
            image (np.ndarray): Input image (BGR or grayscale).

        Returns:
            keypoints (np.ndarray): 2 x N array of detected keypoint locations (x,y).
            m (np.ndarray): The normalized maximum moment from phasecong3.
            eo (list): The log-Gabor filtering output needed in subsequent steps.
        """
        # Convert grayscale if needed
        if image.ndim == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # FeatureDetection() returns (kpts, m, eo)
        keypoints, m, eo = FeatureDetection(
            gray,
            nscale=self.s,
            norient=self.o,
            npt=self.max_keypoints
        )
        return keypoints, m, eo


    def compute_orientation(self, keypoints, m, orientation_flag=True):
        """
        Compute the orientation of keypoints.

        Args:
            keypoints (np.ndarray): 2 x N array of keypoint locations.
            m (np.ndarray): Normalized maximum moment from phasecong3.
            orientation_flag (bool): If True, compute orientation. Otherwise, set 0.

        Returns:
            oriented_keypoints (np.ndarray): 3 x N array of [x, y, orientation].
        """
        kpts_oriented = kptsOrientation(
            keypoints, 
            m,
            orientation_flag,
            self.patch_size
        )
        return kpts_oriented


    def describe_features(self, image, eo, oriented_keypoints):
        """
        Extract the RIFT descriptor for each keypoint.

        Args:
            image (np.ndarray): Original image (uint8).
            eo (list): Output from phasecong3 (Log-Gabor responses).
            oriented_keypoints (np.ndarray): 3 x N array of [x, y, orientation].
        
        Returns:
            descriptors (np.ndarray): (N, D) array of RIFT feature descriptors.
        """
        # Convert to color if single channel, so FeatureDescribe can be consistent
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # The FeatureDescribe function in your code typically is:
        # descriptors = FeatureDescribe(im, eo, oriented_keypoints, patch_size, 6, 6)
        # returning shape (D, N). We will transpose to (N, D).
        descriptors = FeatureDescribe(
            image,
            eo,
            oriented_keypoints,
            self.patch_size,
            6,
            6
        )
        descriptors = descriptors.T  # to shape [N, D]
        return descriptors


    def match_features(self, desc1, desc2):
        """
        Match the descriptors using a BFMatcher with ratio test.

        Args:
            desc1 (np.ndarray): (N1, D) array of descriptors from image1.
            desc2 (np.ndarray): (N2, D) array of descriptors from image2.

        Returns:
            good_matches (list of cv2.DMatch): The filtered matches after ratio test.
        """
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        knn_matches = bf.knnMatch(
            desc1.astype(np.float32),
            desc2.astype(np.float32),
            k=2
        )

        good_matches = []
        for pair in knn_matches:
            # Usually pair has 2 matches: m[0], m[1]
            if len(pair) == 2:
                if pair[0].distance < self.ratio_thresh * pair[1].distance:
                    good_matches.append(pair[0])
            elif len(pair) == 1:
                # Only one neighbor
                good_matches.append(pair[0])

        return good_matches


    def estimate_transformation(self, kp1, kp2):
        """
        Estimate transformation (similarity, affine, or perspective) and
        remove outliers using the FSC method.

        Args:
            kp1 (np.ndarray): Nx2 array of matched keypoint coordinates in image1.
            kp2 (np.ndarray): Nx2 array of matched keypoint coordinates in image2.

        Returns:
            H (np.ndarray): 3x3 transform matrix estimated by FSC.
            rmse (float): Root mean square error of inliers.
            inliers_kp1 (np.ndarray): Inlier coordinates in image1.
            inliers_kp2 (np.ndarray): Inlier coordinates in image2.
        """
        # The FSC function returns (solution, rmse, cor1_new, cor2_new)
        H, rmse, inliers_kp1, inliers_kp2 = FSC(
            kp1,
            kp2,
            change_form=self.change_form,
            error_t=self.error_thresh
        )
        return H, rmse, inliers_kp1, inliers_kp2

    def match_to_coords(self, oriented_k1, oriented_k2, good):
        matched_pts1 = []
        matched_pts2 = []
        for g in good:
            matched_pts1.append(oriented_k1[:2, g.queryIdx])  # [x,y]
            matched_pts2.append(oriented_k2[:2, g.trainIdx])  # [x,y]
        matched_pts1 = np.array(matched_pts1)
        matched_pts2 = np.array(matched_pts2)

        # Remove exact duplicates if needed
        matched_pts2_unique, idxs = np.unique(matched_pts2, axis=0, return_index=True)
        matched_pts1_unique = matched_pts1[idxs]
        return matched_pts1_unique, matched_pts2_unique

    def register_and_fuse(self, im1, im2):
        """
        Complete pipeline: detect, describe, match, estimate transform, and fuse.

        Args:
            im1 (np.ndarray): Image 1 (reference).
            im2 (np.ndarray): Image 2 (to be aligned onto Image 1).

        Returns:
            H (np.ndarray): 3x3 transform from im2 to im1’s space
            fused_image (np.ndarray): Result of image fusion for a sanity check
            (matching_info dict): Extra details for intermediate steps
        """
        print("Running RIFT pipeline...")
        print("Detecting features...")
        # 1) Detect features
        k1, m1, eo1 = self.detect_features(im1)
        k2, m2, eo2 = self.detect_features(im2)

        print("Computing orientation...")
        # 2) Orientation
        oriented_k1 = self.compute_orientation(k1, m1, self.use_orientation)
        oriented_k2 = self.compute_orientation(k2, m2, self.use_orientation)


        print("Describing features...")
        # 3) Describe
        desc1 = self.describe_features(im1, eo1, oriented_k1)
        desc2 = self.describe_features(im2, eo2, oriented_k2)

        print("Matching features...")
        # 4) Match
        good = self.match_features(desc1, desc2)
        if not good:
            print("[Warning] No matches found.")
            return None, None, {}

        # Convert matches to coordinate arrays
        matched_pts1_unique, matched_pts2_unique = self.match_to_coords(oriented_k1, oriented_k2, good)
        print("Unique matches:", len(matched_pts1_unique))
        print("Transform estimation...")
        # 5) FSC transform estimation
        H, rmse, inliers_kp1, inliers_kp2 = self.estimate_transformation(
            matched_pts1_unique,
            matched_pts2_unique
        )

        # 6) Fuse for debugging
        #   If you want to see how well image2 overlays on image1
        #   NOTE: "image_fusion" in your original code creates a figure,
        #         modifies intensities, etc. Possibly you want a simpler warp?
        print("Fusing images...")
        fused = image_fusion(im1, im2, H)

        # Return results
        matching_info = {
            "matchedPoints1": matched_pts1_unique,
            "matchedPoints2": matched_pts2_unique,
            "inliersPoints1": inliers_kp1,
            "inliersPoints2": inliers_kp2,
            "H": H,
            "rmse": rmse
        }

        return H, fused, matching_info


if __name__ == "__main__":
    """
    Example usage:
    """
    # Paths to your images
    path1 = "../DATASET/OSdataset/512/test/sar/sar1.png"
    path2 = "../DATASET/OSdataset/512/test/opt/opt1.png"

    # Read images
    im1 = cv2.imread(path1, cv2.IMREAD_COLOR)
    im2 = cv2.imread(path2, cv2.IMREAD_COLOR)

    if im1 is None or im2 is None:
        print("Error: Unable to read input images. Check paths.")
        exit(1)

    # Create RIFT object
    rift = RIFT(
        s=4,
        o=6,
        max_keypoints=5000,
        patch_size=96,
        use_orientation=True,
        ratio_thresh=1.0,         # you can tweak ratio test
        change_form="similarity", # could also be 'affine' or 'perspective'
        error_thresh=3.0
    )

    # Run the pipeline
    H, fused_image, info = rift.register_and_fuse(im1, im2)

    print("Estimated transformation matrix:")
    print(H)
    print("RMSE:", info["rmse"])
    # Show the fused result if you want
    if fused_image is not None:
        cv2.imshow("Fused", fused_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
