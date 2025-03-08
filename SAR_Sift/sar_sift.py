import numpy as np
import cv2
from typing import List, Tuple, Optional
from dataclasses import dataclass


# Constants (ported from Sar_sift.h)
SAR_SIFT_MAX_KEYPOINTS = 5000  # Maximum number of keypoints to detect
SAR_SIFT_LATERS = 8  # Number of scale space layers
SAR_SIFT_FACT_RADIUS_ORI = 6.0  # Keypoint boundary radius for main orientation
SAR_SIFT_RADIUS_DES = 12.0  # Descriptor boundary radius
SAR_SIFT_BORDER_CONSTANT = 2  # Edge constant for keypoint detection
SAR_SIFT_ORI_BINS = 36  # Orientation histogram dimensions
SAR_SIFT_ORI_RATIO = 0.8  # Peak ratio for orientation histogram
SAR_SIFT_GLOH_ANG_GRID = 8  # GLOH grid angular division count
SAR_SIFT_GLOH_RATIO_R1_R2 = 0.73  # Ratio of middle circle radius to outer radius in GLOH grid
SAR_SIFT_GLOH_RATIO_R1_R3 = 0.25  # Ratio of inner circle radius to outer radius in GLOH grid
SAR_SIFT_DES_ANG_BINS = 8  # Number of orientation divisions in descriptor (0-360 degrees)
DESCR_MAG_THR = 0.2  # Descriptor threshold value


@dataclass
class Keypoint:
    """Class representing a keypoint in the SAR-SIFT algorithm."""
    pt: Tuple[float, float]  # (x, y) coordinates
    size: float  # Scale of the keypoint
    angle: float  # Main orientation in degrees (0-360)
    response: float  # Response strength
    octave: int  # Layer in scale space


def roewa_kernel(size: int, scale: float) -> np.ndarray:
    """
    Generate ROEWA (Ratio of Exponentially Weighted Averages) kernel.
    
    Args:
        size: Kernel radius
        scale: Exponential weight parameter
        
    Returns:
        2D numpy array containing the kernel
    """
    kernel_size = 2 * size + 1
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            kernel[i + size, j + size] = np.exp(-1.0 * (abs(i) + abs(j)) / scale)
    
    return kernel


def gauss_circle(size: int, scale: float) -> np.ndarray:
    """
    Generate a circular Gaussian kernel.
    
    Args:
        size: Circle radius
        scale: Gaussian standard deviation
        
    Returns:
        2D numpy array containing the kernel
    """
    kernel_size = 2 * size + 1
    gauss_kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    exp_temp = -1.0 / (2 * scale * scale)
    sum_val = 0.0
    
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            if (i*i + j*j) <= size*size:
                value = np.exp((i*i + j*j) * exp_temp)
                sum_val += value
                gauss_kernel[i + size, j + size] = value
    
    # Normalize
    if sum_val > 0:
        gauss_kernel /= sum_val
    
    return gauss_kernel


def meshgrid(x_range: Tuple[int, int], y_range: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a meshgrid similar to MATLAB's meshgrid function.
    
    Args:
        x_range: Range of x coordinates (start, end)
        y_range: Range of y coordinates (start, end)
        
    Returns:
        Tuple of (X, Y) arrays
    """
    x_start, x_end = x_range
    y_start, y_end = y_range
    width = x_end - x_start + 1
    height = y_end - y_start + 1
    
    X = np.zeros((height, width), dtype=np.float32)
    Y = np.zeros((height, width), dtype=np.float32)
    
    for i in range(y_start, y_end + 1):
        X[i - y_start, :] = np.arange(x_start, x_end + 1, dtype=np.float32)
    
    for j in range(x_start, x_end + 1):
        Y[:, j - x_start] = np.arange(y_start, y_end + 1, dtype=np.float32)
    
    return X, Y


def calc_orient_hist(amplit: np.ndarray, orient: np.ndarray, 
                     pt: Tuple[float, float], scale: float, 
                     hist: np.ndarray, n: int) -> float:
    """
    Calculate orientation histogram for a keypoint.
    
    Args:
        amplit: Amplitude image
        orient: Orientation image (in degrees, 0-360)
        pt: Keypoint coordinates (x, y)
        scale: Scale of the keypoint
        hist: Output histogram array (pre-allocated)
        n: Number of histogram bins
        
    Returns:
        Maximum value in the histogram
    """
    num_row, num_col = amplit.shape
    point = (int(round(pt[0])), int(round(pt[1])))
    
    radius = int(round(SAR_SIFT_FACT_RADIUS_ORI * scale))
    radius = min(radius, min(num_row // 2, num_col // 2))
    
    gauss_sig = 2 * scale  # Gaussian weighting standard deviation
    exp_temp = -1.0 / (2 * gauss_sig * gauss_sig)
    
    # Calculate region boundaries
    radius_x_left = max(0, point[0] - radius)
    radius_x_right = min(num_col - 1, point[0] + radius)
    radius_y_up = max(0, point[1] - radius)
    radius_y_down = min(num_row - 1, point[1] + radius)
    
    # Center of the region relative to the cropped region
    center_x = point[0] - radius_x_left
    center_y = point[1] - radius_y_up
    
    # Calculate Gaussian weights
    y_size = radius_y_down - radius_y_up + 1
    x_size = radius_x_right - radius_x_left + 1
    
    gauss_weight = np.zeros((y_size, x_size), dtype=np.float32)
    for i in range(y_size):
        for j in range(x_size):
            y_diff = i - center_y
            x_diff = j - center_x
            gauss_weight[i, j] = np.exp((y_diff*y_diff + x_diff*x_diff) * exp_temp)
    
    # Extract subregions
    sub_amplit = amplit[radius_y_up:radius_y_down+1, radius_x_left:radius_x_right+1]
    sub_orient = orient[radius_y_up:radius_y_down+1, radius_x_left:radius_x_right+1]
    
    # Apply Gaussian weighting
    W = sub_amplit  # No Gaussian weighting, matching the C++ code
    
    # Initialize histogram
    temp_hist = np.zeros(n + 4, dtype=np.float32)
    
    # Compute histogram
    for i in range(sub_orient.shape[0]):
        for j in range(sub_orient.shape[1]):
            if ((i - center_y)**2 + (j - center_x)**2) < radius**2:
                bin_idx = int(round(sub_orient[i, j] * n / 360.0))
                if bin_idx >= n:
                    bin_idx = bin_idx - n
                if bin_idx < 0:
                    bin_idx = bin_idx + n
                temp_hist[bin_idx+2] += W[i, j]
    
    # Smooth histogram
    temp_hist[0] = temp_hist[n]
    temp_hist[1] = temp_hist[n+1]
    temp_hist[n+2] = temp_hist[2]
    temp_hist[n+3] = temp_hist[3]
    
    for i in range(n):
        hist[i] = (temp_hist[i] + temp_hist[i+4]) * (1.0/16.0) + \
                 (temp_hist[i+1] + temp_hist[i+3]) * (4.0/16.0) + \
                 temp_hist[i+2] * (6.0/16.0)
    
    # Get maximum value
    return np.max(hist)


def calc_gloh_descriptor(amplit: np.ndarray, orient: np.ndarray, 
                         pt: Tuple[float, float], scale: float, 
                         main_ori: float, d: int, n: int, 
                         desc_array: np.ndarray) -> None:
    """
    Calculate GLOH descriptor for a keypoint.
    
    Args:
        amplit: Amplitude image
        orient: Orientation image (in degrees, 0-360)
        pt: Keypoint coordinates (x, y)
        scale: Scale of the keypoint
        main_ori: Main orientation in degrees
        d: Number of angular bins in GLOH
        n: Number of orientation bins
        desc_array: Pre-allocated array for descriptor
    """
    point = (int(round(pt[0])), int(round(pt[1])))
    
    # Rotation parameters
    cos_t = np.cos(-main_ori / 180.0 * np.pi)
    sin_t = np.sin(-main_ori / 180.0 * np.pi)
    
    num_rows, num_cols = amplit.shape
    radius = int(round(SAR_SIFT_RADIUS_DES * scale))
    radius = min(radius, min(num_rows // 2, num_cols // 2))
    
    # Calculate region boundaries
    radius_x_left = max(0, point[0] - radius)
    radius_x_right = min(num_cols - 1, point[0] + radius)
    radius_y_up = max(0, point[1] - radius)
    radius_y_down = min(num_rows - 1, point[1] + radius)
    
    # Center of the region relative to the cropped region
    center_x = point[0] - radius_x_left
    center_y = point[1] - radius_y_up
    
    # Extract subregions
    sub_amplit = amplit[radius_y_up:radius_y_down+1, radius_x_left:radius_x_right+1]
    sub_orient = orient[radius_y_up:radius_y_down+1, radius_x_left:radius_x_right+1]
    
    # Create rotated grid centered at keypoint
    x_range = (-center_x, radius_x_right - point[0])
    y_range = (-center_y, radius_y_down - point[1])
    X, Y = meshgrid(x_range, y_range)
    
    # Rotate grid
    c_rot = X * cos_t - Y * sin_t
    r_rot = X * sin_t + Y * cos_t
    
    # Calculate angles and amplitudes in rotated grid
    GLOH_angle = np.arctan2(r_rot, c_rot) * 180.0 / np.pi
    GLOH_angle = np.where(GLOH_angle < 0, GLOH_angle + 360.0, GLOH_angle)
    GLOH_amplit = c_rot**2 + r_rot**2
    
    # Circle radii for GLOH
    R1_pow = float(radius * radius)
    R2_pow = (radius * SAR_SIFT_GLOH_RATIO_R1_R2)**2
    R3_pow = (radius * SAR_SIFT_GLOH_RATIO_R1_R3)**2
    
    # Initialize descriptor
    descriptor_len = (2 * d + 1) * n
    hist = np.zeros(descriptor_len, dtype=np.float32)
    
    # Build descriptor
    sub_rows, sub_cols = sub_amplit.shape
    for i in range(sub_rows):
        for j in range(sub_cols):
            if ((i - center_y)**2 + (j - center_x)**2) < radius**2:
                pix_amplit = sub_amplit[i, j]
                pix_orient = sub_orient[i, j]
                pix_GLOH_amp = GLOH_amplit[i, j]
                pix_GLOH_ang = GLOH_angle[i, j]
                
                # Determine bin indices
                if pix_GLOH_amp < R3_pow:
                    rbin = 0  # Inner circle
                elif pix_GLOH_amp > R2_pow:
                    rbin = 2  # Outer circle
                else:
                    rbin = 1  # Middle circle
                
                cbin = int(round(pix_GLOH_ang * d / 360.0))
                if cbin > d:
                    cbin = cbin - d
                elif cbin <= 0:
                    cbin = cbin + d
                
                obin = int(round(pix_orient * n / 360.0))
                if obin > n:
                    obin = obin - n
                elif obin <= 0:
                    obin = obin + n
                
                # Add to histogram
                if rbin == 0:  # Inner circle
                    hist[obin - 1] += pix_amplit
                else:
                    idx = ((rbin - 1) * d + cbin - 1) * n + n + obin - 1
                    hist[idx] += pix_amplit
    
    # Normalize descriptor
    norm = np.sqrt(np.sum(hist * hist))
    if norm > 0:
        hist = hist / norm
    
    # Threshold
    hist = np.minimum(hist, DESCR_MAG_THR)
    
    # Renormalize
    norm = np.sqrt(np.sum(hist * hist))
    if norm > 0:
        desc_array[:] = hist / norm
    else:
        desc_array[:] = 0


class SarSift:
    """
    SAR-SIFT (Synthetic Aperture Radar Scale-Invariant Feature Transform) implementation.
    
    This class implements the SAR-SIFT algorithm for feature detection and description
    in SAR (Synthetic Aperture Radar) images.
    """
    
    def __init__(self, 
                 nFeatures: int = 0, 
                 Mmax: int = 8, 
                 sigma: float = 2.0, 
                 ratio: float = 2.0**(1.0/3.0), 
                 threshold: float = 0.8, 
                 d: float = 0.04):
        """
        Initialize the SAR-SIFT detector.
        
        Args:
            nFeatures: Maximum number of features to detect (0 means no limit)
            Mmax: Number of scale layers (default 8)
            sigma: Initial scale (default 2.0)
            ratio: Scale factor between layers (default 2^(1/3))
            threshold: Harris function response threshold (default 0.8)
            d: Parameter for sar_harris function (default 0.04)
        """
        self.nFeatures = nFeatures
        self.Mmax = Mmax
        self.sigma = sigma
        self.ratio = ratio
        self.threshold = threshold
        self.d = d
    
    def build_sar_sift_space(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Build the SAR-SIFT scale space.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (sar_harris_fun, amplit, orient) lists containing scale space data
        """
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        
        # Convert to float
        float_image = gray_image.astype(np.float32)
        
        # Allocate memory
        sar_harris_fun = [None] * self.Mmax
        amplit = [None] * self.Mmax
        orient = [None] * self.Mmax
        
        for i in range(self.Mmax):
            # Calculate scale for current layer
            scale = float(self.sigma * (self.ratio ** i))
            radius = int(round(2 * scale))
            
            # Create ROEWA kernel
            kernel = roewa_kernel(radius, scale)
            
            # Create 4 directional filters
            W34 = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float32)
            W12 = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float32)
            W14 = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float32)
            W23 = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float32)
            
            W34[radius+1:, :] = kernel[radius+1:, :]
            W12[:radius, :] = kernel[:radius, :]
            W14[:, radius+1:] = kernel[:, radius+1:]
            W23[:, :radius] = kernel[:, :radius]
            
            # Apply filters
            eps = 0.00001
            M34 = cv2.filter2D(float_image, -1, W34, borderType=cv2.BORDER_CONSTANT) + eps
            M12 = cv2.filter2D(float_image, -1, W12, borderType=cv2.BORDER_CONSTANT) + eps
            M14 = cv2.filter2D(float_image, -1, W14, borderType=cv2.BORDER_CONSTANT) + eps
            M23 = cv2.filter2D(float_image, -1, W23, borderType=cv2.BORDER_CONSTANT) + eps
            
            # Calculate horizontal and vertical gradients
            Gx = np.log(M14 / M23)
            Gy = np.log(M34 / M12)
            
            # Calculate gradient magnitude and orientation
            amplit[i] = np.sqrt(Gx * Gx + Gy * Gy)
            orient[i] = np.arctan2(Gy, Gx) * 180.0 / np.pi
            # Convert negative angles to positive [0, 360)
            orient[i] = np.where(orient[i] < 0, orient[i] + 360.0, orient[i])
            
            # Build SAR-Harris matrix
            Csh_11 = scale * scale * Gx * Gx
            Csh_12 = scale * scale * Gx * Gy
            Csh_22 = scale * scale * Gy * Gy
            
            # Apply Gaussian weighting
            gauss_sigma = np.sqrt(2.0) * scale
            size = int(round(3 * gauss_sigma))
            
            kern_size = (2 * size + 1, 2 * size + 1)
            Csh_11 = cv2.GaussianBlur(Csh_11, kern_size, gauss_sigma)
            Csh_12 = cv2.GaussianBlur(Csh_12, kern_size, gauss_sigma)
            Csh_22 = cv2.GaussianBlur(Csh_22, kern_size, gauss_sigma)
            
            Csh_21 = Csh_12
            
            # Compute SAR-Harris function
            temp_add = Csh_11 + Csh_22
            sar_harris_fun[i] = Csh_11 * Csh_22 - Csh_21 * Csh_12 - float(self.d) * temp_add * temp_add
        
        return sar_harris_fun, amplit, orient
    
    def find_space_extrema(self, 
                          harris_fun: List[np.ndarray], 
                          amplit: List[np.ndarray], 
                          orient: List[np.ndarray]) -> List[Keypoint]:
        """
        Find local extrema in scale space to detect keypoints.
        
        Args:
            harris_fun: List of Harris function responses
            amplit: List of amplitude images
            orient: List of orientation images
            
        Returns:
            List of detected keypoints
        """
        keys = []
        num_rows, num_cols = harris_fun[0].shape
        n = SAR_SIFT_ORI_BINS
        
        for i in range(self.Mmax):
            cur_harris_fun = harris_fun[i]
            cur_amplit = amplit[i]
            cur_orient = orient[i]
            
            for r in range(SAR_SIFT_BORDER_CONSTANT, num_rows - SAR_SIFT_BORDER_CONSTANT):
                for c in range(SAR_SIFT_BORDER_CONSTANT, num_cols - SAR_SIFT_BORDER_CONSTANT):
                    cur_value = cur_harris_fun[r, c]
                    
                    # Check if current point is a local maximum
                    if (cur_value > self.threshold and
                        cur_value > cur_harris_fun[r, c-1] and 
                        cur_value > cur_harris_fun[r, c+1] and
                        cur_value > cur_harris_fun[r-1, c-1] and 
                        cur_value > cur_harris_fun[r-1, c] and 
                        cur_value > cur_harris_fun[r-1, c+1] and
                        cur_value > cur_harris_fun[r+1, c-1] and 
                        cur_value > cur_harris_fun[r+1, c] and 
                        cur_value > cur_harris_fun[r+1, c+1]):
                        
                        x, y = float(c), float(r)
                        layer = i
                        scale = float(self.sigma * (self.ratio ** layer))
                        
                        # Calculate orientation histogram
                        hist = np.zeros(n, dtype=np.float32)
                        max_val = calc_orient_hist(amplit[layer], orient[layer], (x, y), scale, hist, n)
                        
                        # Find orientation peaks
                        mag_thr = max_val * SAR_SIFT_ORI_RATIO
                        
                        for k in range(n):
                            k_left = n - 1 if k <= 0 else k - 1
                            k_right = 0 if k >= n - 1 else k + 1
                            
                            if (hist[k] > mag_thr and 
                                hist[k] >= hist[k_left] and 
                                hist[k] >= hist[k_right]):
                                
                                # Refine peak by interpolation
                                bin_val = float(k) + 0.5 * (hist[k_left] - hist[k_right]) / (hist[k_left] + hist[k_right] - 2 * hist[k])
                                
                                if bin_val < 0:
                                    bin_val += n
                                if bin_val >= n:
                                    bin_val -= n
                                
                                # Create keypoint
                                keypoint = Keypoint(
                                    pt=(x, y),
                                    size=scale,
                                    angle=(360.0 / n) * bin_val,
                                    response=cur_value,
                                    octave=i
                                )
                                
                                keys.append(keypoint)
        
        return keys
    
    def calc_descriptors(self, 
                         amplit: List[np.ndarray], 
                         orient: List[np.ndarray], 
                         keys: List[Keypoint]) -> np.ndarray:
        """
        Calculate descriptors for the detected keypoints.
        
        Args:
            amplit: List of amplitude images
            orient: List of orientation images
            keys: List of keypoints
            
        Returns:
            Array of descriptors, shape (num_keypoints, descriptor_size)
        """
        d = SAR_SIFT_GLOH_ANG_GRID
        n = SAR_SIFT_DES_ANG_BINS
        
        num_keys = len(keys)
        grids = 2 * d + 1
        descriptors = np.zeros((num_keys, grids * n), dtype=np.float32)
        
        for i in range(num_keys):
            point = (keys[i].pt[0], keys[i].pt[1])
            scale = keys[i].size
            layer = keys[i].octave
            main_ori = keys[i].angle
            
            # Calculate descriptor for this keypoint
            calc_gloh_descriptor(
                amplit[layer], orient[layer], point, scale, main_ori, 
                d, n, descriptors[i]
            )
        
        return descriptors
    
    def detect_keys(self, 
                   image: np.ndarray) -> Tuple[List[Keypoint], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Detect keypoints in the image.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (keypoints, harris_fun, amplit, orient)
        """
        # Build scale space
        harris_fun, amplit, orient = self.build_sar_sift_space(image)
        
        # Find extrema in scale space
        keys = self.find_space_extrema(harris_fun, amplit, orient)
        
        # Limit number of features if requested
        if self.nFeatures != 0 and self.nFeatures < len(keys):
            # Sort by response (strongest first)
            keys.sort(key=lambda x: x.response, reverse=True)
            
            # Keep only the strongest features
            keys = keys[:self.nFeatures]
        
        return keys, harris_fun, amplit, orient
    
    def compute_descriptors(self, 
                           keys: List[Keypoint], 
                           amplit: List[np.ndarray], 
                           orient: List[np.ndarray]) -> np.ndarray:
        """
        Compute descriptors for the given keypoints.
        
        Args:
            keys: List of keypoints
            amplit: List of amplitude images
            orient: List of orientation images
            
        Returns:
            Array of descriptors
        """
        return self.calc_descriptors(amplit, orient, keys)