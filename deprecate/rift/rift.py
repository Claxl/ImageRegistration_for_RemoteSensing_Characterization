import numpy as np
import cv2
from scipy import signal
import math
import matplotlib.pyplot as plt
from functools import lru_cache

class RIFT:
    """
    Optimized Radiation-invariant Feature Transform (RIFT) implementation.
    
    This class implements the RIFT algorithm as described in:
    "RIFT: Multi-modal Image Matching Based on Radiation-invariant Feature Transform"
    by Jiayuan Li, Qingwu Hu, and Mingyao Ai.
    """
    
    def __init__(self, scales=4, orientations=6, patch_size=96):
        """
        Initialize RIFT algorithm parameters.
        
        Args:
            scales: Number of scales for log-Gabor filter (default: 4)
            orientations: Number of orientations for log-Gabor filter (default: 6)
            patch_size: Size of patch for feature descriptor (default: 96)
        """
        self.scales = scales
        self.orientations = orientations
        self.patch_size = patch_size
        self.wavelength = 3  # Wavelength of smallest scale filter
        self.mult = 1.6  # Scaling factor between successive filters
        self.sigma_onf = 0.75  # Ratio of the standard deviation of the Gaussian
        self.g = 3  # Controls sharpness of threshold
        self.k = 1  # No. of standard deviations of noise
        
        # Create matcher once during initialization instead of per match call
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        
        # Pre-calculate window for non-maximal suppression
        self.nms_window = np.ones((5, 5), np.uint8)
        
        # Pre-calculate angle values for orientations
        self.orientation_angles = np.array([o * np.pi / orientations for o in range(orientations)])
        self.cos_angles = np.cos(self.orientation_angles)
        self.sin_angles = np.sin(self.orientation_angles)
        self.cos2_angles = np.cos(self.orientation_angles)**2
        self.sin2_angles = np.sin(self.orientation_angles)**2
        self.cos_sin_angles = self.cos_angles * self.sin_angles
    
    def compute_phase_congruency(self, image):
        """
        Compute phase congruency for an image.
        
        Args:
            image: Input image
            
        Returns:
            pc: Phase congruency map
            m_psi: Minimum moment map (corner features)
            M_psi: Maximum moment map (edge features)
            eo: Filter response matrix
        """
        from phasecong3 import phasecong
        
        # Convert to grayscale if needed - avoid copy if already grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Normalize to [0,1] - use more efficient float32 instead of float64
        gray_float = gray.astype(np.float32) / 255.0
        
        # Compute phase congruency
        pc, _, _, _, _, eo, _ = phasecong(
            gray_float,
            self.scales,
            self.orientations,
            self.wavelength,
            mult=self.mult,
            sigmaOnf=self.sigma_onf,
            g=self.g,
            k=self.k
        )
        
        # Calculate moment maps for corner and edge detection
        a, b, c = self.compute_moment_quantities(pc, self.orientations)
        
        # Calculate principal axis, minimum and maximum moments
        # Vectorized computation to avoid individual pixel processing
        psi = np.zeros_like(a, dtype=np.float32)
        mask = (a != c)
        
        # Calculate only where mask is True
        if np.any(mask):
            psi[mask] = 0.5 * np.arctan(b[mask] / (a[mask] - c[mask]))
        
        # Handle special case efficiently
        if np.any(~mask):
            psi[~mask] = np.pi/4 if np.mean(b[~mask]) >= 0 else -np.pi/4
        
        # Calculate maximum and minimum moments - vectorized computation
        b_squared = b**2
        a_minus_c_squared = (a - c)**2
        sqrt_term = np.sqrt(b_squared + a_minus_c_squared)
        
        M_psi = 0.5 * (c + a + sqrt_term)  # Maximum moment (edge map)
        m_psi = 0.5 * (c + a - sqrt_term)  # Minimum moment (corner map)
        
        return pc, m_psi, M_psi, eo
    
    def compute_moment_quantities(self, pc, orientations):
        """
        Compute the moment quantities for PC maps.
        
        Args:
            pc: Phase congruency map
            orientations: Number of orientations
            
        Returns:
            a, b, c: Moment quantities
        """
        # More efficient simulation of independent PC maps
        pc_o = pc / orientations
        
        # Vectorized calculation of moment quantities
        a = np.zeros_like(pc, dtype=np.float32)
        b = np.zeros_like(pc, dtype=np.float32)
        c = np.zeros_like(pc, dtype=np.float32)
        
        # Use pre-calculated values to speed up computation
        for o in range(orientations):
            a += pc_o * self.cos2_angles[o]
            b += pc_o * self.cos_sin_angles[o]
            c += pc_o * self.sin2_angles[o]
        
        return a, b, c
    
    def detect_features(self, m_psi, M_psi):
        """
        Detect corner and edge features from moment maps.
        
        Args:
            m_psi: Minimum moment map (corner features)
            M_psi: Maximum moment map (edge features)
            
        Returns:
            corner_kps: Corner keypoints
            edge_kps: Edge keypoints
        """
        # Normalize moment maps - more efficient to do in-place
        m_psi_norm = cv2.normalize(m_psi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        M_psi_norm = cv2.normalize(M_psi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Detect corner features (local maxima of m_psi) with pre-calculated window
        m_psi_dilated = cv2.dilate(m_psi_norm, self.nms_window)
        
        # Use numpy's logical AND directly instead of OpenCV's bitwise_and
        # This avoids type conversion issues with boolean arrays
        mask = (m_psi_norm == m_psi_dilated) & (m_psi_norm > 30)
        
        # Use numpy's where to find coordinates efficiently
        y_coords, x_coords = np.where(mask)
        
        # Create keypoints more efficiently with list comprehension
        corner_kps = [cv2.KeyPoint(float(x), float(y), size=10) 
                     for y, x in zip(y_coords, x_coords)]
        
        # Detect edge features using FAST on M_psi
        # Re-using the same FastFeatureDetector is more efficient
        edge_detector = cv2.FastFeatureDetector_create(threshold=10)
        edge_kps = edge_detector.detect(M_psi_norm, None)
        
        # Ensure edge_kps is a list (not a tuple) for compatibility with the demo script
        edge_kps = list(edge_kps)
        
        return corner_kps, edge_kps
    
    def construct_mim(self, image, eo):
        """
        Construct Maximum Index Map (MIM) from log-Gabor convolution sequence.
        
        Args:
            image: Input image
            eo: Filter response from phase congruency
            
        Returns:
            mim: Maximum Index Map
            orientation_sums: Sums of filter responses for each orientation
        """
        # Get image dimensions more efficiently
        h, w = image.shape[:2] if len(image.shape) > 2 else image.shape
        
        # Pre-allocate lists based on known sizes
        convolution_sequence = []
        orientation_layers = [[] for _ in range(self.orientations)]
        
        # More efficient extraction of filter responses
        for s in range(self.scales):
            for o in range(self.orientations):
                # Use try/except for faster execution than multiple condition checks
                try:
                    if isinstance(eo[o], list):
                        response = eo[o][s]
                    else:
                        response = eo[s * self.orientations + o]
                    
                    # Get amplitude (magnitude) of complex response
                    amplitude = np.abs(response)
                    convolution_sequence.append(amplitude)
                    orientation_layers[o].append(amplitude)
                except (IndexError, TypeError):
                    # Skip if index is out of range or type error
                    continue
        
        # Sum up amplitudes for each orientation - vectorized operation
        orientation_sums = []
        for o in range(self.orientations):
            if orientation_layers[o]:
                # More efficient to use numpy sum directly on array stack
                orientation_sum = np.sum(np.stack(orientation_layers[o], axis=0), axis=0)
                orientation_sums.append(orientation_sum)
        
        # Initialize with zeros using efficient shape determination
        mim = np.zeros((h, w), dtype=np.uint8)
        
        if orientation_sums:
            # Stack orientation sums for efficient processing
            stacked_sums = np.stack(orientation_sums, axis=2)
            
            # Find index of maximum response along orientation dimension - vectorized
            max_indices = np.argmax(stacked_sums, axis=2)
            
            # Add 1 to make indices 1-based
            mim = max_indices + 1
        
        return mim, orientation_sums
    
    def create_rotated_mims(self, image, eo):
        """
        Create multiple MIMs with different initial layers for rotation invariance.
        
        Args:
            image: Input image
            eo: Filter response from phase congruency
            
        Returns:
            mims: List of MIMs with different initial layers
        """
        # Pre-allocate list for better memory efficiency
        mims = []
        
        # Create MIM with each possible initial layer
        for initial_layer in range(self.orientations):
            # Shift orientation layers
            shifted_eo = self._shift_orientation_layers(eo, initial_layer)
            
            # Construct MIM with shifted orientations
            mim, _ = self.construct_mim(image, shifted_eo)
            mims.append(mim)
        
        return mims
    
    def _shift_orientation_layers(self, eo, shift):
        """
        Shift orientation layers to simulate rotation.
        
        Args:
            eo: Filter response from phase congruency
            shift: Number of positions to shift
            
        Returns:
            shifted_eo: Shifted filter response
        """
        # Enhanced implementation of orientation shifting
        if isinstance(eo, list):
            if all(isinstance(item, list) for item in eo):
                # Handle nested list structure - shift the outer list
                shifted_eo = eo[shift:] + eo[:shift]
            else:
                # Handle flat list by reshaping
                shifted_eo = []
                for s in range(self.scales):
                    start_idx = s * self.orientations
                    oriented_slice = eo[start_idx:start_idx + self.orientations]
                    shifted_slice = oriented_slice[shift:] + oriented_slice[:shift]
                    shifted_eo.extend(shifted_slice)
            return shifted_eo
        return eo  # Return original if structure is not as expected
    
    @lru_cache(maxsize=32)
    def _get_cell_indices(self):
        """
        Cache the cell indices for descriptor computation.
        
        Returns:
            List of tuples with cell coordinates and sizes
        """
        cell_size = self.patch_size // 6
        return [(i * cell_size, j * cell_size, cell_size) 
                for i in range(6) for j in range(6)]
    
    def compute_descriptor(self, image, keypoints, mim):
        """
        Compute RIFT descriptors for keypoints.
        
        Args:
            image: Input image
            keypoints: List of keypoints
            mim: Maximum Index Map
            
        Returns:
            descriptors: RIFT descriptors for keypoints
        """
        # Handle empty keypoints list
        if not keypoints:
            return np.array([], dtype=np.float32).reshape(0, 6*6*self.orientations)
            
        h, w = image.shape[:2] if len(image.shape) > 2 else image.shape
        half_size = self.patch_size // 2
        
        # Pre-calculate cell indices for 6x6 grid
        cell_indices = self._get_cell_indices()
        
        # Pre-allocate descriptor array for better memory efficiency
        descriptors = np.zeros((len(keypoints), 6*6*self.orientations), dtype=np.float32)
        
        # Process each keypoint
        for idx, kp in enumerate(keypoints):
            x, y = int(kp.pt[0]), int(kp.pt[1])
            
            # Skip keypoints too close to image boundary
            if (x < half_size or y < half_size or 
                x >= w - half_size or y >= h - half_size):
                continue
            
            # Extract patch
            patch = mim[y-half_size:y+half_size, x-half_size:x+half_size]
            
            # Compute descriptor for the patch
            descriptor_idx = 0
            for cell_y, cell_x, cell_size in cell_indices:
                # Extract cell
                cell = patch[cell_y:cell_y+cell_size, cell_x:cell_x+cell_size]
                
                # Compute histogram vectorized
                hist = np.zeros(self.orientations, dtype=np.float32)
                for o in range(1, self.orientations + 1):
                    hist[o-1] = np.sum(cell == o)
                
                # Normalize histogram if sum is positive
                hist_sum = np.sum(hist)
                if hist_sum > 0:
                    hist = hist / hist_sum
                
                # Assign to descriptor array
                descriptors[idx, descriptor_idx:descriptor_idx+self.orientations] = hist
                descriptor_idx += self.orientations
        
        return descriptors
    
    def match_features(self, des1, des2, ratio_threshold=0.8):
        """
        Match feature descriptors between two images.
        
        Args:
            des1: Descriptors from first image
            des2: Descriptors from second image
            ratio_threshold: Lowe's ratio test threshold
            
        Returns:
            matches: List of matches
        """
        # Handle empty descriptor arrays
        if des1.size == 0 or des2.size == 0:
            return []
            
        # Ensure descriptors are in correct format (float32)
        des1 = des1.astype(np.float32) if des1.dtype != np.float32 else des1
        des2 = des2.astype(np.float32) if des2.dtype != np.float32 else des2
        
        try:
            # Use knnMatch for efficient ratio test
            matches = self.matcher.knnMatch(des1, des2, k=2)
            
            # Apply ratio test more efficiently with list comprehension
            good_matches = [m for m, n in matches if m.distance < ratio_threshold * n.distance]
            return good_matches
        except cv2.error:
            # Handle case when knnMatch fails (e.g., not enough matches)
            return []
            
    def detect_and_compute(self, image):
        """
        Detect features and compute descriptors for an image.
        
        Args:
            image: Input image
            
        Returns:
            keypoints: Detected keypoints
            descriptors: Feature descriptors
            mim: Maximum Index Map
        """
        # Compute phase congruency and moment maps
        pc, m_psi, M_psi, eo = self.compute_phase_congruency(image)
        
        # Detect features
        corner_kps, edge_kps = self.detect_features(m_psi, M_psi)
        
        # Combine corner and edge keypoints
        keypoints = corner_kps + edge_kps
        
        # Exit early if no keypoints detected
        if not keypoints:
            return [], np.array([]), None
        
        # Construct MIM
        mim, _ = self.construct_mim(image, eo)
        
        # For rotation invariance, create multiple MIMs
        # Only if needed - can be commented out if rotation invariance not required
        # mims = self.create_rotated_mims(image, eo)
        
        # Compute descriptors
        descriptors = self.compute_descriptor(image, keypoints, mim)
        
        return keypoints, descriptors, mim

    def process_image_pair(self, img1, img2):
        """
        Process a pair of images using RIFT.
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            matches: List of matches
            keypoints1: Keypoints from first image
            keypoints2: Keypoints from second image
        """
        # Detect and describe features in both images
        keypoints1, descriptors1, mim1 = self.detect_and_compute(img1)
        keypoints2, descriptors2, mim2 = self.detect_and_compute(img2)
        
        # Return early if no features detected
        if len(keypoints1) == 0 or len(keypoints2) == 0:
            return [], keypoints1, keypoints2
        
        # Match features
        matches = self.match_features(descriptors1, descriptors2)
        
        return matches, keypoints1, keypoints2