import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from phasecong3 import phasecong


def block_Harris(img, block_size=2, ksize=3, k=0.04, threshold=0.1):
    """
    Detect Harris corners in the image
    
    Args:
        img: Input image
        block_size: Size of the neighborhood for corner detection
        ksize: Size of the Sobel kernel
        k: Harris detector free parameter
        threshold: Threshold for corner detection
        
    Returns:
        coords: Corner coordinates as (row, col) pairs
    """
    # Convert to grayscale if the image is color
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Convert to float32 (required for cv2.cornerHarris)
    gray = np.float32(gray)
    
    # Calculate Harris response
    harris_response = cv2.cornerHarris(gray, block_size, ksize, k)
    
    # Dilate the response to highlight corners (optional)
    harris_response = cv2.dilate(harris_response, None)
    
    # Find coordinates where response exceeds threshold
    coords = np.argwhere(harris_response > threshold * harris_response.max())
    # Coordinates are returned as (row, column)
    
    return coords


def HOPC_des(gray, coords, NumberScales, NumberAngles, cell_size=8, bin_size=8):
    """
    Extract HOPC descriptors at specified coordinates
    
    Args:
        gray: Input grayscale image
        coords: Coordinates where to extract descriptors
        NumberScales: Number of scales for phase congruency
        NumberAngles: Number of angles for phase congruency
        cell_size: Size of cell for histogram computation
        bin_size: Number of bins in histogram
        
    Returns:
        HOPC_vector: List of HOPC descriptors
    """
    cell_gradient_vector = []
    
    for row, col in coords:
        # Extract 4 cells around the point
        cell1 = gray[int(row-cell_size):int(row), int(col-cell_size):int(col)]  
        cell2 = gray[int(row-cell_size):int(row), int(col):int(col+cell_size)]  
        cell3 = gray[int(row):int(row+cell_size), int(col-cell_size):int(col)]  
        cell4 = gray[int(row):int(row+cell_size), int(col):int(col+cell_size)]
        
        # Check if any cell is empty (outside image boundaries)
        if cell1.size == 0 or cell2.size == 0 or cell3.size == 0 or cell4.size == 0:
            continue
            
        # Compute phase congruency for each cell
        # Note: phasecong returns multiple values, we only need the first (magnitude) and fourth (angle)
        pc_result1 = phasecong(cell1, nscale=NumberScales, norient=NumberAngles)
        cell_magnitude1, _, _, cell_angle1, _, _, _ = pc_result1
        
        pc_result2 = phasecong(cell2, nscale=NumberScales, norient=NumberAngles)
        cell_magnitude2, _, _, cell_angle2, _, _, _ = pc_result2
        
        pc_result3 = phasecong(cell3, nscale=NumberScales, norient=NumberAngles)
        cell_magnitude3, _, _, cell_angle3, _, _, _ = pc_result3
        
        pc_result4 = phasecong(cell4, nscale=NumberScales, norient=NumberAngles)
        cell_magnitude4, _, _, cell_angle4, _, _, _ = pc_result4
        
        # Compute gradients for each cell
        cell_gradient_vector.append([
            cell_gradient(cell_magnitude1, cell_angle1, bin_size),
            cell_gradient(cell_magnitude2, cell_angle2, bin_size),
            cell_gradient(cell_magnitude3, cell_angle3, bin_size),
            cell_gradient(cell_magnitude4, cell_angle4, bin_size)
        ])
    
    # Reshape the gradient vector array
    if not cell_gradient_vector:  # Check if the list is empty
        return []
        
    cell_gradient_vector = np.array(cell_gradient_vector)
    cell_gradient_vector = np.reshape(cell_gradient_vector, (len(cell_gradient_vector), 4*bin_size))
    
    # Normalize the gradient vectors
    HOPC_vector = []
    for i in range(cell_gradient_vector.shape[0]):
        block_vector = []
        block_vector.extend(cell_gradient_vector[i])
        
        # Calculate magnitude
        mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
        magnitude = mag(block_vector)
        
        # Normalize if magnitude is not zero
        if magnitude != 0:
            normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
            block_vector = normalize(block_vector, magnitude)
            
        HOPC_vector.append(block_vector)
        
    return HOPC_vector


def cell_gradient(cell_magnitude, cell_angle, bin_size=8):
    """
    Compute gradient histogram for a cell
    
    Args:
        cell_magnitude: Cell magnitude from phase congruency
        cell_angle: Cell angle from phase congruency
        bin_size: Number of bins in histogram
        
    Returns:
        orientation_centers: Histogram of oriented gradients
    """
    angle_unit = 360 // bin_size
    orientation_centers = [0] * bin_size
    
    # Convert angles from radians to degrees if needed
    if np.max(cell_angle) <= 2*np.pi:
        cell_angle = cell_angle * 180.0 / np.pi
    
    # Ensure angles are positive and within [0, 360)
    cell_angle = cell_angle % 360.0
    
    for i in range(cell_magnitude.shape[0]):
        for j in range(cell_magnitude.shape[1]):
            gradient_strength = cell_magnitude[i][j]
            gradient_angle = cell_angle[i][j]
            
            # Get closest bins and interpolation factor
            min_angle, max_angle, mod = get_closest_bins(gradient_angle, bin_size)
            
            # Weight contribution to the two closest bins
            orientation_centers[min_angle] += (gradient_strength * (1 - (mod / angle_unit)))
            orientation_centers[max_angle] += (gradient_strength * (mod / angle_unit))
            
    return orientation_centers


def get_closest_bins(pc_angle, bin_size=8):
    """
    Find the two closest bins for interpolation
    
    Args:
        pc_angle: Phase congruency angle in degrees
        bin_size: Number of bins
        
    Returns:
        min_angle: Index of first bin
        max_angle: Index of second bin
        mod: Offset for interpolation
    """
    angle_unit = 360 // bin_size
    
    # Ensure angle is within [0, 360)
    pc_angle = pc_angle % 360.0
    
    # Calculate bin index and offset
    idx = int(pc_angle / angle_unit)
    mod = pc_angle % angle_unit
    
    # Handle edge case
    if idx == bin_size:
        return idx - 1, (idx) % bin_size, mod
        
    return idx, (idx + 1) % bin_size, mod 


class HOPC_descriptor():
    """
    Histogram of Phase Congruency (HOPC) descriptor for template matching
    """
    def __init__(self, img, cell_size=16, bin_size=8):
        """
        Initialize HOPC descriptor
        
        Args:
            img: Input image
            cell_size: Size of cell for histogram computation
            bin_size: Number of bins in histogram
        """
        self.img = img
        
        # Normalize image if not already normalized
        if np.max(img) > 1.0:
            self.img = np.sqrt(img / float(np.max(img)))
            self.img = self.img * 255
        
        self.cell_size = cell_size
        self.bin_size = bin_size
        self.angle_unit = 360 // self.bin_size
        self.NumberScales = 4
        self.NumberAngles = 6
        
        # Validate parameters
        assert type(self.bin_size) == int, "bin_size should be integer"
        assert type(self.cell_size) == int, "cell_size should be integer"
        assert type(self.angle_unit) == int, "bin_size should be divisible by 360"

    def extract(self):
        """
        Extract HOPC descriptors from the entire image
        
        Returns:
            hopc_vector: List of HOPC descriptors
            hopc_image: Visualization of HOPC features
        """
        height, width = self.img.shape
        
        # Compute phase congruency
        pc_result = phasecong(self.img, nscale=self.NumberScales, norient=self.NumberAngles)
        pc_magnitude, _, _, pc_angle, _, _, _ = pc_result
        
        # Take absolute magnitude
        pc_magnitude = abs(pc_magnitude)
        
        # Initialize cell vector array
        cell_pc_vector = np.zeros((height // self.cell_size, width // self.cell_size, self.bin_size))
        
        # Process each cell
        for i in range(cell_pc_vector.shape[0]):
            for j in range(cell_pc_vector.shape[1]):
                # Extract cell
                cell_magnitude = pc_magnitude[i * self.cell_size:(i + 1) * self.cell_size,
                                 j * self.cell_size:(j + 1) * self.cell_size]
                cell_angle = pc_angle[i * self.cell_size:(i + 1) * self.cell_size,
                             j * self.cell_size:(j + 1) * self.cell_size]
                
                # Compute cell histogram
                cell_pc_vector[i][j] = self.cell_pc(cell_magnitude, cell_angle)
        
        # Create visualization
        hopc_image = self.render_pc(np.zeros([height, width]), cell_pc_vector)
        
        # Compute block descriptors
        hopc_vector = []
        for i in range(cell_pc_vector.shape[0] - 1):
            for j in range(cell_pc_vector.shape[1] - 1):
                # Create block from 2x2 cells
                block_vector = []
                block_vector.extend(cell_pc_vector[i][j])
                block_vector.extend(cell_pc_vector[i][j + 1])
                block_vector.extend(cell_pc_vector[i + 1][j])
                block_vector.extend(cell_pc_vector[i + 1][j + 1])
                
                # Normalize block vector
                mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
                magnitude = mag(block_vector)
                
                if magnitude != 0:
                    normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                    block_vector = normalize(block_vector, magnitude)
                    
                hopc_vector.append(block_vector)
                
        return hopc_vector, hopc_image

    def cell_pc(self, cell_magnitude, cell_angle):
        """
        Compute histogram for a cell
        
        Args:
            cell_magnitude: Cell magnitude from phase congruency
            cell_angle: Cell angle from phase congruency
            
        Returns:
            orientation_centers: Histogram of oriented phase congruency
        """
        orientation_centers = [0] * self.bin_size
        
        # Convert angles from radians to degrees if needed
        if np.max(cell_angle) <= 2*np.pi:
            cell_angle = cell_angle * 180.0 / np.pi
            
        # Ensure angles are positive and within [0, 360)
        cell_angle = cell_angle % 360.0
        
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                gradient_strength = cell_magnitude[i][j]
                gradient_angle = cell_angle[i][j]
                
                # Get closest bins and interpolation factor
                min_angle, max_angle, mod = self.get_closest_bins(gradient_angle)
                
                # Weight contribution to the two closest bins
                orientation_centers[min_angle] += (gradient_strength * (1 - (mod / self.angle_unit)))
                orientation_centers[max_angle] += (gradient_strength * (mod / self.angle_unit))
                
        return orientation_centers

    def get_closest_bins(self, gradient_angle):
        """
        Find the two closest bins for interpolation
        
        Args:
            gradient_angle: Phase congruency angle in degrees
            
        Returns:
            min_angle: Index of first bin
            max_angle: Index of second bin
            mod: Offset for interpolation
        """
        # Ensure angle is within [0, 360)
        gradient_angle = gradient_angle % 360.0
        
        # Calculate bin index and offset
        idx = int(gradient_angle / self.angle_unit)
        mod = gradient_angle % self.angle_unit
        
        # Handle edge case
        if idx == self.bin_size:
            return idx - 1, (idx) % self.bin_size, mod
            
        return idx, (idx + 1) % self.bin_size, mod

    def render_pc(self, image, cell_gradient):
        """
        Visualize cells by drawing lines representing the histogram bins
        
        Args:
            image: Background image for visualization
            cell_gradient: Cell gradients to visualize
            
        Returns:
            image: Image with cell visualization
        """
        cell_width = self.cell_size / 2
        max_mag = np.array(cell_gradient).max()
        
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[x][y]
                
                # Normalize cell gradient
                if max_mag > 0:
                    cell_grad = cell_grad / max_mag
                
                angle = 0
                angle_gap = self.angle_unit
                
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    
                    # Calculate line endpoints
                    x1 = int(x * self.cell_size + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size - magnitude * cell_width * math.sin(angle_radian))
                    
                    # Draw line
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    
                    angle += angle_gap
                    
        return image