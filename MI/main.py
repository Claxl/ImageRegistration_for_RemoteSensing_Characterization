# -*- coding: utf-8 -*-
"""
Image Registration module based on mutual information.

__author__ = Xinzhe Luo, [Your Name]

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from tqdm import tqdm

# Import the MI module
from MI import MI


class AffineTransformer(nn.Module):
    """
    Affine transformation module for image registration.
    
    For 2D: 6 parameters [a, b, tx, c, d, ty] representing:
        [a b tx]
        [c d ty]
        [0 0  1]
    
    For 3D: 12 parameters [a, b, c, tx, d, e, f, ty, g, h, i, tz] representing:
        [a b c tx]
        [d e f ty]
        [g h i tz]
        [0 0 0  1]
    """
    def __init__(self, dimension):
        super(AffineTransformer, self).__init__()
        self.dimension = dimension
        
        # Initialize parameters with identity transformation
        if dimension == 2:
            # 2D affine has 6 parameters: [a, b, tx, c, d, ty]
            self.params = nn.Parameter(torch.tensor([1., 0., 0., 0., 1., 0.], dtype=torch.float32))
        elif dimension == 3:
            # 3D affine has 12 parameters: [a, b, c, tx, d, e, f, ty, g, h, i, tz]
            self.params = nn.Parameter(torch.tensor([1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.], dtype=torch.float32))
        else:
            raise NotImplementedError(f"Dimension {dimension} not supported")
    
    def get_affine_matrix(self, batch_size=1):
        """
        Convert parameters to affine transformation matrix.
        
        Returns:
            torch.Tensor: Affine matrix of shape [B, D+1, D+1]
        """
        if self.dimension == 2:
            a, b, tx, c, d, ty = self.params
            
            # Create batch of transformation matrices
            matrix = torch.zeros(batch_size, 3, 3, device=self.params.device)
            matrix[:, 0, 0] = a
            matrix[:, 0, 1] = b
            matrix[:, 0, 2] = tx
            matrix[:, 1, 0] = c
            matrix[:, 1, 1] = d
            matrix[:, 1, 2] = ty
            matrix[:, 2, 2] = 1.0
            
            return matrix
        else:  # 3D
            a, b, c, tx, d, e, f, ty, g, h, i, tz = self.params
            
            # Create batch of transformation matrices
            matrix = torch.zeros(batch_size, 4, 4, device=self.params.device)
            matrix[:, 0, 0] = a
            matrix[:, 0, 1] = b
            matrix[:, 0, 2] = c
            matrix[:, 0, 3] = tx
            matrix[:, 1, 0] = d
            matrix[:, 1, 1] = e
            matrix[:, 1, 2] = f
            matrix[:, 1, 3] = ty
            matrix[:, 2, 0] = g
            matrix[:, 2, 1] = h
            matrix[:, 2, 2] = i
            matrix[:, 2, 3] = tz
            matrix[:, 3, 3] = 1.0
            
            return matrix
    
    def forward(self, source):
        """
        Apply affine transformation to source image.
        
        Args:
            source: torch.Tensor of shape [B, C, *vol_shape]
        
        Returns:
            torch.Tensor: Transformed image
        """
        batch_size = source.shape[0]
        
        # Move params to same device as source if needed
        if self.params.device != source.device:
            self.params = nn.Parameter(self.params.to(source.device))
        
        # Get the affine matrix
        if self.dimension == 2:
            h, w = source.shape[2:]
            # For 2D affine transformations, we need a 2x3 matrix
            affine_matrix = self.get_affine_matrix(batch_size)[:, :2, :3]
            # Create the sampling grid
            grid = F.affine_grid(
                affine_matrix, 
                [batch_size, 1, h, w],
                align_corners=True
            )
        else:  # 3D
            d, h, w = source.shape[2:]
            # For 3D affine transformations, we need a 3x4 matrix
            affine_matrix = self.get_affine_matrix(batch_size)[:, :3, :4]
            # Create the sampling grid
            grid = F.affine_grid(
                affine_matrix, 
                [batch_size, 1, d, h, w],
                align_corners=True
            )
        
        # Apply grid sample
        return F.grid_sample(source, grid, align_corners=True, mode='bilinear')


class DeformableTransformer(nn.Module):
    """
    Deformable transformation module using displacement field.
    """
    def __init__(self, img_shape, dimension):
        super(DeformableTransformer, self).__init__()
        self.dimension = dimension
        self.img_shape = img_shape
        
        # Initialize displacement field with zeros (identity transformation)
        if dimension == 2:
            h, w = img_shape
            self.displacement = nn.Parameter(torch.zeros(1, 2, h, w, dtype=torch.float32))
        elif dimension == 3:
            d, h, w = img_shape
            self.displacement = nn.Parameter(torch.zeros(1, 3, d, h, w, dtype=torch.float32))
        else:
            raise NotImplementedError(f"Dimension {dimension} not supported")
    
    def forward(self, source):
        """
        Apply deformable transformation to source image.
        
        Args:
            source: torch.Tensor of shape [B, C, *vol_shape]
        
        Returns:
            torch.Tensor: Transformed image
        """
        batch_size = source.shape[0]
        
        # Move displacement to same device as source if needed
        if self.displacement.device != source.device:
            self.displacement = nn.Parameter(self.displacement.to(source.device))
        
        # Create identity grid
        if self.dimension == 2:
            h, w = source.shape[2:]
            # Create identity affine matrix on the same device as source
            identity = torch.eye(2, 3, device=source.device).unsqueeze(0).repeat(batch_size, 1, 1)
            # Create the grid
            grid = F.affine_grid(
                identity, 
                [batch_size, 1, h, w],
                align_corners=True
            )
            
            # Prepare displacement field
            disp = self.displacement.repeat(batch_size, 1, 1, 1)
            disp = disp.permute(0, 2, 3, 1)  # [B, H, W, 2]
            
        else:  # 3D
            d, h, w = source.shape[2:]
            # Create identity affine matrix on the same device as source
            identity = torch.eye(3, 4, device=source.device).unsqueeze(0).repeat(batch_size, 1, 1)
            # Create the grid
            grid = F.affine_grid(
                identity, 
                [batch_size, 1, d, h, w],
                align_corners=True
            )
            
            # Prepare displacement field
            disp = self.displacement.repeat(batch_size, 1, 1, 1, 1)
            disp = disp.permute(0, 2, 3, 4, 1)  # [B, D, H, W, 3]
        
        # Apply grid sample with deformed grid
        return F.grid_sample(source, grid + disp, align_corners=True, mode='bilinear')


class ImageRegistration(nn.Module):
    """
    Image registration module using mutual information.
    """
    def __init__(self, dimension, transform_type='affine', img_shape=None, device=None, **kwargs):
        """
        Initialize the image registration module.
        
        Args:
            dimension: int, 2 for 2D images, 3 for 3D volumes
            transform_type: str, 'affine' or 'deformable'
            img_shape: tuple, image shape (required for deformable)
            device: str, device to use ('cuda' or 'cpu')
            **kwargs: additional arguments for MI module
        """
        super(ImageRegistration, self).__init__()
        self.dimension = dimension
        self.transform_type = transform_type
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            if self.device == 'cuda' and not torch.cuda.is_available():
                print("CUDA is not available, falling back to CPU")
                self.device = 'cpu'
        
        # Initialize MI module
        self.mi_module = MI(dimension=dimension, **kwargs)
        
        # Initialize transformation module
        if transform_type == 'affine':
            self.transformer = AffineTransformer(dimension=dimension)
        elif transform_type == 'deformable':
            assert img_shape is not None, "img_shape is required for deformable transformation"
            self.transformer = DeformableTransformer(img_shape=img_shape, dimension=dimension)
        else:
            raise ValueError(f"Unknown transform_type: {transform_type}")
        
        # Move modules to the correct device
        self.to(self.device)
    
    def forward(self, source, target, mask=None):
        """
        Forward pass: transform source and compute MI with target.
        
        Args:
            source: torch.Tensor of shape [B, 1, *vol_shape]
            target: torch.Tensor of shape [B, 1, *vol_shape]
            mask: torch.Tensor of shape [B, 1, *vol_shape] or None
        
        Returns:
            transformed_source: torch.Tensor, transformed source image
            mi_value: torch.Tensor, mutual information value
        """
        # Transform source image
        transformed_source = self.transformer(source)
        
        # Compute MI loss (negative since we want to maximize MI)
        mi_value = self.mi_module.mi(transformed_source, target, mask)
        
        return transformed_source, mi_value
    
    def register(self, source, target, mask=None, num_iterations=100, learning_rate=0.01, 
                optimizer_type='adam', verbose=True):
        """
        Register source image to target image.
        
        Args:
            source: torch.Tensor of shape [B, 1, *vol_shape]
            target: torch.Tensor of shape [B, 1, *vol_shape]
            mask: torch.Tensor of shape [B, 1, *vol_shape] or None
            num_iterations: int, number of optimization iterations
            learning_rate: float, learning rate for optimizer
            optimizer_type: str, 'adam' or 'sgd'
            verbose: bool, whether to print progress
        
        Returns:
            dict containing:
                'transformed_source': transformed source image
                'transformation': transformation parameters
                'mi_values': list of MI values during optimization
        """
        # Setup optimizer
        if optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam(self.transformer.parameters(), lr=learning_rate)
        elif optimizer_type.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.transformer.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer_type: {optimizer_type}")
        
        # For tracking progress
        mi_values = []
        
        # Main optimization loop
        iterator = tqdm(range(num_iterations)) if verbose else range(num_iterations)
        for i in iterator:
            optimizer.zero_grad()
            
            # Forward pass
            transformed_source, mi_value = self(source, target, mask)
            
            # Loss is negative MI (since we want to maximize MI)
            loss = -mi_value
            
            # Backward pass and optimization step
            loss.backward()
            optimizer.step()
            
            # Record MI value
            mi_values.append(mi_value.item())
            
            # Update progress bar
            if verbose:
                iterator.set_description(f"Iteration {i+1}/{num_iterations}, MI: {mi_value.item():.6f}")
        
        # Final forward pass to get transformed image
        with torch.no_grad():
            transformed_source, final_mi = self(source, target, mask)
        
        # Return results
        results = {
            'transformed_source': transformed_source,
            'final_mi': final_mi.item(),
            'mi_values': mi_values
        }
        
        # Include transformation parameters
        if self.transform_type == 'affine':
            affine_matrix = self.transformer.get_affine_matrix(source.shape[0])
            results['affine_matrix'] = affine_matrix
            results['affine_params'] = self.transformer.params.detach().clone()
            
            # Print affine transformation parameters
            print("\nAffine Transformation Parameters:")
            if self.dimension == 2:
                param_names = ['a', 'b', 'tx', 'c', 'd', 'ty']
                for name, value in zip(param_names, self.transformer.params.detach().cpu().numpy()):
                    print(f"{name}: {value:.6f}")
                    
                print("\nAffine Transformation Matrix:")
                for i in range(3):
                    row = affine_matrix[0, i, :].detach().cpu().numpy()
                    print(f"  [{', '.join([f'{x:.6f}' for x in row])}]")
            else:  # 3D
                param_names = ['a', 'b', 'c', 'tx', 'd', 'e', 'f', 'ty', 'g', 'h', 'i', 'tz']
                for name, value in zip(param_names, self.transformer.params.detach().cpu().numpy()):
                    print(f"{name}: {value:.6f}")
                    
                print("\nAffine Transformation Matrix:")
                for i in range(4):
                    row = affine_matrix[0, i, :].detach().cpu().numpy()
                    print(f"  [{', '.join([f'{x:.6f}' for x in row])}]")
        else:  # deformable
            displacement_field = self.transformer.displacement.detach().clone()
            results['displacement_field'] = displacement_field
            
            # Print displacement field statistics
            disp_stats = {
                'mean': displacement_field.mean().item(),
                'std': displacement_field.std().item(),
                'min': displacement_field.min().item(),
                'max': displacement_field.max().item()
            }
            print("\nDisplacement Field Statistics:")
            for stat, value in disp_stats.items():
                print(f"{stat}: {value:.6f}")
        
        return results


# Utility functions
def load_and_preprocess_images(source_path, target_path, dimension=2, device='cuda'):
    """
    Load and preprocess images for registration.
    
    Args:
        source_path: str, path to source image
        target_path: str, path to target image
        dimension: int, 2 for 2D, 3 for 3D
        device: str, 'cuda' or 'cpu'
    
    Returns:
        source, target: preprocessed torch tensors
    """
    # Check if CUDA is available when device is set to 'cuda'
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU")
        device = 'cpu'
    try:
        import nibabel as nib
        import SimpleITK as sitk
        from skimage import io
        import numpy as np
    except ImportError:
        raise ImportError("Please install required packages: nibabel, SimpleITK, scikit-image")
    
    # Determine file type and load accordingly
    if source_path.endswith(('.nii', '.nii.gz')):  # NIfTI files (typically 3D)
        source_np = nib.load(source_path).get_fdata()
        target_np = nib.load(target_path).get_fdata()
    elif source_path.endswith(('.dcm')):  # DICOM files
        source_np = sitk.GetArrayFromImage(sitk.ReadImage(source_path))
        target_np = sitk.GetArrayFromImage(sitk.ReadImage(target_path))
    else:  # Assume regular image formats (PNG, JPG, etc. - typically 2D)
        source_np = io.imread(source_path)
        target_np = io.imread(target_path)
        
        # Convert RGB to grayscale if needed
        if len(source_np.shape) > 2 and source_np.shape[2] in [3, 4]:
            source_np = np.mean(source_np[:, :, :3], axis=2)
        if len(target_np.shape) > 2 and target_np.shape[2] in [3, 4]:
            target_np = np.mean(target_np[:, :, :3], axis=2)
    
    # Normalize intensity to [0, 1]
    source_np = (source_np - source_np.min()) / (source_np.max() - source_np.min() + 1e-8)
    target_np = (target_np - target_np.min()) / (target_np.max() - target_np.min() + 1e-8)
    
    # Convert to torch tensors with appropriate dimensions
    if dimension == 2:
        source = torch.from_numpy(source_np).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        target = torch.from_numpy(target_np).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    else:  # 3D
        source = torch.from_numpy(source_np).float().unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        target = torch.from_numpy(target_np).float().unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
    
    return source.to(device), target.to(device)


def visualize_registration_results(source, target, transformed_source, mi_values=None, output_prefix="registration_result"):
    """
    Visualize registration results and save them to files.
    
    Args:
        source: torch.Tensor, source image
        target: torch.Tensor, target image
        transformed_source: torch.Tensor, transformed source image
        mi_values: list, mutual information values during optimization
        output_prefix: str, prefix for output filenames
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        raise ImportError("Please install matplotlib")
    
    # Move tensors to CPU and convert to numpy
    source = source.squeeze().cpu().numpy()
    target = target.squeeze().cpu().numpy()
    transformed_source = transformed_source.squeeze().detach().cpu().numpy()
    
    # Create figure
    if len(source.shape) == 2:  # 2D
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Source image
        axes[0].imshow(source, cmap='gray')
        axes[0].set_title('Source')
        axes[0].axis('off')
        
        # Target image
        axes[1].imshow(target, cmap='gray')
        axes[1].set_title('Target')
        axes[1].axis('off')
        
        # Transformed source
        axes[2].imshow(transformed_source, cmap='gray')
        axes[2].set_title('Registered Source')
        axes[2].axis('off')
        
    else:  # 3D - show middle slice
        mid_slice = source.shape[0] // 2
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Source image
        axes[0].imshow(source[mid_slice], cmap='gray')
        axes[0].set_title(f'Source (Slice {mid_slice})')
        axes[0].axis('off')
        
        # Target image
        axes[1].imshow(target[mid_slice], cmap='gray')
        axes[1].set_title(f'Target (Slice {mid_slice})')
        axes[1].axis('off')
        
        # Transformed source
        axes[2].imshow(transformed_source[mid_slice], cmap='gray')
        axes[2].set_title(f'Registered Source (Slice {mid_slice})')
        axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save registration comparison figure
    plt.savefig(f"{output_prefix}_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved registration comparison to {output_prefix}_comparison.png")
    
    # Plot MI values if provided
    if mi_values is not None:
        plt.figure(figsize=(8, 4))
        plt.plot(mi_values)
        plt.xlabel('Iteration')
        plt.ylabel('Mutual Information')
        plt.title('Optimization Progress')
        plt.grid(True)
        
        # Save MI progress figure
        plt.savefig(f"{output_prefix}_mi_progress.png", dpi=300, bbox_inches='tight')
        print(f"Saved MI progress to {output_prefix}_mi_progress.png")
    
    # Close all figures to free memory
    plt.close('all')


# Example usage:

# 2D Registration Example
source, target = load_and_preprocess_images('SO5a.png', 'SO5b.png', dimension=2)

# Create registration module
reg_module = ImageRegistration(
    dimension=2, 
    transform_type='affine',
    num_bins=256,
    normalized=True
)

# Register images
results = reg_module.register(
    source, 
    target, 
    num_iterations=250,
    learning_rate=0.01
)

# Print transformation matrix (can be accessed directly too)
if reg_module.transform_type == 'affine':
    affine_matrix = results['affine_matrix']
    affine_params = results['affine_params']
    print("Final Affine Matrix:")
    print(affine_matrix[0])  # First batch item
else:
    disp_field = results['displacement_field']
    print("Displacement Field Shape:", disp_field.shape)
    print("Mean Displacement:", disp_field.abs().mean().item())

# Visualize results
visualize_registration_results(
    source, 
    target, 
    results['transformed_source'], 
    results['mi_values']
)
