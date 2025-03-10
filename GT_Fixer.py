import scipy.io
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def process_mat_file(mat_file_path, output_dir=None):
    """
    Process a single MAT file to visualize landmarks and compute homography.

    Parameters:
    mat_file_path (str): Path to the .mat file
    output_dir (str, optional): Directory to save outputs. If None, uses same directory as input

    Returns:
    dict: Dictionary with processed data
    """
    print(f"Processing {mat_file_path}...")

    # Create output directory if it doesn't exist
    if output_dir is None:
        output_dir = os.path.dirname(mat_file_path)
    os.makedirs(output_dir, exist_ok=True)

    # Get base filename without extension
    base_filename = os.path.basename(mat_file_path).rsplit('.', 1)[0]

    # Load the .mat file
    mat_data = scipy.io.loadmat(mat_file_path)

    # Extract images and landmarks
    I_fix = mat_data['I_fix']
    I_move = mat_data['I_move']
    landmarks = mat_data['Landmarks']
    T = mat_data['T'] if 'T' in mat_data else None

    # Extract landmark sets
    landmarks_fix = landmarks[0][0][0]
    landmarks_mov = landmarks[0][0][1]

    # Convert images to RGB if they are grayscale
    if len(I_fix.shape) == 2:
        I_fix_rgb = cv2.cvtColor(I_fix.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    else:
        I_fix_rgb = I_fix.copy()

    if len(I_move.shape) == 2:
        I_move_rgb = cv2.cvtColor(I_move.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    else:
        I_move_rgb = I_move.copy()

    # Colors for points and lines
    color_point_fix = (255, 0, 0)  # Blue for fixed points
    color_point_mov = (0, 0, 255)  # Red for moving points
    color_line = (0, 255, 0)       # Green for lines

    # Draw landmarks on fixed image
    I_fix_with_landmarks = I_fix_rgb.copy()
    for i, (x, y) in enumerate(landmarks_fix):
        cv2.circle(I_fix_with_landmarks, (int(x), int(y)), 5, color_point_fix, -1)
        cv2.putText(I_fix_with_landmarks, str(i), (int(x) + 5, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw landmarks on moving image
    I_move_with_landmarks = I_move_rgb.copy()
    for i, (x, y) in enumerate(landmarks_mov):
        cv2.circle(I_move_with_landmarks, (int(x), int(y)), 5, color_point_mov, -1)
        cv2.putText(I_move_with_landmarks, str(i), (int(x) + 5, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Create combined image
    height = max(I_fix_rgb.shape[0], I_move_rgb.shape[0])
    width = max(I_fix_rgb.shape[1], I_move_rgb.shape[1])

    I_combined = np.zeros((height, width, 3), dtype=np.uint8)

    I_combined[:I_fix_rgb.shape[0], :I_fix_rgb.shape[1]] = I_fix_rgb
    I_combined[:I_move_rgb.shape[0], :I_move_rgb.shape[1]] = I_move_rgb * 0.5  # Overlay with transparency

    # Draw correspondences between landmarks
    for i, ((x1, y1), (x2, y2)) in enumerate(zip(landmarks_fix, landmarks_mov)):
        cv2.line(I_combined, (int(x1), int(y1)), (int(x2), int(y2)), color_line, 1)
        cv2.circle(I_combined, (int(x1), int(y1)), 3, color_point_fix, -1)
        cv2.circle(I_combined, (int(x2), int(y2)), 3, color_point_mov, -1)

    # Compute homography
    H, mask = cv2.findHomography(
        landmarks_fix,  # Source points
        landmarks_mov,  # Destination points
        method=cv2.RANSAC,
        ransacReprojThreshold=5
    )

    # Apply transformation
    I_move_warped = cv2.warpPerspective(I_fix, H, (I_move.shape[1], I_move.shape[0]))

    # Visualize images
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

    ax[0, 0].imshow(I_fix_with_landmarks)
    ax[0, 0].set_title('SAR Image with Landmarks')
    ax[0, 0].axis('off')

    ax[0, 1].imshow(I_move_with_landmarks)
    ax[0, 1].set_title('OPT Image with Landmarks')
    ax[0, 1].axis('off')

    ax[1, 0].imshow(I_combined)
    ax[1, 0].set_title('Correspondences between Landmarks')
    ax[1, 0].axis('off')

    ax[1, 1].imshow(I_move_warped, cmap='gray')
    ax[1, 1].set_title('Transformed SAR Image')
    ax[1, 1].axis('off')

    plt.tight_layout()

    # Save figure
    figure_path = os.path.join(output_dir, f"{base_filename}_visualization.png")
    plt.savefig(figure_path)
    print(f"Saved visualization to {figure_path}")

    # Save transformed data to new mat file
    output_dict = {
        'I_fix': I_fix,
        'I_move': I_move,
        'Landmarks': landmarks,
        'T': H
    }

    output_mat_path = os.path.join(output_dir, f"{base_filename}.mat")
    scipy.io.savemat(output_mat_path, output_dict)
    print(f"Saved processed data to {output_mat_path}")

    plt.close(fig)  # Close the figure to free memory

    return output_dict


def process_all_mat_files(input_dir, output_dir=None, pattern="*.mat"):
    """
    Process all .mat files in a directory.

    Parameters:
    input_dir (str): Directory containing .mat files
    output_dir (str, optional): Directory to save outputs. If None, uses input_dir
    pattern (str, optional): Glob pattern to match files, defaults to "*.mat"
    """
    # Get all .mat files in the directory
    mat_files = glob.glob(os.path.join(input_dir, pattern))

    if not mat_files:
        print(f"No .mat files found in {input_dir} matching pattern {pattern}")
        return

    print(f"Found {len(mat_files)} .mat files to process")

    # Process each file
    results = {}
    for mat_file in mat_files:
        try:
            result = process_mat_file(mat_file, output_dir)
            results[mat_file] = "Success"
        except Exception as e:
            print(f"Error processing {mat_file}: {str(e)}")
            results[mat_file] = f"Error: {str(e)}"

    # Print summary
    print("\nProcessing Summary:")
    for file, status in results.items():
        print(f"{os.path.basename(file)}: {status}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process all .mat files in a directory")
    parser.add_argument("input_dir", type=str, help="Directory containing .mat files")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save outputs (default: same as input)")
    parser.add_argument("--pattern", type=str, default="*.mat",
                        help="File pattern to match (default: *.mat)")

    args = parser.parse_args()

    process_all_mat_files(args.input_dir, args.output_dir, args.pattern)