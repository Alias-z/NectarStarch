"""Module providing core functions"""
# pylint: disable=line-too-long, multiple-statements, c-extension-no-member, no-member, no-name-in-module, relative-beyond-top-level, wildcard-import
import os  # interact with the operating system
import glob  # Unix style pathname pattern expansion
import json  # manipulate json files
import cv2  # OpenCV
import numpy as np  # NumPy
import matplotlib.pyplot as plt  # Matplotlib
import pandas as pd  # Add to imports at top

image_types = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif', '.ico', '.jfif', '.webp']  # supported image types


def get_json_paths(json_dir: str) -> list:
    """Under a given folder, get all json files paths"""
    return glob.glob(os.path.join(json_dir, '*.json'))


def get_subfolder_paths(input_dir: str) -> list:
    """Get paths of lowest level subfolders or folders without subfolders.

    For example, if structure is:
    input_dir/
        folder1/
            subfolder1/
            subfolder2/
        folder2/

    Returns: [input_dir/folder1/subfolder1, input_dir/folder1/subfolder2, input_dir/folder2]

    Args:
        input_dir (str): Directory to search for subfolders

    Returns:
        list: List of paths to lowest level or leaf folders
    """
    result = []
    for root, dirs, _ in os.walk(input_dir):
        if not dirs:  # if this folder has no subfolders
            if root != input_dir:  # exclude the input_dir itself
                result.append(root)
    return result


class NectarStarch:
    """Class for analyzing nectar starch content in microscopy images.

    Args:
        input_dir (str): Directory containing image files and their corresponding JSON annotations
    """
    def __init__(self, input_dir: str):
        self.input_dir = input_dir
        self.image_paths = []
        self.json_paths = []
        self.results_data = []  # Add new attribute to store results

    def load_files(self) -> None:
        """Load all valid image and JSON file pairs"""
        # First collect all potential image paths
        image_paths = []
        for extension in image_types:
            image_paths.extend(glob.glob(os.path.join(self.input_dir, f'*{extension}')))

        # Create matched pairs of image and JSON paths
        self.image_paths = []
        self.json_paths = []

        for img_path in image_paths:
            json_path = os.path.splitext(img_path)[0] + '.json'
            if os.path.exists(json_path):
                self.image_paths.append(img_path)
                self.json_paths.append(json_path)

        return None

    def __len__(self) -> int:
        return len(self.image_paths)

    def get_image_annotation_pair(self, index: int) -> tuple[np.ndarray, dict]:
        """Get inverted grayscale image and its corresponding annotation at given index
        Inverted so that darker pixels (more starch) have higher values
        """
        if index >= len(self.image_paths):
            raise IndexError('Index out of range')

        image = cv2.imread(self.image_paths[index], cv2.IMREAD_GRAYSCALE)
        image = 255 - image  # Invert the image
        with open(self.json_paths[index], 'r', encoding='utf-8') as file:
            annotation = json.load(file)
        return image, annotation

    def find_reference_point(self, image: np.ndarray) -> tuple[float, tuple[int, int]]:
        """Find darkest non-black point in the original image (highest in inverted).

        Args:
            image (np.ndarray): Inverted grayscale image

        Returns:
            tuple[float, tuple[int, int]]: Tuple containing:
                - reference value (float): Intensity value at reference point
                - position (tuple[int, int]): (y, x) coordinates of reference point
        """
        height, width = image.shape

        # Create region of interest excluding bottom 20% and edges
        roi = image[:int(height * 0.8), :]  # Exclude bottom 20%

        # Create mask excluding very dark and very bright pixels
        valid_pixels = (roi > 20) & (roi < 250)  # More strict thresholding

        if np.any(valid_pixels):
            # Find the position of maximum value in the valid region
            max_pos = np.unravel_index(np.argmax(roi * valid_pixels), roi.shape)
            ref_value = float(roi[max_pos])
            return ref_value, max_pos
        return 255.0, (0, 0)  # fallback values

    def compute_nectar_starch(self, image: np.ndarray, annotation: dict) -> dict:
        """Compute nectar starch content for segmented regions.

        Args:
            image (np.ndarray): Inverted grayscale image
            annotation (dict): Image annotation containing segmentation data

        Returns:
            dict: Dictionary containing for each region:
                - raw_value (float): Average intensity in region
                - normalized_value (float): Intensity normalized by reference point
        """
        info = annotation.get('info', {})
        width = info.get('width', None)
        height = info.get('height', None)
        objects = annotation.get('objects', [])
        results = {}

        # Find reference point
        ref_value, ref_pos = self.find_reference_point(image)
        annotation['reference_point'] = {'value': ref_value, 'position': ref_pos}

        for obj in objects:
            segmentation = obj['segmentation']
            points = np.array(segmentation, dtype=np.int32)
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [points], 1)
            avg_value = float(np.mean(image[mask == 1]))
            normalized_value = avg_value / ref_value if ref_value > 0 else 0

            obj_id = obj.get('id', len(results))
            results[obj_id] = {
                'raw_value': avg_value,
                'normalized_value': normalized_value
            }
            obj['avg_gray_value'] = avg_value
            obj['normalized_value'] = normalized_value

        # Store results for Excel export
        image_name = annotation.get('info', {}).get('name', 'unknown')
        for obj_id, values in results.items():
            self.results_data.append({
                'Image': image_name,
                'Region': obj_id + 1,  # 1-based region numbering
                'Raw_Value': values['raw_value'],
                'Normalized_Value': values['normalized_value'],
                'Reference_Value': ref_value
            })

        return results

    def export_to_excel(self, output_path: str) -> None:
        """Export all results to Excel file.

        Args:
            output_path (str): Path to save the Excel file

        Returns:
            None: Saves Excel file to specified path
        """
        if not self.results_data:
            print("No results to export")
            return

        df = pd.DataFrame(self.results_data)
        df.to_excel(output_path, index=False)
        print(f"Results exported to: {output_path}")

    def visualize_segmentation(self, image: np.ndarray, annotation: dict, save_path: str = None, display_name: str = None) -> None:
        """Visualize image with region masks and reference point.

        Args:
            image (np.ndarray): Inverted grayscale image
            annotation (dict): Image annotation containing segmentation data
            save_path (str, optional): Path to save visualization. Defaults to None.
            display_name (str, optional): Name to display in print output. Defaults to None.

        Returns:
            None: Displays visualization and optionally saves to file
        """
        info = annotation.get('info', {})
        width = info.get('width', None)
        height = info.get('height', None)

        # Print image info and values using display_name if provided
        print(f"\nImage: {display_name if display_name else info.get('name', 'unknown')}")
        print(f"Reference value: {annotation['reference_point']['value']:.1f}")
        objects = annotation.get('objects', [])
        for i, obj in enumerate(objects, 1):
            if 'avg_gray_value' in obj and 'normalized_value' in obj:
                print(f"Region {i}: raw={obj['avg_gray_value']:.1f}, normalized={obj['normalized_value']:.2f}")

        # Create RGB image from grayscale
        rgb_image = np.stack((image,) * 3, axis=-1)
        overlay_mask = np.zeros_like(rgb_image)

        # Draw masks in red
        for obj in objects:
            points = np.array(obj['segmentation'], dtype=np.int32)
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [points], 1)
            overlay_mask[mask == 1] = [255, 0, 0]

        # Draw reference point in blue with larger size and full opacity
        ref_pos = annotation['reference_point']['position']
        # Draw larger blue circle with no transparency
        cv2.circle(rgb_image, (ref_pos[1], ref_pos[0]), 15, (0, 0, 255), -1)  # Solid blue circle
        cv2.circle(rgb_image, (ref_pos[1], ref_pos[0]), 15, (255, 255, 255), 2)  # White border

        # Create the final overlay with masks
        overlay = cv2.addWeighted(rgb_image, 1, overlay_mask, 0.3, 0)

        # Display only the overlay
        plt.figure(figsize=(8, 8))
        plt.imshow(overlay)
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.show()
