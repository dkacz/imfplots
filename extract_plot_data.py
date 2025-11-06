#!/usr/bin/env python3
"""
Plot Data Extractor
Extracts numeric data from plot images using image processing techniques.
Supports bar charts, scatter plots, and combined visualizations.
"""

import cv2
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse


class PlotDataExtractor:
    """Extract numeric data from plot images."""

    def __init__(self, image_path: str):
        """Initialize extractor with image path."""
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.height, self.width = self.image.shape[:2]

    def detect_plot_area(self) -> Tuple[int, int, int, int]:
        """
        Detect the plot area boundaries.
        Returns: (x_min, y_min, x_max, y_max)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Find edges
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Look for the largest rectangular contour (likely the plot frame)
        # For now, use conservative estimates based on typical plot layouts
        margin_left = int(self.width * 0.1)
        margin_right = int(self.width * 0.95)
        margin_top = int(self.height * 0.1)
        margin_bottom = int(self.height * 0.85)

        return margin_left, margin_top, margin_right, margin_bottom

    def detect_colors(self, n_colors: int = 5) -> List[Tuple[int, int, int]]:
        """
        Detect dominant colors in the plot (excluding white/gray background).
        Returns: List of RGB colors
        """
        # Reshape image to list of pixels
        pixels = self.image_rgb.reshape(-1, 3)

        # Filter out near-white and near-gray pixels
        mask = ~((pixels[:, 0] > 200) & (pixels[:, 1] > 200) & (pixels[:, 2] > 200))
        mask &= ~((np.abs(pixels[:, 0] - pixels[:, 1]) < 20) &
                  (np.abs(pixels[:, 1] - pixels[:, 2]) < 20) &
                  (pixels[:, 0] > 150))

        filtered_pixels = pixels[mask]

        # Use k-means clustering to find dominant colors
        from sklearn.cluster import KMeans

        if len(filtered_pixels) > 0:
            kmeans = KMeans(n_clusters=min(n_colors, len(filtered_pixels)), random_state=42, n_init=10)
            kmeans.fit(filtered_pixels)
            colors = kmeans.cluster_centers_.astype(int)
            return [tuple(color) for color in colors]

        return []

    def extract_bar_chart_data(self, plot_area: Tuple[int, int, int, int],
                               colors: List[Tuple[int, int, int]]) -> Dict:
        """
        Extract data from bar charts.
        Returns: Dictionary with series data
        """
        x_min, y_min, x_max, y_max = plot_area
        plot_height = y_max - y_min
        plot_width = x_max - x_min

        # Extract vertical bars
        data = {}

        for color_idx, color in enumerate(colors):
            series_data = []
            color_name = f"series_{color_idx}"

            # Create color mask
            color_array = np.array(color)
            color_tolerance = 30

            lower = np.clip(color_array - color_tolerance, 0, 255)
            upper = np.clip(color_array + color_tolerance, 0, 255)

            mask = cv2.inRange(self.image_rgb, lower, upper)

            # Find vertical bars
            # Scan horizontally to find bar positions
            x_positions = []
            bar_heights = []

            # Divide into vertical slices
            n_slices = 100
            slice_width = plot_width // n_slices

            for i in range(n_slices):
                x = x_min + i * slice_width
                if x >= x_max:
                    break

                # Check vertical line for colored pixels
                column = mask[y_min:y_max, x:min(x + slice_width, x_max)]

                if np.any(column):
                    # Find the topmost and bottommost colored pixels
                    colored_rows = np.where(np.any(column, axis=1))[0]
                    if len(colored_rows) > 0:
                        top = colored_rows[0]
                        bottom = colored_rows[-1]

                        # Calculate height as percentage
                        bar_top_y = y_min + top
                        bar_bottom_y = y_min + bottom

                        # Height from bottom of plot
                        height_pixels = y_max - bar_top_y
                        height_percent = (height_pixels / plot_height) * 100

                        x_positions.append(i)
                        bar_heights.append(height_percent)

            if bar_heights:
                data[color_name] = {
                    'x_positions': x_positions,
                    'values': bar_heights,
                    'color': color
                }

        return data

    def extract_scatter_plot_data(self, plot_area: Tuple[int, int, int, int],
                                  colors: List[Tuple[int, int, int]]) -> Dict:
        """
        Extract data from scatter plots.
        Returns: Dictionary with point coordinates
        """
        x_min, y_min, x_max, y_max = plot_area
        plot_height = y_max - y_min
        plot_width = x_max - x_min

        data = {}

        for color_idx, color in enumerate(colors):
            color_name = f"series_{color_idx}"

            # Create color mask
            color_array = np.array(color)
            color_tolerance = 30

            lower = np.clip(color_array - color_tolerance, 0, 255)
            upper = np.clip(color_array + color_tolerance, 0, 255)

            mask = cv2.inRange(self.image_rgb, lower, upper)

            # Find contours (points)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            x_coords = []
            y_coords = []

            for contour in contours:
                # Get center of each contour
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    # Check if point is within plot area
                    if x_min <= cx <= x_max and y_min <= cy <= y_max:
                        # Convert to plot coordinates (normalized)
                        x_norm = ((cx - x_min) / plot_width) * 100
                        y_norm = ((y_max - cy) / plot_height) * 100

                        x_coords.append(x_norm)
                        y_coords.append(y_norm)

            if x_coords:
                data[color_name] = {
                    'x_values': x_coords,
                    'y_values': y_coords,
                    'color': color
                }

        return data

    def extract_data(self, plot_type: str = 'auto') -> Dict:
        """
        Extract data from plot.

        Args:
            plot_type: 'bar', 'scatter', or 'auto' for automatic detection

        Returns:
            Dictionary containing extracted data
        """
        # Detect plot area
        plot_area = self.detect_plot_area()

        # Detect colors
        colors = self.detect_colors(n_colors=5)

        # Extract data based on plot type
        if plot_type == 'auto':
            # Try to detect plot type
            # For now, try both methods
            bar_data = self.extract_bar_chart_data(plot_area, colors)
            scatter_data = self.extract_scatter_plot_data(plot_area, colors)

            # Choose based on which has more data
            bar_count = sum(len(v.get('values', [])) for v in bar_data.values())
            scatter_count = sum(len(v.get('x_values', [])) for v in scatter_data.values())

            if bar_count > scatter_count:
                data = bar_data
                detected_type = 'bar'
            else:
                data = scatter_data
                detected_type = 'scatter'
        elif plot_type == 'bar':
            data = self.extract_bar_chart_data(plot_area, colors)
            detected_type = 'bar'
        else:
            data = self.extract_scatter_plot_data(plot_area, colors)
            detected_type = 'scatter'

        return {
            'plot_type': detected_type,
            'image_path': self.image_path,
            'plot_area': plot_area,
            'data': data,
            'colors': colors
        }

    def save_to_csv(self, data: Dict, output_path: str):
        """Save extracted data to CSV format."""
        plot_data = data['data']

        if data['plot_type'] == 'bar':
            # Create DataFrame for bar chart
            all_series = []
            for series_name, series_info in plot_data.items():
                df = pd.DataFrame({
                    'series': series_name,
                    'x_position': series_info.get('x_positions', []),
                    'value': series_info.get('values', []),
                    'color_r': series_info['color'][0],
                    'color_g': series_info['color'][1],
                    'color_b': series_info['color'][2]
                })
                all_series.append(df)

            if all_series:
                combined_df = pd.concat(all_series, ignore_index=True)
                combined_df.to_csv(output_path, index=False)
        else:
            # Create DataFrame for scatter plot
            all_series = []
            for series_name, series_info in plot_data.items():
                df = pd.DataFrame({
                    'series': series_name,
                    'x_value': series_info.get('x_values', []),
                    'y_value': series_info.get('y_values', []),
                    'color_r': series_info['color'][0],
                    'color_g': series_info['color'][1],
                    'color_b': series_info['color'][2]
                })
                all_series.append(df)

            if all_series:
                combined_df = pd.concat(all_series, ignore_index=True)
                combined_df.to_csv(output_path, index=False)

    def save_to_json(self, data: Dict, output_path: str):
        """Save extracted data to JSON format."""

        def convert_to_native(obj):
            """Convert numpy types to native Python types."""
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native(item) for item in obj]
            else:
                return obj

        # Convert numpy types to Python types for JSON serialization
        json_data = {
            'plot_type': data['plot_type'],
            'image_path': data['image_path'],
            'plot_area': data['plot_area'],
            'data': convert_to_native(data['data'])
        }

        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)


def process_all_images(input_dir: str, output_dir: str, plot_type: str = 'auto'):
    """Process all plot images in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Find all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f'*{ext}'))

    results = {}

    for image_file in sorted(image_files):
        print(f"Processing {image_file.name}...")

        try:
            extractor = PlotDataExtractor(str(image_file))
            data = extractor.extract_data(plot_type=plot_type)

            # Save outputs
            base_name = image_file.stem
            csv_path = output_path / f"{base_name}.csv"
            json_path = output_path / f"{base_name}.json"

            extractor.save_to_csv(data, str(csv_path))
            extractor.save_to_json(data, str(json_path))

            results[image_file.name] = {
                'status': 'success',
                'csv': str(csv_path),
                'json': str(json_path),
                'plot_type': data['plot_type']
            }

            print(f"  ✓ Saved to {csv_path} and {json_path}")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            results[image_file.name] = {
                'status': 'error',
                'error': str(e)
            }

    # Save summary
    summary_path = output_path / 'extraction_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nProcessing complete! Summary saved to {summary_path}")
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Extract data from plot images')
    parser.add_argument('--input', '-i', default='.',
                       help='Input directory containing plot images (default: current directory)')
    parser.add_argument('--output', '-o', default='extracted_data',
                       help='Output directory for extracted data (default: extracted_data)')
    parser.add_argument('--plot-type', '-t', choices=['auto', 'bar', 'scatter'],
                       default='auto', help='Type of plot (default: auto)')
    parser.add_argument('--single', '-s', help='Process a single image file')

    args = parser.parse_args()

    if args.single:
        # Process single file
        print(f"Processing {args.single}...")
        extractor = PlotDataExtractor(args.single)
        data = extractor.extract_data(plot_type=args.plot_type)

        output_path = Path(args.output)
        output_path.mkdir(exist_ok=True)

        base_name = Path(args.single).stem
        csv_path = output_path / f"{base_name}.csv"
        json_path = output_path / f"{base_name}.json"

        extractor.save_to_csv(data, str(csv_path))
        extractor.save_to_json(data, str(json_path))

        print(f"✓ Data extracted successfully!")
        print(f"  CSV: {csv_path}")
        print(f"  JSON: {json_path}")
    else:
        # Process all files in directory
        process_all_images(args.input, args.output, args.plot_type)


if __name__ == '__main__':
    main()
