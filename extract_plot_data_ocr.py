#!/usr/bin/env python3
"""
OCR-Based Plot Data Extraction
Uses OCR to read actual labeled values from plot images for accurate data extraction.
"""

import cv2
import pytesseract
from PIL import Image
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re


class OCRPlotExtractor:
    """Extract plot data using OCR to read actual labeled values."""

    def __init__(self, image_path: str, metadata_path: Optional[str] = None):
        """
        Initialize the OCR extractor.

        Args:
            image_path: Path to the plot image
            metadata_path: Optional path to metadata JSON file
        """
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")

        self.height, self.width = self.image.shape[:2]
        self.metadata = None

        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                all_metadata = json.load(f)
                image_name = Path(image_path).name
                self.metadata = all_metadata.get(image_name, {})

    def preprocess_for_ocr(self, image: np.ndarray, invert: bool = False) -> np.ndarray:
        """
        Preprocess image region for better OCR results.

        Args:
            image: OpenCV image region
            invert: Whether to invert colors (useful for light text on dark background)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # Apply adaptive threshold
        binary = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        if invert:
            binary = cv2.bitwise_not(binary)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary)

        # Scale up for better OCR (3x works better than 2x)
        scaled = cv2.resize(denoised, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        return scaled

    def extract_y_axis_values(self) -> List[float]:
        """
        Extract y-axis tick values using OCR, with metadata as fallback.

        Returns:
            List of y-axis values in ascending order
        """
        # First, try to use metadata if available (more reliable)
        if self.metadata and 'y_axis' in self.metadata:
            y_range = self.metadata['y_axis'].get('range', [])
            if len(y_range) == 2:
                y_min, y_max = y_range
                # Create reasonable tick marks (typically 5-6 ticks)
                num_ticks = 6
                step = (y_max - y_min) / (num_ticks - 1)
                tick_values = [y_min + i * step for i in range(num_ticks)]
                print(f"Using y-axis range from metadata: {y_range}")
                return tick_values

        # Fallback to OCR if no metadata
        # Try full image first to catch all text
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Try multiple preprocessing approaches
        processed_images = []

        # Approach 1: Y-axis region with standard preprocessing
        y_axis_x = int(self.width * 0.15)
        y_axis_region = self.image[
            int(self.height * 0.05):int(self.height * 0.95),  # Extend to capture 0 and 100
            0:y_axis_x
        ]
        processed_images.append(self.preprocess_for_ocr(y_axis_region))

        # Approach 2: Left region with inverted colors
        processed_images.append(self.preprocess_for_ocr(y_axis_region, invert=True))

        # Approach 3: Simple threshold on y-axis region
        y_gray = cv2.cvtColor(y_axis_region, cv2.COLOR_BGR2GRAY)
        _, simple_thresh = cv2.threshold(y_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        scaled_simple = cv2.resize(simple_thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        processed_images.append(scaled_simple)

        # Try multiple OCR configurations
        custom_configs = [
            '--psm 6',  # Assume uniform block of text
            '--psm 4',  # Assume single column of text
            '--psm 11', # Sparse text
            '--psm 12', # Sparse text without OSD
        ]

        all_numbers = []
        for processed in processed_images:
            for config in custom_configs:
                try:
                    text = pytesseract.image_to_string(
                        Image.fromarray(processed),
                        config=config + ' -c tessedit_char_whitelist=0123456789.-'
                    )
                    # Extract all numbers (including decimals and negatives)
                    numbers = re.findall(r'-?\d+\.?\d*', text)
                    all_numbers.extend([float(n) for n in numbers if n and n != '.'])
                except:
                    continue

        # Remove duplicates and sort
        unique_values = sorted(set(all_numbers))

        # Filter out obvious outliers (keep values in reasonable range)
        # Most plots are 0-100 or similar ranges
        if unique_values:
            # Remove values that are too far from the median cluster
            median = np.median(unique_values)
            # Keep values within reasonable range (filter out stray detections)
            filtered = [v for v in unique_values if abs(v - median) <= 200]
            unique_values = filtered if filtered else unique_values

            # Further filtering: remove very small numbers that are likely misreads
            # and very large numbers that don't fit the pattern
            if len(unique_values) >= 3:
                # Find the most common step size
                diffs = [unique_values[i+1] - unique_values[i] for i in range(len(unique_values)-1)]
                if diffs:
                    typical_step = np.median(diffs)
                    # Remove outliers that don't fit the step pattern
                    filtered2 = []
                    for i, val in enumerate(unique_values):
                        if i == 0:
                            filtered2.append(val)
                        else:
                            # Check if this value is roughly one step from previous
                            expected_range = (filtered2[-1] + typical_step * 0.3,
                                            filtered2[-1] + typical_step * 2.5)
                            if expected_range[0] <= val <= expected_range[1]:
                                filtered2.append(val)
                    if len(filtered2) >= 3:
                        unique_values = filtered2

        # If we got reasonable values, try to infer missing ones
        if len(unique_values) >= 3:
            # Check if it looks like a regular scale (e.g., 0, 20, 40, 60, 80, 100)
            if len(unique_values) >= 2:
                # Calculate likely step size
                diffs = [unique_values[i+1] - unique_values[i] for i in range(len(unique_values)-1)]
                avg_diff = np.median(diffs) if diffs else 20

                # Infer min and max if missing
                min_val = min(unique_values)
                max_val = max(unique_values)

                # Try to add 0 if it seems like it should be there
                if min_val > 0 and min_val <= avg_diff * 1.5:
                    unique_values.insert(0, 0.0)

                # Try to add max if it seems like it should be there
                expected_max = max_val + avg_diff
                if expected_max <= max_val * 1.5 and max_val < 150:
                    unique_values.append(expected_max)

        # Use metadata as fallback
        if len(unique_values) < 2 and self.metadata and 'y_axis' in self.metadata:
            y_range = self.metadata['y_axis'].get('range', [])
            if len(y_range) == 2:
                # Create reasonable tick marks
                y_min, y_max = y_range
                step = (y_max - y_min) / 5
                unique_values = [y_min + i * step for i in range(6)]

        return unique_values

    def extract_x_axis_labels(self) -> List[str]:
        """
        Extract x-axis labels (country codes) using OCR.

        Returns:
            List of country codes or categories
        """
        # Define x-axis region (bottom of image)
        x_axis_y = int(self.height * 0.85)
        x_axis_region = self.image[
            x_axis_y:self.height,
            int(self.width * 0.1):int(self.width * 0.9)
        ]

        # Preprocess
        processed = self.preprocess_for_ocr(x_axis_region)

        # Extract text
        text = pytesseract.image_to_string(
            Image.fromarray(processed),
            config='--psm 6'
        )

        # Extract country codes (2-3 uppercase letters)
        codes = re.findall(r'\b[A-Z]{2,3}\b', text)

        # Remove duplicates while preserving order
        seen = set()
        unique_codes = []
        for code in codes:
            if code not in seen:
                seen.add(code)
                unique_codes.append(code)

        return unique_codes

    def detect_plot_area(self) -> Tuple[int, int, int, int]:
        """
        Detect the plot area boundaries.

        Returns:
            (x_min, y_min, x_max, y_max) coordinates
        """
        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest rectangular contour
            max_area = 0
            best_rect = None

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h

                # Filter reasonable plot areas
                if (area > max_area and
                    w > self.width * 0.3 and h > self.height * 0.3):
                    max_area = area
                    best_rect = (x, y, x + w, y + h)

            if best_rect:
                return best_rect

        # Fallback to estimated area
        x_min = int(self.width * 0.15)
        y_min = int(self.height * 0.15)
        x_max = int(self.width * 0.85)
        y_max = int(self.height * 0.85)

        return (x_min, y_min, x_max, y_max)

    def detect_colors(self, plot_area: Tuple[int, int, int, int],
                     n_colors: int = 5) -> List[Tuple[int, int, int]]:
        """
        Detect dominant colors in the plot area (for identifying series).

        Args:
            plot_area: (x_min, y_min, x_max, y_max)
            n_colors: Number of colors to detect

        Returns:
            List of RGB color tuples
        """
        x_min, y_min, x_max, y_max = plot_area
        plot_img = self.image[y_min:y_max, x_min:x_max]

        # Reshape to list of pixels
        pixels = plot_img.reshape(-1, 3)

        # Convert to float
        pixels = np.float32(pixels)

        # Apply k-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)

        # Get colors and their frequencies
        labels = kmeans.labels_
        colors = kmeans.cluster_centers_.astype(int)

        # Count frequency of each color
        unique, counts = np.unique(labels, return_counts=True)
        color_freq = dict(zip(unique, counts))

        # Filter out background colors
        filtered_colors = []
        for idx, color in enumerate(colors):
            r, g, b = color

            # Skip white and near-white
            if r > 240 and g > 240 and b > 240:
                continue

            # Skip very light gray (likely background)
            if r > 220 and g > 220 and b > 220:
                continue

            # Skip colors that are too similar to gray/background
            # Gray has r≈g≈b, so check color saturation
            max_rgb = max(r, g, b)
            min_rgb = min(r, g, b)
            saturation = (max_rgb - min_rgb) / (max_rgb + 1)  # +1 to avoid division by zero

            # Skip low saturation colors (gray-ish) unless they're frequent
            if saturation < 0.15 and color_freq.get(idx, 0) < pixels.shape[0] * 0.1:
                continue

            # Skip very dark colors (likely borders/text)
            if max(r, g, b) < 30:
                continue

            filtered_colors.append((int(r), int(g), int(b)))

        # If we have metadata series colors, try to match them
        if self.metadata and 'series' in self.metadata:
            expected_colors = []
            for series_info in self.metadata['series'].values():
                color_name = series_info.get('color', '').lower()
                # Map common color names to RGB ranges
                if 'blue' in color_name:
                    expected_colors.append('blue')
                elif 'red' in color_name:
                    expected_colors.append('red')
                elif 'green' in color_name:
                    expected_colors.append('green')
                elif 'yellow' in color_name or 'orange' in color_name:
                    expected_colors.append('yellow/orange')

        # Merge similar colors (colors within 40 RGB distance)
        merged_colors = []
        for color in filtered_colors:
            is_similar = False
            for merged in merged_colors:
                r1, g1, b1 = color
                r2, g2, b2 = merged
                distance = np.sqrt((r1-r2)**2 + (g1-g2)**2 + (b1-b2)**2)
                if distance < 40:
                    is_similar = True
                    break
            if not is_similar:
                merged_colors.append(color)

        return merged_colors[:4]  # Limit to 4 colors max

    def extract_bar_values_with_ocr(self, y_axis_values: List[float]) -> Dict:
        """
        Extract bar chart values by combining vision (bar detection) with OCR (axis values).

        Args:
            y_axis_values: Y-axis tick values from OCR

        Returns:
            Dictionary with extracted data
        """
        # Detect plot area
        plot_area = self.detect_plot_area()
        x_min, y_min, x_max, y_max = plot_area

        # Detect colors
        colors = self.detect_colors(plot_area)

        # Get metadata for context
        countries = []
        if self.metadata and 'x_axis' in self.metadata:
            countries = self.metadata['x_axis'].get('countries', [])

        # Calculate y-axis mapping
        if len(y_axis_values) >= 2:
            y_min_val = min(y_axis_values)
            y_max_val = max(y_axis_values)

            # Create pixel to value mapping
            def pixel_y_to_value(pixel_y):
                """Convert pixel y-coordinate to actual value."""
                # Normalize pixel position within plot area
                normalized = (y_max - pixel_y) / (y_max - y_min)
                # Map to value range
                value = y_min_val + normalized * (y_max_val - y_min_val)
                return value
        else:
            # Fallback if OCR didn't work
            pixel_y_to_value = lambda py: ((y_max - py) / (y_max - y_min)) * 100

        # Extract data points by scanning for continuous colored regions
        data_points = []

        # Scan horizontally across plot area
        scan_width = x_max - x_min
        num_scans = 100  # Sample 100 positions

        for i in range(num_scans):
            x_pos = x_min + int(i * scan_width / num_scans)

            # Extract column pixels
            column = self.image[y_min:y_max, x_pos]

            # For each color, find continuous regions from bottom
            for color_idx, target_color in enumerate(colors):
                # Scan from bottom to top to find bars
                tb, tg, tr = target_color

                # Find all pixels matching this color
                matching_pixels = []
                for y_offset in range(len(column) - 1, -1, -1):  # Bottom to top
                    pixel = column[y_offset]
                    b, g, r = pixel

                    # Calculate color distance
                    color_dist = abs(int(r)-int(tr)) + abs(int(g)-int(tg)) + abs(int(b)-int(tb))

                    if color_dist < 80:  # Threshold for color matching
                        matching_pixels.append(y_offset)

                # If we found matching pixels, get the top (minimum y_offset)
                if matching_pixels:
                    top_y_offset = min(matching_pixels)
                    pixel_y = y_min + top_y_offset
                    value = pixel_y_to_value(pixel_y)

                    # Map to country if available
                    country = None
                    if countries:
                        country_idx = int(i * len(countries) / num_scans)
                        if 0 <= country_idx < len(countries):
                            country = countries[country_idx]

                    # Only add if value is significant (not near baseline)
                    if value > y_min_val + 1:  # More than 1 unit above baseline
                        data_points.append({
                            'x_position': i,
                            'country': country,
                            'value': round(value, 2),
                            'series': f'series_{color_idx}',
                            'color_r': int(tr),
                            'color_g': int(tg),
                            'color_b': int(tb)
                        })

        # Group by x_position and series, take median value to reduce noise
        import pandas as pd
        if data_points:
            df = pd.DataFrame(data_points)
            # Group nearby x_positions (within ±2 positions) and same series
            df['x_group'] = (df['x_position'] / 3).astype(int)

            # Take one value per x_group and series
            grouped = df.groupby(['x_group', 'series', 'country'], dropna=False).agg({
                'x_position': 'first',
                'value': 'median',
                'color_r': 'first',
                'color_g': 'first',
                'color_b': 'first'
            }).reset_index()

            # Drop the x_group column, keep series and country
            grouped = grouped.drop('x_group', axis=1)

            data_points = grouped.to_dict('records')

        return {
            'data_points': data_points,
            'y_axis_values': y_axis_values,
            'plot_area': plot_area,
            'colors': colors
        }

    def extract_data(self) -> Dict:
        """
        Main extraction method that combines OCR and computer vision.

        Returns:
            Dictionary with extracted data and metadata
        """
        print(f"Processing: {self.image_path}")
        print(f"Image size: {self.width}x{self.height}")

        # Extract y-axis values using OCR
        print("Extracting y-axis values with OCR...")
        y_axis_values = self.extract_y_axis_values()
        print(f"Y-axis values detected: {y_axis_values}")

        # Extract x-axis labels using OCR
        print("Extracting x-axis labels with OCR...")
        x_axis_labels = self.extract_x_axis_labels()
        print(f"X-axis labels detected: {x_axis_labels[:10]}...")  # Show first 10

        # Prefer metadata countries if available
        if self.metadata and 'x_axis' in self.metadata:
            countries = self.metadata['x_axis'].get('countries', [])
            if countries:
                print(f"Using {len(countries)} countries from metadata")

        # Extract bar values
        print("Extracting bar values...")
        result = self.extract_bar_values_with_ocr(y_axis_values)

        # Add metadata
        result['metadata'] = self.metadata or {}
        result['ocr_x_labels'] = x_axis_labels

        return result

    def save_to_csv(self, data: Dict, output_path: str):
        """Save extracted data to CSV with metadata columns."""
        df = pd.DataFrame(data['data_points'])

        # Add metadata columns
        if data.get('metadata'):
            meta = data['metadata']
            df['plot_title'] = meta.get('title', '')
            df['plot_description'] = meta.get('description', '')

            if 'x_axis' in meta:
                df['x_axis_label'] = meta['x_axis'].get('label', '')

            if 'y_axis' in meta:
                df['y_axis_label'] = meta['y_axis'].get('label', '')
                df['value_unit'] = meta['y_axis'].get('unit', '')

        df.to_csv(output_path, index=False)
        print(f"Saved CSV: {output_path}")

    def save_to_json(self, data: Dict, output_path: str):
        """Save extracted data to JSON."""
        # Convert to serializable format
        output = {
            'data_points': data['data_points'],
            'y_axis_values': data['y_axis_values'],
            'ocr_x_labels': data.get('ocr_x_labels', []),
            'metadata': data.get('metadata', {}),
            'plot_area': data['plot_area'],
            'colors': [{'r': c[0], 'g': c[1], 'b': c[2]} for c in data['colors']]
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Saved JSON: {output_path}")


def main():
    """Process all plot images with OCR-based extraction."""

    # Setup paths
    script_dir = Path(__file__).parent
    metadata_path = script_dir / 'plot_metadata.json'
    output_dir = script_dir / 'extracted_data_ocr'
    output_dir.mkdir(exist_ok=True)

    # Find all plot images
    image_files = list(script_dir.glob('*.jpg')) + list(script_dir.glob('*.png'))

    if not image_files:
        print("No plot images found!")
        return

    print(f"Found {len(image_files)} plot images")
    print("=" * 80)

    # Process each image
    results = {}

    for image_file in sorted(image_files):
        try:
            print(f"\n{'='*80}")

            # Extract data
            extractor = OCRPlotExtractor(str(image_file), str(metadata_path))
            data = extractor.extract_data()

            # Save outputs
            base_name = image_file.stem
            csv_path = output_dir / f"{base_name}.csv"
            json_path = output_dir / f"{base_name}.json"

            extractor.save_to_csv(data, str(csv_path))
            extractor.save_to_json(data, str(json_path))

            results[image_file.name] = {
                'status': 'success',
                'csv': str(csv_path),
                'json': str(json_path),
                'num_points': len(data['data_points']),
                'y_axis_values': data['y_axis_values']
            }

        except Exception as e:
            print(f"Error processing {image_file.name}: {e}")
            results[image_file.name] = {
                'status': 'error',
                'error': str(e)
            }

    # Save summary
    summary_path = output_dir / 'extraction_summary_ocr.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Processing complete!")
    print(f"Results saved to: {output_dir}/")
    print(f"Summary: {summary_path}")

    # Print statistics
    successful = sum(1 for r in results.values() if r['status'] == 'success')
    failed = len(results) - successful
    print(f"\nSuccessful: {successful}/{len(results)}")
    if failed > 0:
        print(f"Failed: {failed}/{len(results)}")


if __name__ == '__main__':
    main()
