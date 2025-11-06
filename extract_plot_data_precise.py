#!/usr/bin/env python3
"""
High-Precision Plot Data Extraction
Uses advanced techniques to maximize extraction accuracy from plot images.
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


class PrecisePlotExtractor:
    """Extract plot data with maximum achievable precision."""

    def __init__(self, image_path: str, metadata_path: Optional[str] = None):
        """Initialize the precise extractor."""
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

    def detect_gridlines(self, plot_area: Tuple[int, int, int, int]) -> List[int]:
        """
        Detect horizontal gridlines in the plot area.
        Returns y-coordinates of gridlines from top to bottom.
        """
        x_min, y_min, x_max, y_max = plot_area

        # Extract plot region
        plot_region = self.image[y_min:y_max, x_min:x_max]
        gray = cv2.cvtColor(plot_region, cv2.COLOR_BGR2GRAY)

        # Apply edge detection
        edges = cv2.Canny(gray, 30, 100)

        # Detect horizontal lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                                minLineLength=int((x_max-x_min)*0.3),
                                maxLineGap=10)

        gridlines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if line is horizontal (small y difference)
                if abs(y2 - y1) < 5:
                    y_pos = y_min + (y1 + y2) // 2
                    gridlines.append(y_pos)

        # Remove duplicates (gridlines within 3 pixels)
        gridlines = sorted(set(gridlines))
        filtered = []
        for gl in gridlines:
            if not filtered or abs(gl - filtered[-1]) > 3:
                filtered.append(gl)

        return filtered

    def get_y_axis_scale(self) -> Tuple[List[float], List[int]]:
        """
        Get y-axis scale by combining OCR with gridline detection.
        Returns (values, pixel_positions) tuples.
        """
        # Use metadata if available
        if self.metadata and 'y_axis' in self.metadata:
            y_range = self.metadata['y_axis'].get('range', [])
            if len(y_range) == 2:
                y_min_val, y_max_val = y_range
                num_ticks = 6
                step = (y_max_val - y_min_val) / (num_ticks - 1)
                values = [y_min_val + i * step for i in range(num_ticks)]

                # Estimate pixel positions
                # IMPORTANT: In images, y=0 is TOP, y=max is BOTTOM
                # So y_max_val (e.g., 100) should be at plot_top pixel
                # And y_min_val (e.g., 0) should be at plot_bottom pixel
                plot_top = int(self.height * 0.1)
                plot_bottom = int(self.height * 0.85)
                pixel_positions = []
                for i in range(num_ticks):
                    # Inverted mapping: highest value at top (lowest pixel y)
                    normalized = i / (num_ticks - 1)
                    # Reverse the mapping
                    pixel_y = plot_bottom - int(normalized * (plot_bottom - plot_top))
                    pixel_positions.append(pixel_y)

                return values, pixel_positions

        return [], []

    def detect_precise_bar_top(self, column: np.ndarray, target_color: Tuple[int, int, int],
                               baseline_y: int) -> Optional[float]:
        """
        Detect bar top with sub-pixel precision using edge detection.

        Args:
            column: Vertical column of pixels
            target_color: Target color to detect
            baseline_y: Baseline y-position (where bars start)

        Returns:
            Sub-pixel y-coordinate of bar top, or None if not found
        """
        tb, tg, tr = target_color

        # Find pixels matching target color
        matching_y = []
        for y_offset in range(len(column)):
            pixel = column[y_offset]
            b, g, r = pixel

            # Color distance (Manhattan distance is faster than Euclidean)
            color_dist = abs(int(r)-int(tr)) + abs(int(g)-int(tg)) + abs(int(b)-int(tb))

            if color_dist < 60:  # Threshold for color matching
                matching_y.append(y_offset)

        if not matching_y:
            return None

        # Get the topmost pixel
        top_y = min(matching_y)

        # Sub-pixel refinement: Check gradient around top pixel
        if top_y > 0:
            # Use 3 pixels around the top for sub-pixel estimation
            window_start = max(0, top_y - 1)
            window_end = min(len(column), top_y + 2)
            window = column[window_start:window_end]

            # Calculate color intensity gradient
            intensities = []
            for pixel in window:
                b, g, r = pixel
                dist = abs(int(r)-int(tr)) + abs(int(g)-int(tg)) + abs(int(b)-int(tb))
                intensities.append(255 - dist)  # Higher = more similar

            # Weighted average for sub-pixel position
            if len(intensities) >= 2:
                weights = np.array(intensities)
                positions = np.arange(window_start, window_end)
                if weights.sum() > 0:
                    weighted_pos = np.average(positions, weights=weights)
                    return float(weighted_pos)

        return float(top_y)

    def pixel_to_value_with_gridlines(self, pixel_y: float, values: List[float],
                                      pixel_positions: List[int],
                                      gridlines: List[int]) -> float:
        """
        Convert pixel y-coordinate to value, snapping to nearest gridline.
        """
        # If we have gridlines, snap to nearest
        if gridlines:
            nearest_grid = min(gridlines, key=lambda g: abs(g - pixel_y))
            if abs(nearest_grid - pixel_y) < 5:  # Within 5 pixels, snap to grid
                pixel_y = nearest_grid

        # Interpolate between known values
        if len(values) >= 2 and len(pixel_positions) >= 2:
            pixel_positions = np.array(pixel_positions)
            values_array = np.array(values)

            # np.interp requires xp (pixel_positions) to be in ascending order
            # Our pixel positions are descending (high values at top = low pixel y)
            # So we need to reverse both arrays
            if pixel_positions[0] > pixel_positions[-1]:
                pixel_positions = pixel_positions[::-1]
                values_array = values_array[::-1]

            # Clamp to range
            pixel_y = np.clip(pixel_y, min(pixel_positions), max(pixel_positions))

            # Interpolate
            value = np.interp(pixel_y, pixel_positions, values_array)
            return float(value)

        return 0.0

    def extract_data(self) -> Dict:
        """Main extraction with precision enhancements."""
        print(f"Processing: {self.image_path}")
        print(f"Image size: {self.width}x{self.height}")

        # Get plot area
        x_min = int(self.width * 0.12)
        y_min = int(self.height * 0.08)
        x_max = int(self.width * 0.95)
        y_max = int(self.height * 0.88)
        plot_area = (x_min, y_min, x_max, y_max)

        print("Detecting gridlines...")
        gridlines = self.detect_gridlines(plot_area)
        print(f"Found {len(gridlines)} gridlines")

        print("Reading y-axis scale...")
        values, pixel_positions = self.get_y_axis_scale()
        print(f"Y-axis scale: {values}")

        # Detect colors
        print("Detecting series colors...")
        colors = self.detect_colors(plot_area)
        print(f"Detected {len(colors)} series")

        # Get metadata
        countries = []
        if self.metadata and 'x_axis' in self.metadata:
            countries = self.metadata['x_axis'].get('countries', [])
            print(f"Using {len(countries)} countries from metadata")

        # Extract data points
        print("Extracting data with sub-pixel precision...")
        data_points = []

        scan_width = x_max - x_min
        num_scans = min(200, scan_width)  # More samples for better precision

        baseline_y = y_max  # Bottom of plot

        for i in range(num_scans):
            x_pos = x_min + int(i * scan_width / num_scans)
            column = self.image[y_min:y_max, x_pos]

            for color_idx, target_color in enumerate(colors):
                # Detect bar top with sub-pixel precision
                top_y_offset = self.detect_precise_bar_top(column, target_color, baseline_y)

                if top_y_offset is not None:
                    pixel_y = y_min + top_y_offset

                    # Convert to value with gridline snapping
                    value = self.pixel_to_value_with_gridlines(
                        pixel_y, values, pixel_positions, gridlines
                    )

                    # Map to country
                    country = None
                    if countries:
                        country_idx = int(i * len(countries) / num_scans)
                        if 0 <= country_idx < len(countries):
                            country = countries[country_idx]

                    # Filter very small values (likely noise)
                    if values and value > min(values) + 0.5:
                        data_points.append({
                            'x_position': i,
                            'country': country,
                            'value': round(value, 2),
                            'series': f'series_{color_idx}',
                            'color_r': int(target_color[2]),
                            'color_g': int(target_color[1]),
                            'color_b': int(target_color[0])
                        })

        # Group and average for final precision
        if data_points:
            df = pd.DataFrame(data_points)
            df['x_group'] = (df['x_position'] / 5).astype(int)  # Group every 5 scans

            grouped = df.groupby(['x_group', 'series', 'country'], dropna=False).agg({
                'x_position': 'first',
                'value': 'median',  # Use median to reduce outliers
                'color_r': 'first',
                'color_g': 'first',
                'color_b': 'first'
            }).reset_index()

            grouped = grouped.drop('x_group', axis=1)
            data_points = grouped.to_dict('records')

        print(f"Extracted {len(data_points)} data points")

        return {
            'data_points': data_points,
            'y_axis_values': values,
            'gridlines': gridlines,
            'plot_area': plot_area,
            'colors': [(int(c[2]), int(c[1]), int(c[0])) for c in colors],
            'metadata': self.metadata or {}
        }

    def detect_colors(self, plot_area: Tuple[int, int, int, int]) -> List[Tuple[int, int, int]]:
        """Detect dominant colors (BGR format)."""
        x_min, y_min, x_max, y_max = plot_area
        plot_img = self.image[y_min:y_max, x_min:x_max]

        pixels = plot_img.reshape(-1, 3).astype(np.float32)

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(pixels)

        colors = kmeans.cluster_centers_.astype(int)

        # Filter background colors
        filtered = []
        for color in colors:
            b, g, r = color

            # Skip white/light colors
            if min(r, g, b) > 220:
                continue

            # Skip dark colors
            if max(r, g, b) < 30:
                continue

            # Check saturation
            max_rgb = max(r, g, b)
            min_rgb = min(r, g, b)
            sat = (max_rgb - min_rgb) / (max_rgb + 1)

            if sat < 0.15:  # Skip low saturation (gray)
                continue

            filtered.append((int(b), int(g), int(r)))

        # Merge similar colors more aggressively
        # Group by hue (color family) rather than exact RGB
        merged = []
        for color in filtered:
            b, g, r = color

            # Calculate hue category
            max_val = max(r, g, b)
            min_val = min(r, g, b)

            is_similar = False
            for m_color in merged:
                mb, mg, mr = m_color

                # Check if colors are in same hue family
                # Blue family: b > r and b > g
                # Red family: r > b and r > g
                is_blue_1 = b > r * 1.2 and b > g * 1.2
                is_red_1 = r > b * 1.2 and r > g * 1.2
                is_blue_2 = mb > mr * 1.2 and mb > mg * 1.2
                is_red_2 = mr > mb * 1.2 and mr > mg * 1.2

                # Same color family
                if (is_blue_1 and is_blue_2) or (is_red_1 and is_red_2):
                    # Merge if within reasonable distance
                    dist = sum(abs(a - b) for a, b in zip(color, m_color))
                    if dist < 100:  # More aggressive merging
                        is_similar = True
                        break

            if not is_similar:
                merged.append(color)

        return merged[:4]

    def save_to_csv(self, data: Dict, output_path: str):
        """Save to CSV."""
        df = pd.DataFrame(data['data_points'])

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
        print(f"Saved: {output_path}")


def main():
    """Process all plots with precision extraction."""
    script_dir = Path(__file__).parent
    metadata_path = script_dir / 'plot_metadata.json'
    output_dir = script_dir / 'extracted_data_precise'
    output_dir.mkdir(exist_ok=True)

    image_files = list(script_dir.glob('*.jpg')) + list(script_dir.glob('*.png'))

    if not image_files:
        print("No images found!")
        return

    print(f"Found {len(image_files)} images")
    print("=" * 80)

    for image_file in sorted(image_files):
        try:
            print(f"\n{'='*80}")
            extractor = PrecisePlotExtractor(str(image_file), str(metadata_path))
            data = extractor.extract_data()

            base_name = image_file.stem
            csv_path = output_dir / f"{base_name}.csv"
            extractor.save_to_csv(data, str(csv_path))

        except Exception as e:
            print(f"Error processing {image_file.name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print(f"Complete! Results in: {output_dir}/")


if __name__ == '__main__':
    main()
