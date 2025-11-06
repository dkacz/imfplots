#!/usr/bin/env python3
"""
Grid-Aligned Precision Extraction
Matches OCR labels to actual gridlines for maximum calibration accuracy.
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


class GridAlignedExtractor:
    """Extract with precise grid-label alignment."""

    def __init__(self, image_path: str, metadata_path: Optional[str] = None):
        """Initialize extractor."""
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load: {image_path}")

        # Super-resolution: upscale 8x for maximum precision
        print("Upscaling image 8x for maximum precision...")
        self.image = cv2.resize(self.original_image, None, fx=8, fy=8,
                                interpolation=cv2.INTER_CUBIC)

        self.height, self.width = self.image.shape[:2]
        self.scale = 8

        self.metadata = None
        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                all_metadata = json.load(f)
                image_name = Path(image_path).name
                self.metadata = all_metadata.get(image_name, {})

    def find_yaxis_labels_with_boxes(self) -> List[Tuple[float, float, float, float]]:
        """
        Find Y-axis labels with their bounding boxes.
        Returns list of (value, y_top, y_bottom, y_center) tuples.
        """
        print("Finding Y-axis labels with bounding boxes...")

        # Extract Y-axis region
        y_axis_width = int(self.width * 0.15)
        y_axis_region = self.image[:, :y_axis_width]

        # Preprocess for OCR
        gray = cv2.cvtColor(y_axis_region, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        denoised = cv2.fastNlMeansDenoising(binary)

        # Get OCR data with bounding boxes
        ocr_data = pytesseract.image_to_data(Image.fromarray(denoised),
                                             config='--psm 6 -c tessedit_char_whitelist=0123456789',
                                             output_type=pytesseract.Output.DICT)

        labels_with_boxes = []

        for i in range(len(ocr_data['text'])):
            text = ocr_data['text'][i].strip()
            conf = int(ocr_data['conf'][i])

            if text and conf > 40:
                try:
                    value = float(text)

                    # Get bounding box
                    top = ocr_data['top'][i]
                    height = ocr_data['height'][i]
                    bottom = top + height
                    center = top + height / 2.0

                    # Validate value range
                    if self.metadata and 'y_axis' in self.metadata:
                        y_range = self.metadata['y_axis'].get('range', [0, 100])
                        if not (y_range[0] <= value <= y_range[1]):
                            continue
                    elif not (0 <= value <= 100):
                        continue

                    labels_with_boxes.append((value, top, bottom, center))
                    print(f"  Label '{value}': Y={top:.1f}-{bottom:.1f} (center={center:.1f})")
                except ValueError:
                    continue

        return labels_with_boxes

    def find_gridlines_in_plot(self, plot_area: Tuple[int, int, int, int]) -> List[Tuple[float, float]]:
        """
        Find horizontal gridlines in plot area.
        Returns list of (y_position, line_length) tuples.
        """
        x_min, y_min, x_max, y_max = plot_area

        print("\nDetecting gridlines in plot area...")

        # Extract plot region
        plot_region = self.image[y_min:y_max, x_min:x_max]
        gray = cv2.cvtColor(plot_region, cv2.COLOR_BGR2GRAY)

        # Apply edge detection
        edges = cv2.Canny(gray, 15, 45)

        # Detect horizontal lines
        min_line_length = int((x_max - x_min) * 0.4)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=80,
                                minLineLength=min_line_length, maxLineGap=30)

        gridlines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Check if horizontal
                if abs(y2 - y1) < 3:
                    y_pos = y_min + (y1 + y2) / 2.0
                    length = abs(x2 - x1)
                    gridlines.append((y_pos, length))

        # Cluster nearby gridlines
        if gridlines:
            gridlines.sort(key=lambda x: x[0])
            clustered = []
            current_cluster = [gridlines[0]]

            for gl in gridlines[1:]:
                if abs(gl[0] - current_cluster[0][0]) < 8:  # Within 8 pixels
                    current_cluster.append(gl)
                else:
                    # Average the cluster, weighted by length
                    total_weight = sum(length for _, length in current_cluster)
                    avg_y = sum(y * length for y, length in current_cluster) / total_weight
                    avg_length = np.mean([length for _, length in current_cluster])
                    clustered.append((avg_y, avg_length))
                    current_cluster = [gl]

            if current_cluster:
                total_weight = sum(length for _, length in current_cluster)
                avg_y = sum(y * length for y, length in current_cluster) / total_weight
                avg_length = np.mean([length for _, length in current_cluster])
                clustered.append((avg_y, avg_length))

            gridlines = clustered

        gridlines.sort(key=lambda x: x[0])
        print(f"Found {len(gridlines)} gridlines:")
        for i, (y, length) in enumerate(gridlines):
            print(f"  Gridline {i}: Y={y:.1f} (length={length:.0f}px)")

        return gridlines

    def match_gridlines_to_labels(self, labels: List[Tuple[float, float, float, float]],
                                  gridlines: List[Tuple[float, float]]) -> Dict[float, float]:
        """
        Match gridlines to labels and create calibration.

        Args:
            labels: List of (value, y_top, y_bottom, y_center)
            gridlines: List of (y_position, length)

        Returns:
            Dict mapping gridline_y -> value
        """
        print("\nMatching gridlines to labels...")

        calibration = {}

        # For each label, find the nearest gridline
        for value, y_top, y_bottom, y_center in labels:
            # Look for gridline near the label (usually aligned with or just above text)
            # Check range from above text to center of text
            search_range = (y_top - 30, y_center + 10)  # Generous range

            candidates = [(y, length) for y, length in gridlines
                         if search_range[0] <= y <= search_range[1]]

            if candidates:
                # Take the longest line (most likely to be the actual gridline)
                best_gridline = max(candidates, key=lambda x: x[1])
                gridline_y = best_gridline[0]
                calibration[gridline_y] = value
                print(f"  {value}% → Gridline at Y={gridline_y:.1f} (label center={y_center:.1f}, offset={gridline_y-y_center:.1f}px)")
            else:
                # No gridline found, use label center as fallback
                print(f"  {value}% → No gridline found, using label center Y={y_center:.1f}")
                calibration[y_center] = value

        return calibration

    def interpolate_calibration(self, calibration: Dict[float, float]) -> Dict[float, float]:
        """Add interpolated points for missing values (e.g., if 60 or 0 are missing)."""
        if len(calibration) < 2:
            return calibration

        points = sorted(calibration.items())

        # Calculate average spacing
        spacings = []
        for i in range(len(points) - 1):
            y1, v1 = points[i]
            y2, v2 = points[i+1]
            if v1 != v2:
                px_per_unit = (y2 - y1) / (v1 - v2)
                spacings.append(abs(px_per_unit))

        if not spacings:
            return calibration

        avg_spacing = np.median(spacings)
        print(f"\nAverage spacing: {avg_spacing:.2f} pixels per unit")

        # Get value range
        if self.metadata and 'y_axis' in self.metadata:
            y_range = self.metadata['y_axis'].get('range', [0, 100])
            y_min_val, y_max_val = y_range
        else:
            y_min_val, y_max_val = 0, 100

        # Create complete calibration
        values_present = set(calibration.values())
        complete_cal = dict(calibration)

        # Expected values (e.g., 0, 20, 40, 60, 80, 100)
        step = 20
        expected_values = list(range(int(y_min_val), int(y_max_val) + 1, step))

        for expected in expected_values:
            if expected not in values_present:
                # Interpolate position
                # Find nearest known points
                lower = [(y, v) for y, v in points if v < expected]
                upper = [(y, v) for y, v in points if v > expected]

                if lower and upper:
                    y_lower, v_lower = lower[-1]
                    y_upper, v_upper = upper[0]
                    # Linear interpolation
                    fraction = (expected - v_lower) / (v_upper - v_lower)
                    y_interpolated = y_lower + fraction * (y_upper - y_lower)
                    complete_cal[y_interpolated] = float(expected)
                    print(f"  Interpolated {expected}% at Y={y_interpolated:.1f}")

        return complete_cal

    def pixel_to_value(self, pixel_y: float, calibration: Dict[float, float]) -> float:
        """Convert pixel Y to value using calibration."""
        if len(calibration) < 2:
            return 0.0

        cal_points = sorted(calibration.items())
        pixels = np.array([p for p, v in cal_points])
        values = np.array([v for p, v in cal_points])

        pixel_y = np.clip(pixel_y, pixels.min(), pixels.max())
        value = np.interp(pixel_y, pixels, values)
        return float(value)

    def detect_colors(self, plot_area: Tuple[int, int, int, int]) -> List[Tuple[int, int, int]]:
        """Detect dominant colors (BGR format)."""
        x_min, y_min, x_max, y_max = plot_area
        plot_img = self.image[y_min:y_max, x_min:x_max]

        # Sample
        sampled = plot_img[::8, ::8].reshape(-1, 3).astype(np.float32)

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
        kmeans.fit(sampled)

        colors = kmeans.cluster_centers_.astype(int)

        # Filter
        filtered = []
        for color in colors:
            b, g, r = color
            if min(r, g, b) > 220 or max(r, g, b) < 30:
                continue
            max_rgb, min_rgb = max(r, g, b), min(r, g, b)
            if (max_rgb - min_rgb) / (max_rgb + 1) < 0.15:
                continue
            filtered.append((int(b), int(g), int(r)))

        # Merge by hue
        merged = []
        for color in filtered:
            b, g, r = color
            is_similar = False
            for mb, mg, mr in merged:
                is_blue_1 = b > r * 1.15 and b > g * 1.15
                is_red_1 = r > b * 1.15 and r > g * 1.15
                is_blue_2 = mb > mr * 1.15 and mb > mg * 1.15
                is_red_2 = mr > mb * 1.15 and mr > mg * 1.15
                if (is_blue_1 and is_blue_2) or (is_red_1 and is_red_2):
                    if abs(r-mr) + abs(g-mg) + abs(b-mb) < 120:
                        is_similar = True
                        break
            if not is_similar:
                merged.append(color)

        return merged[:2]

    def extract_data(self) -> Dict:
        """Extract with grid-aligned calibration."""
        print(f"\nProcessing: {self.image_path}")
        print(f"Upscaled to: {self.width}x{self.height}")

        # Find labels with bounding boxes
        labels = self.find_yaxis_labels_with_boxes()

        if not labels:
            print("ERROR: No Y-axis labels found!")
            return {'data_points': [], 'calibration': {}, 'colors': [], 'metadata': self.metadata or {}}

        # Define plot area
        x_min = int(self.width * 0.12)
        y_min = int(self.height * 0.08)
        x_max = int(self.width * 0.95)
        y_max = int(self.height * 0.90)
        plot_area = (x_min, y_min, x_max, y_max)

        # Find gridlines
        gridlines = self.find_gridlines_in_plot(plot_area)

        # Match gridlines to labels
        calibration = self.match_gridlines_to_labels(labels, gridlines)

        # Interpolate missing values
        calibration = self.interpolate_calibration(calibration)

        print(f"\nFinal calibration with {len(calibration)} points:")
        for pixel_y, value in sorted(calibration.items()):
            print(f"  Y={pixel_y:.1f} → {value:.1f}%")

        # Detect colors
        print("\nDetecting series colors...")
        colors = self.detect_colors(plot_area)
        print(f"Detected {len(colors)} series:")
        for i, (b, g, r) in enumerate(colors):
            print(f"  Series {i}: RGB({r}, {g}, {b})")

        # Get countries
        countries = []
        if self.metadata and 'x_axis' in self.metadata:
            countries = self.metadata['x_axis'].get('countries', [])
            print(f"\nUsing {len(countries)} countries from metadata")

        # Extract data
        print("\nExtracting data with grid-aligned calibration...")
        data_points = []

        scan_width = x_max - x_min
        num_scans = min(400, scan_width // 3)

        for i in range(num_scans):
            x_center = x_min + int(i * scan_width / num_scans)

            for color_idx, (tb, tg, tr) in enumerate(colors):
                # Multi-sample for robustness
                samples = []
                for x_offset in range(-6, 7, 2):
                    x = x_center + x_offset
                    if x < x_min or x >= x_max:
                        continue

                    column = self.image[y_min:y_max, x]

                    # Find bar/point top
                    for y_offset, pixel in enumerate(column):
                        b, g, r = pixel
                        dist = abs(int(r)-int(tr)) + abs(int(g)-int(tg)) + abs(int(b)-int(tb))
                        if dist < 40:  # Tight threshold
                            samples.append(y_min + y_offset)
                            break

                if samples:
                    top_y = np.median(samples)
                    value = self.pixel_to_value(top_y, calibration)

                    # Map to country
                    country = None
                    if countries:
                        country_idx = int(i * len(countries) / num_scans)
                        if 0 <= country_idx < len(countries):
                            country = countries[country_idx]

                    # Filter valid range
                    y_min_val, y_max_val = 0, 100
                    if self.metadata and 'y_axis' in self.metadata:
                        y_range = self.metadata['y_axis'].get('range', [0, 100])
                        y_min_val, y_max_val = y_range

                    if y_min_val - 3 <= value <= y_max_val + 3:
                        data_points.append({
                            'x_position': i,
                            'country': country,
                            'value': round(value, 2),
                            'series': f'series_{color_idx}',
                            'color_r': int(tr),
                            'color_g': int(tg),
                            'color_b': int(tb)
                        })

        # Statistical filtering
        if data_points:
            df = pd.DataFrame(data_points)
            df['x_group'] = (df['x_position'] / 10).astype(int)

            grouped = df.groupby(['x_group', 'series', 'country'], dropna=False).agg({
                'x_position': 'first',
                'value': 'median',
                'color_r': 'first',
                'color_g': 'first',
                'color_b': 'first'
            }).reset_index()

            grouped = grouped.drop('x_group', axis=1)
            data_points = grouped.to_dict('records')

        print(f"Extracted {len(data_points)} data points")

        return {
            'data_points': data_points,
            'calibration': calibration,
            'colors': [(int(c[2]), int(c[1]), int(c[0])) for c in colors],
            'metadata': self.metadata or {}
        }

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
    """Test on Poland plot."""
    extractor = GridAlignedExtractor('9781484392935_f0013-01.jpg', 'plot_metadata.json')
    data = extractor.extract_data()

    # Save
    output_dir = Path('extracted_data_grid')
    output_dir.mkdir(exist_ok=True)
    extractor.save_to_csv(data, 'extracted_data_grid/9781484392935_f0013-01.csv')

    # Show Poland results
    df = pd.DataFrame(data['data_points'])
    if len(df) > 0:
        poland = df[df['country'] == 'POL'].sort_values('series')
        if not poland.empty:
            print(f"\n{'='*70}")
            print("POLAND RESULTS:")
            print("="*70)
            for _, row in poland.iterrows():
                is_blue = row['color_b'] > row['color_r']
                color_name = 'Blue (Revenue-neutral)' if is_blue else 'Red (Current CIT)'
                print(f"{color_name}: {row['value']:.2f}%")
            print("\nExpected from visual inspection:")
            print("  Blue: ~12-13%")
            print("  Red: ~19%")

            # Calculate error
            blue_vals = poland[poland['color_b'] > poland['color_r']]['value'].values
            if len(blue_vals) > 0:
                blue_avg = np.mean(blue_vals)
                blue_error = abs(blue_avg - 12.5)
                print(f"\nBlue average: {blue_avg:.2f}%")
                print(f"Blue error: {blue_error:.2f} percentage points")

            print("="*70)


if __name__ == '__main__':
    main()
