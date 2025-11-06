#!/usr/bin/env python3
"""
Ultra-Precise Plot Data Extraction
Maximum precision extraction using gridline calibration and super-resolution.
"""

import cv2
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class UltraPreciseExtractor:
    """Extract with maximum precision using advanced techniques."""

    def __init__(self, image_path: str, metadata_path: Optional[str] = None):
        """Initialize extractor."""
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load: {image_path}")

        # Super-resolution: upscale 4x using bicubic interpolation
        print("Upscaling image 4x for better precision...")
        self.image = cv2.resize(self.original_image, None, fx=4, fy=4,
                                interpolation=cv2.INTER_CUBIC)

        self.height, self.width = self.image.shape[:2]
        self.scale_factor = 4

        self.metadata = None
        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                all_metadata = json.load(f)
                image_name = Path(image_path).name
                self.metadata = all_metadata.get(image_name, {})

    def detect_gridlines_advanced(self, plot_area: Tuple[int, int, int, int]) -> List[Tuple[int, float]]:
        """
        Detect horizontal gridlines with sub-pixel precision.
        Returns list of (pixel_y, confidence) tuples.

        Args:
            plot_area: (x_min, y_min, x_max, y_max) to constrain search
        """
        x_min, y_min, x_max, y_max = plot_area

        # Extract plot region only
        plot_img = self.image[y_min:y_max, x_min:x_max]
        gray = cv2.cvtColor(plot_img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use Canny edge detection
        edges = cv2.Canny(blurred, 30, 100)

        # Detect lines using probabilistic Hough transform
        min_line_length = int((x_max - x_min) * 0.3)  # At least 30% of plot width
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=60,
                                minLineLength=min_line_length, maxLineGap=20)

        gridlines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Check if line is horizontal (small y difference, large x difference)
                if abs(y2 - y1) < 3 and abs(x2 - x1) > min_line_length:
                    y_pos = (y1 + y2) / 2.0  # Sub-pixel position relative to plot area
                    y_absolute = y_min + y_pos  # Convert to absolute image coordinates
                    length = abs(x2 - x1)
                    plot_width = x_max - x_min
                    confidence = length / plot_width  # Longer lines = more confident
                    gridlines.append((y_absolute, confidence))

        # Cluster nearby gridlines (within 5 pixels)
        if gridlines:
            gridlines.sort(key=lambda x: x[0])
            clustered = []
            current_cluster = [gridlines[0]]

            for gl in gridlines[1:]:
                if abs(gl[0] - current_cluster[0][0]) < 5:
                    current_cluster.append(gl)
                else:
                    # Take weighted average of cluster
                    total_conf = sum(c for _, c in current_cluster)
                    avg_y = sum(y * c for y, c in current_cluster) / total_conf
                    clustered.append((avg_y, total_conf / len(current_cluster)))
                    current_cluster = [gl]

            # Don't forget last cluster
            if current_cluster:
                total_conf = sum(c for _, c in current_cluster)
                avg_y = sum(y * c for y, c in current_cluster) / total_conf
                clustered.append((avg_y, total_conf / len(current_cluster)))

            gridlines = clustered

        # Sort by y position
        gridlines.sort(key=lambda x: x[0])

        return gridlines

    def calibrate_with_gridlines(self, gridlines: List[Tuple[int, float]]) -> Dict[float, float]:
        """
        Calibrate gridlines to y-axis values.
        Returns mapping of pixel_y -> value.
        """
        # Get expected y-axis values from metadata
        if not self.metadata or 'y_axis' not in self.metadata:
            return {}

        y_range = self.metadata['y_axis'].get('range', [])
        if len(y_range) != 2:
            return {}

        y_min, y_max = y_range

        # Expected gridline values (e.g., 0, 20, 40, 60, 80, 100)
        num_expected = 6  # Typically 5-6 gridlines
        step = (y_max - y_min) / (num_expected - 1)
        expected_values = [y_min + i * step for i in range(num_expected)]

        # Match detected gridlines to expected values
        # Gridlines should be evenly spaced from top to bottom
        if len(gridlines) >= 3:  # Need at least 3 to calibrate
            # Take the most confident gridlines
            gridlines = sorted(gridlines, key=lambda x: x[1], reverse=True)[:num_expected]
            gridlines = sorted(gridlines, key=lambda x: x[0])  # Re-sort by position

            # Map gridlines to values
            # In image coordinates: low Y = top of image = high values
            # High Y = bottom of image = low values
            calibration = {}
            if len(gridlines) == len(expected_values):
                # Perfect match - map first gridline (lowest Y, top) to max value
                for (pixel_y, _), value in zip(gridlines, reversed(expected_values)):
                    calibration[pixel_y] = value
            elif len(gridlines) >= 2:
                # Linear mapping from gridlines to expected values
                # First gridline (top, min pixel_y) = max value (y_max)
                # Last gridline (bottom, max pixel_y) = min value (y_min)
                pixel_positions = np.array([gl[0] for gl in gridlines])
                # Create values array in descending order (top to bottom)
                values_array = np.linspace(y_max, y_min, len(gridlines))

                for pixel_y, value in zip(pixel_positions, values_array):
                    calibration[pixel_y] = value

            return calibration

        return {}

    def pixel_to_value_calibrated(self, pixel_y: float, calibration: Dict[float, float],
                                  snap_threshold: float = 8.0) -> float:
        """
        Convert pixel to value using calibration points.
        Snaps to gridline if within threshold.
        """
        if not calibration:
            return 0.0

        # Check if pixel is very close to a calibration point
        for cal_pixel, cal_value in calibration.items():
            if abs(pixel_y - cal_pixel) < snap_threshold:
                return cal_value  # Snap to gridline value

        # Otherwise, interpolate between calibration points
        cal_points = sorted(calibration.items(), key=lambda x: x[0])
        pixels = np.array([p for p, v in cal_points])
        values = np.array([v for p, v in cal_points])

        # Clamp to range
        pixel_y = np.clip(pixel_y, pixels.min(), pixels.max())

        # Linear interpolation
        value = np.interp(pixel_y, pixels, values)
        return float(value)

    def detect_bar_top_multi_sample(self, x_start: int, x_end: int,
                                    target_color: Tuple[int, int, int],
                                    y_min: int, y_max: int) -> Optional[float]:
        """
        Detect bar top by sampling multiple columns and using median.
        More robust against noise.

        Args:
            x_start, x_end: X range to sample
            target_color: BGR color to detect
            y_min, y_max: Y range to search (plot area only)
        """
        tb, tg, tr = target_color

        top_positions = []

        # Sample every 2 pixels across the bar width
        for x in range(x_start, x_end, 2):
            if x >= self.width:
                break

            # Only search within plot area
            column = self.image[y_min:y_max, x]

            # Find pixels matching target color (scan from top to bottom)
            for y_offset in range(len(column)):
                pixel = column[y_offset]
                b, g, r = pixel

                # Color distance (use L1 distance for speed)
                dist = abs(int(r)-int(tr)) + abs(int(g)-int(tg)) + abs(int(b)-int(tb))

                if dist < 50:  # Tighter threshold
                    # Found the bar - record absolute y position
                    top_positions.append(float(y_min + y_offset))
                    break

        if not top_positions:
            return None

        # Use median to filter outliers
        return float(np.median(top_positions))

    def detect_colors(self, plot_area: Tuple[int, int, int, int]) -> List[Tuple[int, int, int]]:
        """Detect dominant colors in plot area (BGR format)."""
        x_min, y_min, x_max, y_max = plot_area
        plot_img = self.image[y_min:y_max, x_min:x_max]

        # Sample pixels (don't need all for color detection)
        sampled = plot_img[::4, ::4].reshape(-1, 3).astype(np.float32)

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
        kmeans.fit(sampled)

        colors = kmeans.cluster_centers_.astype(int)

        # Filter background colors
        filtered = []
        for color in colors:
            b, g, r = color

            # Skip white/very light
            if min(r, g, b) > 220:
                continue

            # Skip very dark
            if max(r, g, b) < 30:
                continue

            # Check saturation
            max_rgb = max(r, g, b)
            min_rgb = min(r, g, b)
            sat = (max_rgb - min_rgb) / (max_rgb + 1)

            if sat < 0.15:  # Low saturation = gray
                continue

            filtered.append((int(b), int(g), int(r)))

        # Merge similar colors by hue
        merged = []
        for color in filtered:
            b, g, r = color

            is_similar = False
            for mb, mg, mr in merged:
                # Check if same color family (blue/red/green)
                is_blue_1 = b > r * 1.15 and b > g * 1.15
                is_red_1 = r > b * 1.15 and r > g * 1.15
                is_blue_2 = mb > mr * 1.15 and mb > mg * 1.15
                is_red_2 = mr > mb * 1.15 and mr > mg * 1.15

                if (is_blue_1 and is_blue_2) or (is_red_1 and is_red_2):
                    dist = abs(r-mr) + abs(g-mg) + abs(b-mb)
                    if dist < 120:
                        is_similar = True
                        break

            if not is_similar:
                merged.append(color)

        return merged[:3]  # Limit to 3 series

    def extract_data(self) -> Dict:
        """Extract data with ultra-precision techniques."""
        print(f"Processing: {self.image_path}")
        print(f"Original: {self.original_image.shape[1]}x{self.original_image.shape[0]}")
        print(f"Upscaled: {self.width}x{self.height} ({self.scale_factor}x)")

        # Define plot area (scaled by 4)
        x_min = int(self.width * 0.12)
        y_min = int(self.height * 0.08)
        x_max = int(self.width * 0.95)
        y_max = int(self.height * 0.88)
        plot_area = (x_min, y_min, x_max, y_max)

        # Detect gridlines within plot area
        print("\nDetecting gridlines with sub-pixel precision...")
        gridlines = self.detect_gridlines_advanced(plot_area)
        print(f"Detected {len(gridlines)} gridlines")
        for i, (y, conf) in enumerate(gridlines):
            print(f"  Gridline {i+1}: y={y:.1f}, confidence={conf:.2f}")

        # Calibrate gridlines to values
        print("\nCalibrating gridlines to y-axis values...")
        calibration = self.calibrate_with_gridlines(gridlines)
        if calibration:
            print("Calibration points:")
            for pixel_y, value in sorted(calibration.items()):
                print(f"  Pixel {pixel_y:.1f} → {value:.1f}")
        else:
            print("  Warning: Could not calibrate gridlines - using metadata fallback")
            # Fallback: use plot area boundaries
            if self.metadata and 'y_axis' in self.metadata:
                y_range = self.metadata['y_axis'].get('range', [0, 100])
                y_min_val, y_max_val = y_range
                # Top of plot = max value, bottom = min value
                calibration = {
                    float(y_min): y_max_val,  # Top
                    float(y_max): y_min_val   # Bottom
                }
                print(f"  Using plot boundaries: top={y_max_val}, bottom={y_min_val}")

        # Detect colors
        print("\nDetecting series colors...")
        colors = self.detect_colors(plot_area)
        print(f"Detected {len(colors)} series:")
        for i, (b, g, r) in enumerate(colors):
            print(f"  Series {i}: RGB({r}, {g}, {b})")

        # Get countries from metadata
        countries = []
        if self.metadata and 'x_axis' in self.metadata:
            countries = self.metadata['x_axis'].get('countries', [])
            print(f"\nUsing {len(countries)} countries from metadata")

        # Extract data with multi-sampling
        print("\nExtracting data with multi-sampling...")
        data_points = []

        scan_width = x_max - x_min
        num_scans = min(300, scan_width // 4)  # More samples with upscaled image

        for i in range(num_scans):
            x_center = x_min + int(i * scan_width / num_scans)
            x_start = max(x_min, x_center - 4)
            x_end = min(x_max, x_center + 4)

            for color_idx, target_color in enumerate(colors):
                # Multi-sample detection within plot area
                top_y = self.detect_bar_top_multi_sample(x_start, x_end, target_color, y_min, y_max)

                if top_y is not None:
                    # Convert to value using calibration
                    value = self.pixel_to_value_calibrated(top_y, calibration, snap_threshold=8.0)

                    # Map to country
                    country = None
                    if countries:
                        country_idx = int(i * len(countries) / num_scans)
                        if 0 <= country_idx < len(countries):
                            country = countries[country_idx]

                    # Get expected range from metadata
                    y_min_val, y_max_val = 0, 100
                    if self.metadata and 'y_axis' in self.metadata:
                        y_range = self.metadata['y_axis'].get('range', [0, 100])
                        y_min_val, y_max_val = y_range

                    # Filter unreasonable values
                    if y_min_val - 5 <= value <= y_max_val + 5:
                        data_points.append({
                            'x_position': i,
                            'country': country,
                            'value': round(value, 2),
                            'series': f'series_{color_idx}',
                            'color_r': int(target_color[2]),
                            'color_g': int(target_color[1]),
                            'color_b': int(target_color[0])
                        })

        # Statistical filtering: group and take median
        if data_points:
            df = pd.DataFrame(data_points)
            df['x_group'] = (df['x_position'] / 8).astype(int)

            grouped = df.groupby(['x_group', 'series', 'country'], dropna=False).agg({
                'x_position': 'first',
                'value': 'median',  # Median filters outliers
                'color_r': 'first',
                'color_g': 'first',
                'color_b': 'first'
            }).reset_index()

            grouped = grouped.drop('x_group', axis=1)
            data_points = grouped.to_dict('records')

        print(f"Extracted {len(data_points)} data points")

        return {
            'data_points': data_points,
            'gridlines': [(y, c) for y, c in gridlines],
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
    """Test on Poland plot first."""
    extractor = UltraPreciseExtractor('9781484392935_f0013-01.jpg', 'plot_metadata.json')
    data = extractor.extract_data()

    # Save result
    output_dir = Path('extracted_data_ultra')
    output_dir.mkdir(exist_ok=True)
    extractor.save_to_csv(data, 'extracted_data_ultra/9781484392935_f0013-01.csv')

    # Show Poland results
    df = pd.DataFrame(data['data_points'])
    if len(df) > 0:
        poland = df[df['country'] == 'POL'].sort_values('series')
        if not poland.empty:
            print(f"\n{'='*60}")
            print("POLAND RESULTS:")
            print("="*60)
            for _, row in poland.iterrows():
                color_name = 'Blue (Revenue-neutral)' if row['color_b'] > row['color_r'] else 'Red (Current CIT)'
                print(f"{color_name}: {row['value']:.2f}%")
            print("\nExpected:")
            print("  Blue: ~12-13%")
            print("  Red: ~19%")
            print("="*60)


if __name__ == '__main__':
    main()
