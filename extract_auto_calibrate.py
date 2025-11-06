#!/usr/bin/env python3
"""
Auto-Calibrated Plot Extraction
Finds actual 0% and 100% lines by analyzing the bars themselves.
"""

import cv2
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class AutoCalibratedExtractor:
    """Extract with auto-calibration from actual bar positions."""

    def __init__(self, image_path: str, metadata_path: Optional[str] = None):
        """Initialize extractor."""
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load: {image_path}")

        # Super-resolution: upscale 4x
        print("Upscaling image 4x...")
        self.image = cv2.resize(self.original_image, None, fx=4, fy=4,
                                interpolation=cv2.INTER_CUBIC)

        self.height, self.width = self.image.shape[:2]

        self.metadata = None
        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                all_metadata = json.load(f)
                image_name = Path(image_path).name
                self.metadata = all_metadata.get(image_name, {})

    def auto_calibrate(self, plot_area: Tuple[int, int, int, int],
                       primary_color: Tuple[int, int, int]) -> Dict[str, float]:
        """
        Auto-calibrate by finding 0% baseline and estimating plot structure.

        Args:
            plot_area: Initial plot area estimate
            primary_color: Primary series color (BGR) to analyze

        Returns:
            Dict with 'baseline_y', 'top_y', 'y_min_val', 'y_max_val'
        """
        x_min, y_min, x_max, y_max = plot_area
        tb, tg, tr = primary_color

        print("Auto-calibrating from bar analysis...")

        # Find baseline (where bars end at the bottom) - this is 0%
        baseline_samples = []
        for scan_pos in range(5, 95, 3):  # Sample across entire width
            x = x_min + int(scan_pos * (x_max - x_min) / 100)
            column = self.image[y_min:y_max, x]

            # Find bottom of bars
            bar_pixels = []
            for y_offset, pixel in enumerate(column):
                b, g, r = pixel
                dist = abs(r-tr) + abs(g-tg) + abs(b-tb)
                if dist < 50:
                    bar_pixels.append(y_min + y_offset)

            if bar_pixels:
                baseline_samples.append(max(bar_pixels))

        calibration = {}

        if not baseline_samples:
            print("  Warning: Could not find baseline")
            return calibration

        baseline_y = np.median(baseline_samples)
        calibration['baseline_y'] = float(baseline_y)
        print(f"  Baseline (0%) at Y={baseline_y:.1f}")

        # Get value range from metadata
        y_min_val, y_max_val = 0, 100
        if self.metadata and 'y_axis' in self.metadata:
            y_range = self.metadata['y_axis'].get('range', [0, 100])
            y_min_val, y_max_val = y_range
            calibration['y_min_val'] = y_min_val
            calibration['y_max_val'] = y_max_val
            print(f"  Value range: {y_min_val} to {y_max_val}")

        # Estimate where 100% should be based on plot structure
        # Typically there's a small margin at top, so 100% is slightly below y_min
        # Use the baseline and assume the plot fills most of the vertical space

        # Method 1: Look for the top gridline by detecting horizontal lines
        gray = cv2.cvtColor(self.image[y_min:int(baseline_y), x_min:x_max], cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                                minLineLength=int((x_max-x_min)*0.3), maxLineGap=20)

        top_gridline = None
        if lines is not None:
            # Find the topmost horizontal line
            horizontal_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) < 3:  # Horizontal
                    y_pos = y_min + (y1 + y2) / 2.0
                    horizontal_lines.append(y_pos)

            if horizontal_lines:
                top_gridline = min(horizontal_lines)
                print(f"  Found top gridline at Y={top_gridline:.1f}")

        # Use top gridline if found, otherwise estimate
        if top_gridline:
            calibration['top_y'] = float(top_gridline)
            print(f"  Top (100%) at Y={top_gridline:.1f}")
        else:
            # Fallback: estimate based on typical plot proportions
            # Usually 5-10% margin at top
            plot_height = baseline_y - y_min
            estimated_top = y_min + plot_height * 0.05  # 5% margin
            calibration['top_y'] = float(estimated_top)
            print(f"  Estimated top (100%) at Y={estimated_top:.1f}")

        return calibration

    def pixel_to_value(self, pixel_y: float, calibration: Dict) -> float:
        """Convert pixel Y to value using calibration."""
        if 'baseline_y' not in calibration or 'top_y' not in calibration:
            return 0.0

        baseline_y = calibration['baseline_y']
        top_y = calibration['top_y']
        y_min_val = calibration.get('y_min_val', 0)
        y_max_val = calibration.get('y_max_val', 100)

        # Linear interpolation
        # pixel_y between top_y (100%) and baseline_y (0%)
        if baseline_y == top_y:
            return y_min_val

        normalized = (baseline_y - pixel_y) / (baseline_y - top_y)
        value = y_min_val + normalized * (y_max_val - y_min_val)

        return float(np.clip(value, y_min_val - 5, y_max_val + 5))

    def detect_colors(self, plot_area: Tuple[int, int, int, int]) -> List[Tuple[int, int, int]]:
        """Detect dominant colors (BGR format)."""
        x_min, y_min, x_max, y_max = plot_area
        plot_img = self.image[y_min:y_max, x_min:x_max]

        # Sample pixels
        sampled = plot_img[::4, ::4].reshape(-1, 3).astype(np.float32)

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
        kmeans.fit(sampled)

        colors = kmeans.cluster_centers_.astype(int)

        # Filter
        filtered = []
        for color in colors:
            b, g, r = color

            if min(r, g, b) > 220:  # Too light
                continue
            if max(r, g, b) < 30:  # Too dark
                continue

            max_rgb = max(r, g, b)
            min_rgb = min(r, g, b)
            sat = (max_rgb - min_rgb) / (max_rgb + 1)

            if sat < 0.15:  # Low saturation
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
                    dist = abs(r-mr) + abs(g-mg) + abs(b-mb)
                    if dist < 120:
                        is_similar = True
                        break

            if not is_similar:
                merged.append(color)

        return merged[:2]  # Limit to 2 series for cleaner results

    def extract_data(self) -> Dict:
        """Extract with auto-calibration."""
        print(f"\nProcessing: {self.image_path}")
        print(f"Upscaled to: {self.width}x{self.height}")

        # Initial plot area estimate
        x_min = int(self.width * 0.12)
        y_min = int(self.height * 0.08)
        x_max = int(self.width * 0.95)
        y_max = int(self.height * 0.88)
        plot_area = (x_min, y_min, x_max, y_max)

        # Detect colors
        print("\nDetecting series colors...")
        colors = self.detect_colors(plot_area)
        print(f"Detected {len(colors)} series:")
        for i, (b, g, r) in enumerate(colors):
            print(f"  Series {i}: RGB({r}, {g}, {b})")

        # Auto-calibrate using primary color (first/blue)
        primary_color = colors[0] if colors else (189, 132, 77)
        calibration = self.auto_calibrate(plot_area, primary_color)

        # Update plot area based on calibration
        if 'baseline_y' in calibration and 'top_y' in calibration:
            y_min = int(calibration['top_y'] - 20)  # Small margin above
            y_max = int(calibration['baseline_y'] + 20)  # Small margin below
            plot_area = (x_min, y_min, x_max, y_max)
            print(f"Adjusted plot area: Y from {y_min} to {y_max}")

        # Get countries
        countries = []
        if self.metadata and 'x_axis' in self.metadata:
            countries = self.metadata['x_axis'].get('countries', [])
            print(f"\nUsing {len(countries)} countries from metadata")

        # Extract data
        print("\nExtracting data with auto-calibration...")
        data_points = []

        scan_width = x_max - x_min
        num_scans = min(300, scan_width // 3)

        for i in range(num_scans):
            x_center = x_min + int(i * scan_width / num_scans)

            for color_idx, (tb, tg, tr) in enumerate(colors):
                # Sample multiple columns
                samples = []
                for x_offset in range(-4, 5, 2):
                    x = x_center + x_offset
                    if x < x_min or x >= x_max:
                        continue

                    column = self.image[y_min:y_max, x]

                    # Find bar top
                    for y_offset, pixel in enumerate(column):
                        b, g, r = pixel
                        dist = abs(r-tr) + abs(g-tg) + abs(b-tb)
                        if dist < 50:
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
                    y_min_val = calibration.get('y_min_val', 0)
                    y_max_val = calibration.get('y_max_val', 100)

                    if y_min_val - 2 <= value <= y_max_val + 2:
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
            df['x_group'] = (df['x_position'] / 8).astype(int)

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
    extractor = AutoCalibratedExtractor('9781484392935_f0013-01.jpg', 'plot_metadata.json')
    data = extractor.extract_data()

    # Save
    output_dir = Path('extracted_data_auto')
    output_dir.mkdir(exist_ok=True)
    extractor.save_to_csv(data, 'extracted_data_auto/9781484392935_f0013-01.csv')

    # Show Poland results
    df = pd.DataFrame(data['data_points'])
    if len(df) > 0:
        poland = df[df['country'] == 'POL'].sort_values('series')
        if not poland.empty:
            print(f"\n{'='*60}")
            print("POLAND RESULTS:")
            print("="*60)
            for _, row in poland.iterrows():
                is_blue = row['color_b'] > row['color_r']
                color_name = 'Blue (Revenue-neutral)' if is_blue else 'Red (Current CIT)'
                print(f"{color_name}: {row['value']:.2f}%")
            print("\nExpected:")
            print("  Blue: ~12-13%")
            print("  Red: ~19%")
            print("="*60)


if __name__ == '__main__':
    main()
