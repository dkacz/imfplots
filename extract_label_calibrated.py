#!/usr/bin/env python3
"""
Label-Calibrated Extraction
Uses OCR to find exact Y positions of axis labels (0, 20, 40, 60, 80, 100)
and calibrates based on those exact positions.
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


class LabelCalibratedExtractor:
    """Extract with calibration from actual Y-axis label positions."""

    def __init__(self, image_path: str, metadata_path: Optional[str] = None):
        """Initialize extractor."""
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load: {image_path}")

        # Super-resolution: upscale 4x
        print("Upscaling image 4x for precision...")
        self.image = cv2.resize(self.original_image, None, fx=4, fy=4,
                                interpolation=cv2.INTER_CUBIC)

        self.height, self.width = self.image.shape[:2]

        self.metadata = None
        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                all_metadata = json.load(f)
                image_name = Path(image_path).name
                self.metadata = all_metadata.get(image_name, {})

    def find_yaxis_labels(self) -> Dict[float, float]:
        """
        Use OCR to find Y-axis labels and their exact Y positions.
        Returns dict mapping pixel_y -> value (e.g., {150.5: 100.0, 300.2: 80.0, ...})
        """
        print("Finding Y-axis labels with OCR...")

        # Extract Y-axis region (left 15% of image)
        y_axis_width = int(self.width * 0.15)
        y_axis_region = self.image[:, :y_axis_width]

        # Preprocess for OCR
        gray = cv2.cvtColor(y_axis_region, cv2.COLOR_BGR2GRAY)

        # Apply threshold to get black text on white background
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary)

        # Use OCR with bounding boxes to get text positions
        ocr_data = pytesseract.image_to_data(Image.fromarray(denoised),
                                             config='--psm 6 -c tessedit_char_whitelist=0123456789',
                                             output_type=pytesseract.Output.DICT)

        # Extract numbers and their Y positions
        label_positions = {}

        for i in range(len(ocr_data['text'])):
            text = ocr_data['text'][i].strip()
            conf = int(ocr_data['conf'][i])

            if text and conf > 30:  # Reasonable confidence
                try:
                    value = float(text)

                    # Get the center Y position of this text
                    top = ocr_data['top'][i]
                    height = ocr_data['height'][i]
                    y_center = top + height / 2.0

                    # Only accept reasonable values
                    if self.metadata and 'y_axis' in self.metadata:
                        y_range = self.metadata['y_axis'].get('range', [0, 100])
                        if y_range[0] <= value <= y_range[1]:
                            label_positions[y_center] = value
                            print(f"  Found label '{value}' at Y={y_center:.1f}")
                    else:
                        # No metadata, accept 0-100 range
                        if 0 <= value <= 100:
                            label_positions[y_center] = value
                            print(f"  Found label '{value}' at Y={y_center:.1f}")
                except ValueError:
                    continue

        return label_positions

    def calibrate_from_labels(self, label_positions: Dict[float, float]) -> Dict[float, float]:
        """
        Create calibration mapping from label positions.
        Returns dict mapping pixel_y -> value for interpolation.
        """
        if not label_positions:
            return {}

        print(f"\nCalibration points from {len(label_positions)} labels:")
        calibration = {}
        for pixel_y, value in sorted(label_positions.items()):
            calibration[pixel_y] = value
            print(f"  Y={pixel_y:.1f} px → {value:.1f}")

        return calibration

    def pixel_to_value(self, pixel_y: float, calibration: Dict[float, float]) -> float:
        """Convert pixel Y to value using label-based calibration."""
        if len(calibration) < 2:
            return 0.0

        # Get sorted calibration points
        cal_points = sorted(calibration.items())
        pixels = np.array([p for p, v in cal_points])
        values = np.array([v for p, v in cal_points])

        # Clamp to range
        pixel_y = np.clip(pixel_y, pixels.min(), pixels.max())

        # Linear interpolation between calibration points
        value = np.interp(pixel_y, pixels, values)
        return float(value)

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

        return merged[:2]  # Limit to 2 series

    def extract_data(self) -> Dict:
        """Extract with label-based calibration."""
        print(f"\nProcessing: {self.image_path}")
        print(f"Upscaled to: {self.width}x{self.height}")

        # Find Y-axis labels with OCR
        label_positions = self.find_yaxis_labels()

        if not label_positions:
            print("ERROR: Could not find any Y-axis labels!")
            return {'data_points': [], 'calibration': {}, 'colors': [], 'metadata': self.metadata or {}}

        # Create calibration
        calibration = self.calibrate_from_labels(label_positions)

        # Define plot area
        x_min = int(self.width * 0.12)
        y_min = int(self.height * 0.08)
        x_max = int(self.width * 0.95)
        y_max = int(self.height * 0.90)
        plot_area = (x_min, y_min, x_max, y_max)

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
        print("\nExtracting data with label calibration...")
        data_points = []

        scan_width = x_max - x_min
        num_scans = min(300, scan_width // 3)

        for i in range(num_scans):
            x_center = x_min + int(i * scan_width / num_scans)

            for color_idx, (tb, tg, tr) in enumerate(colors):
                # Sample multiple columns for robustness
                samples = []
                for x_offset in range(-4, 5, 2):
                    x = x_center + x_offset
                    if x < x_min or x >= x_max:
                        continue

                    column = self.image[y_min:y_max, x]

                    # Find bar/point top
                    for y_offset, pixel in enumerate(column):
                        b, g, r = pixel
                        dist = abs(int(r)-int(tr)) + abs(int(g)-int(tg)) + abs(int(b)-int(tb))
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
    extractor = LabelCalibratedExtractor('9781484392935_f0013-01.jpg', 'plot_metadata.json')
    data = extractor.extract_data()

    # Save
    output_dir = Path('extracted_data_label')
    output_dir.mkdir(exist_ok=True)
    extractor.save_to_csv(data, 'extracted_data_label/9781484392935_f0013-01.csv')

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
