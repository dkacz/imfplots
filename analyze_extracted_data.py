#!/usr/bin/env python3
"""
Sample Analysis Script
Demonstrates how to load and analyze extracted plot data.
"""

import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path


def load_extracted_data(csv_path):
    """Load extracted data from CSV file."""
    return pd.read_csv(csv_path)


def load_json_data(json_path):
    """Load extracted data from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def analyze_bar_chart(csv_path):
    """Analyze bar chart data."""
    df = pd.read_csv(csv_path)

    print(f"\nAnalyzing: {csv_path}")
    print(f"Total data points: {len(df)}")
    print(f"Number of series: {df['series'].nunique()}")

    # Statistics by series
    print("\nStatistics by series:")
    for series in df['series'].unique():
        series_data = df[df['series'] == series]
        print(f"\n{series}:")
        print(f"  Count: {len(series_data)}")
        print(f"  Mean value: {series_data['value'].mean():.2f}")
        print(f"  Min value: {series_data['value'].min():.2f}")
        print(f"  Max value: {series_data['value'].max():.2f}")
        print(f"  Color (RGB): ({series_data['color_r'].iloc[0]}, "
              f"{series_data['color_g'].iloc[0]}, {series_data['color_b'].iloc[0]})")


def analyze_scatter_plot(csv_path):
    """Analyze scatter plot data."""
    df = pd.read_csv(csv_path)

    print(f"\nAnalyzing: {csv_path}")
    print(f"Total points: {len(df)}")
    print(f"Number of series: {df['series'].nunique()}")

    # Statistics by series
    print("\nStatistics by series:")
    for series in df['series'].unique():
        series_data = df[df['series'] == series]
        print(f"\n{series}:")
        print(f"  Points: {len(series_data)}")
        print(f"  X range: [{series_data['x_value'].min():.2f}, {series_data['x_value'].max():.2f}]")
        print(f"  Y range: [{series_data['y_value'].min():.2f}, {series_data['y_value'].max():.2f}]")
        print(f"  Color (RGB): ({series_data['color_r'].iloc[0]}, "
              f"{series_data['color_g'].iloc[0]}, {series_data['color_b'].iloc[0]})")


def recreate_plot(csv_path, output_path=None):
    """
    Recreate the plot from extracted data.

    Args:
        csv_path: Path to CSV file with extracted data
        output_path: Optional path to save the recreated plot
    """
    df = pd.read_csv(csv_path)

    # Determine plot type from columns
    if 'x_position' in df.columns and 'value' in df.columns:
        # Bar chart
        plt.figure(figsize=(12, 6))

        for series in df['series'].unique():
            series_data = df[df['series'] == series]
            color_rgb = (series_data['color_r'].iloc[0] / 255,
                        series_data['color_g'].iloc[0] / 255,
                        series_data['color_b'].iloc[0] / 255)

            plt.bar(series_data['x_position'], series_data['value'],
                   label=series, color=color_rgb, alpha=0.7)

        plt.xlabel('X Position')
        plt.ylabel('Value (%)')
        plt.title(f'Recreated Plot from {Path(csv_path).name}')
        plt.legend()
        plt.grid(True, alpha=0.3)

    elif 'x_value' in df.columns and 'y_value' in df.columns:
        # Scatter plot
        plt.figure(figsize=(10, 8))

        for series in df['series'].unique():
            series_data = df[df['series'] == series]
            color_rgb = (series_data['color_r'].iloc[0] / 255,
                        series_data['color_g'].iloc[0] / 255,
                        series_data['color_b'].iloc[0] / 255)

            plt.scatter(series_data['x_value'], series_data['y_value'],
                       label=series, color=color_rgb, alpha=0.6, s=50)

        plt.xlabel('X Value (%)')
        plt.ylabel('Y Value (%)')
        plt.title(f'Recreated Plot from {Path(csv_path).name}')
        plt.legend()
        plt.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def compare_series(csv_path):
    """Compare different series in the same plot."""
    df = pd.read_csv(csv_path)

    if 'value' in df.columns:
        # Bar chart comparison
        comparison = df.groupby('series')['value'].agg(['mean', 'std', 'min', 'max'])
        print(f"\nSeries comparison for {Path(csv_path).name}:")
        print(comparison)
    elif 'x_value' in df.columns:
        # Scatter plot comparison
        for series in df['series'].unique():
            series_data = df[df['series'] == series]
            print(f"\n{series}:")
            print(f"  X mean: {series_data['x_value'].mean():.2f}")
            print(f"  Y mean: {series_data['y_value'].mean():.2f}")


def export_for_excel(csv_path, output_path):
    """
    Export data in a format suitable for Excel analysis.

    Args:
        csv_path: Path to CSV file with extracted data
        output_path: Path to output Excel file
    """
    df = pd.read_csv(csv_path)

    # Create Excel writer
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Write raw data
        df.to_excel(writer, sheet_name='Raw Data', index=False)

        # Write summary statistics
        if 'value' in df.columns:
            summary = df.groupby('series')['value'].agg(['count', 'mean', 'std', 'min', 'max'])
            summary.to_excel(writer, sheet_name='Summary')

    print(f"Excel file saved to {output_path}")


def main():
    """Main demonstration function."""
    extracted_dir = Path('/home/user/imfplots/extracted_data')

    # Load summary
    with open(extracted_dir / 'extraction_summary.json', 'r') as f:
        summary = json.load(f)

    print("=" * 80)
    print("EXTRACTED DATA ANALYSIS")
    print("=" * 80)

    # Analyze first few plots
    for i, (filename, info) in enumerate(list(summary.items())[:3]):
        if info['status'] == 'success':
            csv_path = info['csv']

            if info['plot_type'] == 'bar':
                analyze_bar_chart(csv_path)
            elif info['plot_type'] == 'scatter':
                analyze_scatter_plot(csv_path)

            # Show comparison
            compare_series(csv_path)

            print("\n" + "-" * 80)

    # Demonstrate plot recreation
    print("\nRecreating plots...")
    output_dir = Path('/home/user/imfplots/recreated_plots')
    output_dir.mkdir(exist_ok=True)

    for i, (filename, info) in enumerate(list(summary.items())[:3]):
        if info['status'] == 'success':
            csv_path = info['csv']
            plot_name = Path(csv_path).stem
            output_path = output_dir / f"{plot_name}_recreated.png"

            try:
                recreate_plot(csv_path, str(output_path))
            except Exception as e:
                print(f"Error recreating {plot_name}: {e}")

    print(f"\nRecreated plots saved to {output_dir}/")

    # Show how to access specific data
    print("\n" + "=" * 80)
    print("EXAMPLE: Loading specific plot data")
    print("=" * 80)

    # Example with first bar chart
    first_bar = [info for info in summary.values()
                 if info['status'] == 'success' and info['plot_type'] == 'bar'][0]

    df = pd.read_csv(first_bar['csv'])
    print(f"\nLoaded data from {Path(first_bar['csv']).name}")
    print(f"Shape: {df.shape}")
    print(f"\nFirst 10 rows:")
    print(df.head(10))

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
