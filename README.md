# IMF Plot Data Extractor

⚠️ **IMPORTANT: Accuracy Limitations**

This tool uses **pixel-based computer vision** to extract data from plot images. While it automates the extraction process, the results are **APPROXIMATE** and should be verified for accuracy before use in analysis.

**Known Limitations:**
- Values may be off by several percentage points
- Mixed charts (bars + scatter) may produce multiple/incorrect detections
- Complex or overlapping plots reduce accuracy
- Works best for simple, well-separated bar charts
- **Manual verification strongly recommended**

**Best Used For:**
- Quick exploratory data extraction
- Getting approximate trends
- Identifying patterns across many plots

**NOT Recommended For:**
- Precise numeric analysis requiring exact values
- Publication or citation of extracted numbers
- Critical economic decisions

For precise values, consider manual extraction or OCR-based tools.

---

This tool extracts numeric data from plot images (PNG, JPG) using advanced image processing techniques.

## Features

- **Automatic plot type detection**: Detects bar charts, scatter plots, and combined visualizations
- **Color-based series extraction**: Identifies different data series by color
- **Rich metadata annotations**: Comprehensive plot information including titles, axis labels, series descriptions, and contextual notes
- **Multiple output formats**: Exports data to both CSV and JSON formats
- **Batch processing**: Process all plots in a directory at once
- **Pixel-based extraction**: Uses OpenCV and K-means clustering (approximate results - see limitations above)
- **Self-documenting data**: Extracted CSV/JSON files include metadata for immediate analysis

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Process all plots in the current directory

```bash
python extract_plot_data.py
```

This will:
- Scan the current directory for image files (.png, .jpg, .jpeg)
- Extract data from each plot
- Save results to the `extracted_data/` directory

### Process plots from a specific directory

```bash
python extract_plot_data.py --input /path/to/plots --output /path/to/output
```

### Process a single plot

```bash
python extract_plot_data.py --single plot_image.jpg --output results/
```

### Specify plot type

```bash
python extract_plot_data.py --plot-type bar    # For bar charts
python extract_plot_data.py --plot-type scatter # For scatter plots
python extract_plot_data.py --plot-type auto    # Auto-detect (default)
```

## Output Format

### CSV Format

For **bar charts**:
```csv
series,x_position,value,color_r,color_g,color_b,plot_title,plot_description,x_axis_label,y_axis_label
series_0,0,85.2,100,149,237,CIT and DBCFT Revenue by Country,Comparison of Corporate Income Tax...,Countries,Percent of GDP
series_0,1,82.1,100,149,237,CIT and DBCFT Revenue by Country,Comparison of Corporate Income Tax...,Countries,Percent of GDP
series_1,0,60.3,220,20,60,CIT and DBCFT Revenue by Country,Comparison of Corporate Income Tax...,Countries,Percent of GDP
```

For **scatter plots**:
```csv
series,x_value,y_value,color_r,color_g,color_b,plot_title,plot_description,x_axis_label,y_axis_label
series_0,25.5,68.3,100,149,237,Revenue Gain vs Loss,Scatter plot showing...,Trade Balance,Revenue Change
series_0,30.2,72.1,100,149,237,Revenue Gain vs Loss,Scatter plot showing...,Trade Balance,Revenue Change
series_1,15.8,45.2,220,20,60,Revenue Gain vs Loss,Scatter plot showing...,Trade Balance,Revenue Change
```

### JSON Format

```json
{
  "plot_type": "bar",
  "image_path": "plot.jpg",
  "plot_area": [50, 40, 450, 380],
  "data": {
    "series_0": {
      "x_positions": [0, 1, 2],
      "values": [85.2, 82.1, 78.5],
      "color": [100, 149, 237]
    }
  },
  "metadata": {
    "title": "CIT and DBCFT Revenue by Country",
    "description": "Comparison of Corporate Income Tax (CIT) revenue versus Destination-Based Cash Flow Tax (DBCFT) revenue across countries",
    "plot_type": "grouped_bar_chart",
    "x_axis": {
      "label": "Countries",
      "type": "categorical",
      "countries": ["USA", "MEX", "JPN", ...]
    },
    "y_axis": {
      "label": "Percent of GDP",
      "unit": "percent",
      "range": [-5, 15]
    },
    "series": {
      "CIT revenue": {
        "color": "blue",
        "description": "Corporate Income Tax revenue as percent of GDP"
      },
      "DBCFT revenue": {
        "color": "red",
        "description": "Destination-Based Cash Flow Tax revenue as percent of GDP"
      }
    },
    "notes": "Shows significant variation in tax revenue across countries..."
  }
}
```

## Metadata Annotations

The extraction system includes comprehensive metadata for all plots, manually curated by examining each image. The `plot_metadata.json` file contains:

### Metadata Fields

- **title**: Plot title
- **description**: Detailed description of what the plot shows
- **plot_type**: Specific type (e.g., "grouped_bar_chart", "scatter_plot", "multi_panel_time_series")
- **x_axis**: Label, type, unit, range, and categorical values (e.g., country lists)
- **y_axis**: Label, unit, and range
- **series**: Names, colors, and descriptions of each data series
- **reference_lines**: Information about trend lines, mean lines, or other reference markers
- **labeled_countries**: Country codes visible in the plot
- **notes**: Analytical insights and key observations

### Metadata Coverage

All 13 plots include metadata describing:
- **Economic concepts**: CIT (Corporate Income Tax), DBCFT (Destination-Based Cash Flow Tax), revenue neutrality, trade balance, Net IIP
- **Country data**: Lists of countries with ISO 3-letter codes where applicable
- **Axes and units**: Percent of GDP, ratios, US dollars, years
- **Contextual insights**: Trade deficit vs surplus effects, resource richness, development level

The metadata is automatically loaded and merged into extracted CSV and JSON outputs, making the data self-documenting.

## How It Works

1. **Plot Area Detection**: Identifies the main plotting region using edge detection
2. **Color Extraction**: Uses K-means clustering to identify dominant colors representing different data series
3. **Data Extraction**:
   - For bar charts: Measures bar heights and positions
   - For scatter plots: Detects point coordinates
4. **Normalization**: Converts pixel coordinates to normalized values (0-100 scale)
5. **Export**: Saves data in CSV and JSON formats

## Plot Types Supported

- Vertical bar charts (single or grouped)
- Horizontal bar charts
- Scatter plots
- Line plots with markers
- Combined bar and line plots

## Notes and Accuracy Considerations

⚠️ **Critical:** This tool provides APPROXIMATE values only. Extracted data should be verified against source plots before use.

**Accuracy Factors:**
- Values extracted via pixel measurement (not from labeled numbers)
- Measurement accuracy varies by plot complexity
- Simple bar charts: Better accuracy (~±2-5%)
- Complex/mixed charts: Lower accuracy (~±5-15% or more)
- Overlapping elements reduce accuracy significantly

**Recommendations:**
1. Always visually compare extracted data with source plots
2. Use this for exploratory analysis and trend identification
3. Verify critical values manually before publication
4. Consider this a starting point, not ground truth

**Technical Details:**
- Values initially extracted as percentages of plot pixel height (0-100 scale)
- Converted to actual y-axis values using metadata ranges
- Colors identified via RGB K-means clustering
- Best results with high-resolution, clear plot images

## Example Workflow

```bash
# 1. Extract data from all plots
python extract_plot_data.py

# 2. Check the extraction summary
cat extracted_data/extraction_summary.json

# 3. Load data in Python for analysis
import pandas as pd
data = pd.read_csv('extracted_data/plot_name.csv')
print(data)
```

## Troubleshooting

**Issue**: Colors not detected correctly
- **Solution**: The tool uses automatic color detection. If results are inaccurate, try adjusting the `color_tolerance` parameter in the code.

**Issue**: Plot area not detected correctly
- **Solution**: The tool uses heuristic boundaries. For non-standard plots, you may need to manually adjust the plot area detection logic.

**Issue**: Low accuracy for complex plots
- **Solution**: Ensure images are high resolution and have good contrast. Complex overlapping plots may require manual adjustment.

## Advanced Usage

You can also use the `PlotDataExtractor` class directly in Python:

```python
from extract_plot_data import PlotDataExtractor

# Load and extract
extractor = PlotDataExtractor('my_plot.jpg')
data = extractor.extract_data(plot_type='bar')

# Save results
extractor.save_to_csv(data, 'output.csv')
extractor.save_to_json(data, 'output.json')

# Access raw data
print(data['data'])
```

## License

This tool is provided as-is for extracting data from IMF economic plots.
